#!POPCORN leaderboard amd-all2all
#!POPCORN gpu MI300x8

import torch
import torch.distributed as dist
from task import input_t, output_t
from torch import Tensor


def dispatch(dp_x: Tensor, indices: Tensor, num_local_experts: int, world_size: int, rank: int):
    # dp_x: [num_tokens, hidden_dim]
    # indices: [num_tokens, experts_per_token]
    device = dp_x.device
    hidden_dim = dp_x.shape[1]
    meta_dim = 5

    # 1. find how many tokens to send and receive from other ranks
    # TODO: study MoE kernels. usually they will sort the tokens -> naturally group
    # tokens together
    dst_ranks = indices // num_local_experts
    send_counts_t = dst_ranks.view(-1).bincount(minlength=world_size)

    # recv count = send count from other ranks
    # TODO: async all2all
    recv_counts_t = dp_x.new_empty(world_size, dtype=torch.long)
    dist.all_to_all_single(recv_counts_t, send_counts_t)

    # NOTE: this synchronizes
    recv_counts = recv_counts_t.tolist()
    send_counts = send_counts_t.tolist()

    # 2. prepare and exchange tokens (+metadata)
    token_map = [[] for _ in range(world_size)]  # list of tokens to send
    meta_map = [[] for _ in range(world_size)]  # metadata

    # TODO: remove loop
    # NOTE: a token might be sent twice to a DP rank if two experts reside on that rank
    # NOTE: we can send weight instead of expert_index (numerics will be a bit different)
    # NOTE: we can remove the last 0
    for src_pos, expert_list in enumerate(indices.tolist()):
        for k, expert_id in enumerate(expert_list):
            dst_rank = expert_id // num_local_experts
            token_map[dst_rank].append(src_pos)
            meta_map[dst_rank].append([expert_id, rank, src_pos, k, 0])

    # tokens data
    total_recv = sum(recv_counts)
    send_buf = torch.cat([dp_x[idx_list] for idx_list in token_map], dim=0)
    recv_buf = dp_x.new_empty(total_recv, hidden_dim)
    dist.all_to_all_single(
        recv_buf,
        send_buf,
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
    )

    send_meta = torch.tensor([v for sub in meta_map for v in sub], dtype=torch.int32, device=device)  # [..., 5]
    recv_meta = send_meta.new_empty(total_recv, meta_dim)
    dist.all_to_all_single(
        recv_meta.view(-1),
        send_meta.view(-1),
        output_split_sizes=[c * 5 for c in recv_counts],
        input_split_sizes=[c * 5 for c in send_counts],
    )

    # 3. group received tokens (+ metadata) belonging to the same expert together
    # NOTE: we can use packed/ragged layout for this
    expert_x = dp_x.new_empty(num_local_experts, total_recv, hidden_dim)
    expert_meta = send_meta.new_empty(num_local_experts, total_recv, meta_dim)

    expert_num_tokens = [0] * num_local_experts

    # convert global->local expert_id
    # D2H and convert to python once
    # only needed for Python optimization
    local_expert_ids = (recv_meta[:, 0] % num_local_experts).tolist()

    # TODO: remove this loop
    for i, local_expert in enumerate(local_expert_ids):
        pos = expert_num_tokens[local_expert]
        expert_x[local_expert, pos] = recv_buf[i]
        expert_meta[local_expert, pos] = recv_meta[i]
        expert_num_tokens[local_expert] += 1

    return expert_num_tokens, expert_x, expert_meta, send_counts, recv_counts


def combine(
    expert_y: Tensor,
    expert_meta: Tensor,
    expert_num_tokens: list,
    recv_counts: list[int],
    send_counts: list[int],
    topk_weights: Tensor,
):
    # experts_y: [num_local_experts, max_tokens, hidden_dim]
    # experts_meta: [num_local_experts, max_tokens, 5]
    # recv/send_counts: [world_size]
    # topk_weights: [num_tokens, topk]
    num_local_experts, _, hidden_dim = expert_y.shape
    world_size = len(recv_counts)
    meta_dim = 5

    # 1. prepare data for all2all
    # group tokens belonging to the same original rank together
    y_map = [[] for _ in range(world_size)]  # token to send back
    meta_map = [[] for _ in range(world_size)]  # metadata

    # TODO: remove this loop
    for local_eid in range(num_local_experts):
        cnt = int(expert_num_tokens[local_eid])
        for j in range(cnt):
            # meta info token j of local eid
            meta = expert_meta[local_eid, j]
            dst_rank = int(meta[1].item())  # original rank

            # token j and its meta that send back to dst rank/local eid
            y_map[dst_rank].append(expert_y[local_eid, j])
            meta_map[dst_rank].append(meta)

    # 2. call all2all
    # tokens data
    total_recv = sum(recv_counts)
    send_buf = torch.stack([v for sub in y_map for v in sub], dim=0)
    recv_buf = expert_y.new_empty(total_recv, hidden_dim)
    dist.all_to_all_single(
        recv_buf,
        send_buf,
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
    )

    # metadata
    send_meta = torch.stack([v for sub in meta_map for v in sub], dim=0)  # [..., 5]
    recv_meta = send_meta.new_empty(total_recv, meta_dim)
    dist.all_to_all_single(
        recv_meta.view(-1),
        send_meta.view(-1),
        output_split_sizes=[c * meta_dim for c in recv_counts],
        input_split_sizes=[c * meta_dim for c in send_counts],
    )

    # 3. rearrange data back to original arrangement
    # apply tok-k weighting
    out_tokens = expert_y.new_zeros(topk_weights.shape[0], hidden_dim)

    # TODO: remove this loop
    # this is scatter reduce
    recv_meta_py = recv_meta.tolist()
    for i in range(total_recv):
        src_token = recv_meta_py[i][2]
        src_k = recv_meta_py[i][3]

        # weighted sum can be fused (or sth)
        # NOTE: multiplication in FP32, but accumulation in out_dtype
        w = topk_weights[src_token, src_k].to(torch.float32)
        out_tokens[src_token] += recv_buf[i].to(torch.float32) * w

    return out_tokens


def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data
    num_local_experts = cfg.num_experts // world_size

    # x: (num_tokens, hidden_dim)
    # indices: (num_tokens, experts_per_token) - map token_id to list of experts
    # weights: (num_tokens, experts_per_token) -> for weighted average later

    # scatter tokens to their corresponding DP rank
    expert_num_tokens, expert_x, expert_meta, send_counts, recv_counts = dispatch(
        rank_data.x,
        rank_data.indices,
        num_local_experts=num_local_experts,
        world_size=world_size,
        rank=rank,
    )

    # local computation
    # TODO: for each expert, only need to compute for the first expert_num
    # OR, we can use ragged tensor format
    expert_y = expert_x.to(cfg.out_dtype) * (1 + rank)

    # gather tokens back
    # recv_counts from dispatch (how many to receive from other ranks) become
    # send_counts in combine (how many to send to other ranks)
    # and vice versa
    y = combine(
        expert_y,
        expert_meta,
        expert_num_tokens,
        recv_counts=send_counts,
        send_counts=recv_counts,
        topk_weights=rank_data.weights,
    )

    return y
