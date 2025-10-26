#!POPCORN leaderboard amd-all2all
#!POPCORN gpu MI300x8

import torch
import torch.distributed as dist
from task import input_t, output_t
from torch import Tensor


def dispatch(dp_x: Tensor, indices: Tensor, num_local_experts: int, world_size: int, rank: int):
    # dp_x: [num_tokens, hidden_dim]
    # indices: [num_tokens, topk]
    hidden_dim = dp_x.shape[1]
    topk = indices.shape[1]

    # 1. find how many tokens to send and receive from other ranks.
    # recv count = send count from other ranks
    # TODO: async all2all
    dst_ranks = indices // num_local_experts
    send_counts_t = dst_ranks.view(-1).bincount(minlength=world_size)
    recv_counts_t = dp_x.new_empty(world_size, dtype=torch.long)
    dist.all_to_all_single(recv_counts_t, send_counts_t)

    # NOTE: this synchronizes
    recv_counts = recv_counts_t.tolist()
    send_counts = send_counts_t.tolist()

    # 2. sort tokens data (+metadata) by their destination for all2all.
    # we also sort by expert_id, which has a stronger order.
    flat_indices = indices.view(-1)
    sorted_expert_ids, sort_indices = flat_indices.sort()  # [num_tokens * topk]

    # tokens data
    total_recv = sum(recv_counts)
    send_buf = dp_x[sort_indices // topk]
    recv_buf = dp_x.new_empty(total_recv, hidden_dim)
    dist.all_to_all_single(
        recv_buf,
        send_buf,
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
    )

    # metadata
    send_meta = torch.stack(
        [sorted_expert_ids, torch.full_like(sorted_expert_ids, rank), sort_indices],
        dim=1,
    )
    meta_dim = send_meta.shape[1]
    recv_meta = send_meta.new_empty(total_recv, meta_dim)
    dist.all_to_all_single(
        recv_meta.view(-1),
        send_meta.view(-1),
        output_split_sizes=[c * meta_dim for c in recv_counts],
        input_split_sizes=[c * meta_dim for c in send_counts],
    )

    # 3. sort by local expert id for MoE
    # NOTE: this is already sorted within each source rank. exploit this.
    # NOTE: we are using packed/ragged layout for this
    # NOTE: since fake MoE is just * (rank + 1), even if this step is wrong, we wouldn't know
    local_expert_ids = recv_meta[:, 0] % num_local_experts
    sort_indices = local_expert_ids.argsort()
    expert_x = recv_buf[sort_indices]
    expert_meta = recv_meta[sort_indices]

    return expert_x, expert_meta, send_counts, recv_counts


def combine(
    expert_y: Tensor,
    expert_meta: Tensor,
    recv_counts: list[int],
    send_counts: list[int],
    topk_weights: Tensor,
):
    # experts_y: [all_tokens, hidden_dim]
    # experts_meta: [all_tokens, 3]
    # recv/send_counts: [world_size]
    # topk_weights: [num_tokens, topk]
    hidden_dim = expert_y.shape[-1]
    meta_dim = expert_meta.shape[-1]

    # 1. sort by original rank -> dst rank to send for all2all now.
    # TODO: we can invert the indices from dispatch
    sort_indices = expert_meta[:, 1].argsort()

    # tokens data
    total_recv = sum(recv_counts)
    send_buf = expert_y[sort_indices]
    recv_buf = expert_y.new_empty(total_recv, hidden_dim)
    dist.all_to_all_single(
        recv_buf,
        send_buf,
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
    )

    # metadata
    send_meta = expert_meta[sort_indices]
    recv_meta = send_meta.new_empty(total_recv, meta_dim)
    dist.all_to_all_single(
        recv_meta.view(-1),
        send_meta.view(-1),
        output_split_sizes=[c * meta_dim for c in recv_counts],
        input_split_sizes=[c * meta_dim for c in send_counts],
    )

    # 2. sort by original position and topk ids i.e. flat indices
    sort_indices = recv_meta[:, 2].argsort()
    y = recv_buf[sort_indices].view(*topk_weights.shape, hidden_dim)

    # apply tok-k weighting
    # TODO: fused kernel for mixed mm and gather
    out_tokens = (y.float() * topk_weights.unsqueeze(-1)).sum(1).to(y.dtype)

    return out_tokens


def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data
    num_local_experts = cfg.num_experts // world_size

    # x: (num_tokens, hidden_dim)
    # indices: (num_tokens, experts_per_token) - map token_id to list of experts
    # weights: (num_tokens, experts_per_token) -> for weighted average later

    # scatter tokens to their corresponding DP rank
    expert_x, expert_meta, send_counts, recv_counts = dispatch(
        rank_data.x,
        rank_data.indices,
        num_local_experts=num_local_experts,
        world_size=world_size,
        rank=rank,
    )

    # local computation
    expert_y = expert_x.to(cfg.out_dtype) * (1 + rank)

    # gather tokens back
    # recv_counts from dispatch (how many to receive from other ranks) become
    # send_counts in combine (how many to send to other ranks)
    # and vice versa
    y = combine(
        expert_y,
        expert_meta,
        recv_counts=send_counts,
        send_counts=recv_counts,
        topk_weights=rank_data.weights,
    )

    return y
