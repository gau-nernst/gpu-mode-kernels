#!POPCORN leaderboard amd-all2all
#!POPCORN gpu MI300x8

from typing import NamedTuple

import torch
import torch.distributed as dist
from task import input_t, output_t
from torch import Tensor


class DispatchAuxData(NamedTuple):
    recv_counts: list[int]
    send_counts: list[int]
    sort_indices1: Tensor
    sort_indices2: Tensor


def dispatch(dp_x: Tensor, indices: Tensor, num_local_experts: int, world_size: int):
    # dp_x: [num_tokens, hidden_dim]
    # indices: [num_tokens, topk]
    hidden_dim = dp_x.shape[1]
    topk = indices.shape[1]

    # 1. find how many tokens to send and receive from other ranks.
    # recv count = send count from other ranks
    # TODO: async all2all
    dst_ranks = indices // num_local_experts
    send_counts_t = dst_ranks.view(-1).bincount(minlength=world_size)
    recv_counts_t = send_counts_t.new_empty(world_size)
    dist.all_to_all_single(recv_counts_t, send_counts_t)

    # NOTE: this synchronizes
    recv_counts = recv_counts_t.tolist()
    send_counts = send_counts_t.tolist()

    # 2. sort tokens data (+metadata) by their destination for all2all.
    # we also sort by expert_id, which has a stronger order.
    sorted_expert_ids, sort_indices1 = indices.view(-1).sort()  # [num_tokens * topk]

    # tokens data
    total_recv = sum(recv_counts)
    send_buf = dp_x[sort_indices1 // topk]
    recv_buf = dp_x.new_empty(total_recv, hidden_dim)
    dist.all_to_all_single(recv_buf, send_buf, recv_counts, send_counts)

    # metadata
    send_meta = sorted_expert_ids % num_local_experts
    recv_meta = send_meta.new_empty(total_recv)
    dist.all_to_all_single(recv_meta, send_meta, recv_counts, send_counts)

    # 3. sort by local expert id for MoE
    # NOTE: this is already sorted within each source rank. exploit this.
    # NOTE: we are using packed/ragged layout for this
    # NOTE: since fake MoE is just * (rank + 1), even if this step is wrong, we wouldn't know
    local_expert_ids = recv_meta
    sort_indices2 = local_expert_ids.argsort()
    expert_x = recv_buf[sort_indices2]

    aux_data = DispatchAuxData(
        recv_counts=recv_counts,
        send_counts=send_counts,
        sort_indices1=sort_indices1,
        sort_indices2=sort_indices2,
    )

    return expert_x, aux_data


def combine(expert_y: Tensor, aux_data: DispatchAuxData, topk_weights: Tensor):
    # experts_y: [all_tokens, hidden_dim]
    # topk_weights: [num_tokens, topk]
    hidden_dim = expert_y.shape[-1]

    # NOTE: recv_counts from dispatch (how many to receive from other ranks) become
    # send_counts in combine (how many to send to other ranks)
    recv_counts = aux_data.send_counts
    send_counts = aux_data.recv_counts

    total_recv = sum(recv_counts)
    total_send = sum(send_counts)

    # 1. sort by original rank -> dst rank to send for all2all now.
    # invert the indices.
    #   if aux_data.sort_indices2[i] = j
    #   then sort_indices2[j] = i
    # TODO: implement custom scatter kernel to avoid explicit index inversion
    sort_indices2 = aux_data.sort_indices2.new_empty(total_send)
    sort_indices2[aux_data.sort_indices2] = torch.arange(total_send, device=expert_y.device)

    # tokens data
    # do scatter directly here
    send_buf = expert_y[sort_indices2]
    recv_buf = expert_y.new_empty(total_recv, hidden_dim)
    dist.all_to_all_single(recv_buf, send_buf, recv_counts, send_counts)

    # 2. sort by original position and topk ids i.e. flat indices
    sort_indices1 = aux_data.sort_indices1.new_empty(total_recv)
    sort_indices1[aux_data.sort_indices1] = torch.arange(total_recv, device=expert_y.device)

    # apply tok-k weighting
    # TODO: fused kernel for scatter mixed mm
    y = recv_buf[sort_indices1]
    y = y.view(*topk_weights.shape, hidden_dim)
    out_tokens = (y.float() * topk_weights.unsqueeze(-1)).sum(1).to(y.dtype)

    return out_tokens


def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data
    num_local_experts = cfg.num_experts // world_size

    # x: (num_tokens, hidden_dim)
    # indices: (num_tokens, experts_per_token) - map token_id to list of experts
    # weights: (num_tokens, experts_per_token) -> for weighted average later

    # scatter tokens to their corresponding DP rank
    expert_x, aux_data = dispatch(rank_data.x, rank_data.indices, num_local_experts, world_size)

    # local computation
    expert_y = expert_x.to(cfg.out_dtype) * (1 + rank)

    # gather tokens back
    y = combine(expert_y, aux_data, rank_data.weights)

    return y
