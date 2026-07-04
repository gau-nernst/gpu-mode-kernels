#!POPCORN leaderboard qr_v2
#!POPCORN gpu B200

import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    bs, n, _ = data.shape

    A = data.clone()
    taus = data.new_empty(bs, n)

    for b_id in range(bs):
        vecs = []  # save householder vectors

        # skip last column
        for i in range(n - 1):
            x = A[b_id, i:, i].clone()

            # avoid degenerate case
            norm = torch.linalg.vector_norm(x)
            if norm == 0:
                vecs.append(None)
                taus[b_id, i] = 0.0
                continue

            # NOTE: signbit(x) * 2 - 1 is -1 when x >= 0, 1 otherwise
            target = (x[0].signbit() * 2 - 1) * norm
            x[0] -= target  # w = x - Hx
            x = x / x[0]  # v = w / w[0]
            vecs.append(x[1:])
            taus[b_id, i] = 2.0 / torch.linalg.vecdot(x, x)

            # update A with HA = A - tau v(v^TA)
            v = x @ A[b_id, i:, i:]  # v^T A
            A[b_id, i:, i:] -= taus[b_id, i] * x.view(-1, 1) * v

        # copy householder vectors to lapack result
        for i in range(n - 1):
            if vecs[i] is not None:
                A[b_id, i + 1 :, i] = vecs[i]

        taus[b_id, n - 1] = 0.0

    return A, taus
