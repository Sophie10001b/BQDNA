import os
import torch

if __name__ == "__main__":
    dist = torch.randn((32, 1024, 256), dtype=torch.float)
    normalized_cost = (dist - dist.min()) / (dist.std() + 1e-6)
    cost = normalized_cost - normalized_cost.min()
    cost = cost.flatten(0, 1)
    mcost = cost.chunk(2)

    Q = torch.exp(-cost * 10).t() # (K, B)
    K, B = Q.shape

    mQ = [torch.exp(-_ * 10).t() for _ in mcost]
    mK, mB = mQ[0].shape
    mB *= 2

    sum_Q = Q.sum()
    Q /= (sum_Q + 1e-8)

    msum_Q = [_.sum() for _ in mQ]
    msum_Q = sum(msum_Q)
    mQ = [_ / (msum_Q + 1e-8) for _ in mQ]

    for _ in range(5):
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= (sum_of_rows + 1e-8)
        Q /= K

        msum_of_rows = [_.sum(dim=1, keepdim=True) for _ in mQ]
        msum_of_rows = sum(msum_of_rows)
        mQ = [_ / (msum_of_rows + 1e-8) / mK for _ in mQ]

        # normalize each column: total weight per sample must be 1/B
        Q /= (torch.sum(Q, dim=0, keepdim=True) + 1e-8)
        Q /= B

        mQ = [_ / (_.sum(dim=0, keepdim=True) + 1e-8) / mB for _ in mQ]
        pass

    Q *= B
    res = Q.t().argmax(-1)
    res = res.chunk(2)

    mQ = [_ * mB for _ in mQ]
    mres = [_.t().argmax(-1) for _ in mQ]

    pass