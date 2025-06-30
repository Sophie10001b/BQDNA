import os
import torch
import torch.nn as nn

import triton
import triton.language as tl

from functools import partial
from triton.testing import do_bench

@triton.jit
def row_normalization(
    tQ: tl.tensor, # [K, B],
    K: tl.constexpr,
    B: tl.constexpr,
    eps: tl.constexpr,
    REDUCE_BLK: tl.constexpr
):
    row = tl.program_id(0)

    tQ_ptr = tl.make_block_ptr(
        base=tQ,
        shape=(K, B),
        strides=(B, 1),
        offsets=(row, 0),
        block_shape=(1, REDUCE_BLK),
        order=(1, 0)
    )

    cache = tl.zeros((1,), dtype=tl.float32)
    tQ_tiled_ptr = tl.advance(tQ_ptr, (0, 0))

    for i in tl.range(0, B, REDUCE_BLK, num_stages=2):
        tQ_tiled_data = tl.load(tQ_tiled_ptr, boundary_check=(0,), padding_option="zero")
        cache += tl.sum(tQ_tiled_data, axis=1)
        tQ_tiled_ptr = tl.advance(tQ_tiled_ptr, (0, REDUCE_BLK))

    tQ_tiled_ptr = tl.advance(tQ_ptr, (0, 0))
    for i in tl.range(0, B, REDUCE_BLK, num_stages=2):
        tQ_tiled_data = tl.load(tQ_tiled_ptr, boundary_check=(0,), padding_option="zero")
        tQ_tiled_data = tQ_tiled_data / (cache * K + eps * K)
        tl.store(tQ_tiled_ptr, tQ_tiled_data, boundary_check=(0,))
        tQ_tiled_ptr = tl.advance(tQ_tiled_ptr, (0, REDUCE_BLK))

@triton.jit
def col_normalization(
    tQ: tl.tensor, # [K, B],
    K: tl.constexpr,
    B: tl.constexpr,
    eps: tl.constexpr,
    REDUCE_BLK: tl.constexpr
):
    col = tl.program_id(0)

    tQ_ptr = tl.make_block_ptr(
        base=tQ,
        shape=(K, B),
        strides=(B, 1),
        offsets=(0, col),
        block_shape=(REDUCE_BLK, 1),
        order=(1, 0)
    )

    cache = tl.zeros((1,), dtype=tl.float32)
    tQ_tiled_ptr = tl.advance(tQ_ptr, (0, 0))
    for i in tl.range(0, K, REDUCE_BLK, num_stages=2):
        tQ_tiled_data = tl.load(tQ_tiled_ptr, boundary_check=(1,), padding_option="zero")
        cache += tl.sum(tQ_tiled_data, axis=0)
        tQ_tiled_ptr = tl.advance(tQ_tiled_ptr, (REDUCE_BLK, 0))

    tQ_tiled_ptr = tl.advance(tQ_ptr, (0, 0))
    for i in tl.range(0, K, REDUCE_BLK, num_stages=2):
        tQ_tiled_data = tl.load(tQ_tiled_ptr, boundary_check=(1,), padding_option="zero")
        tQ_tiled_data = tQ_tiled_data / (cache * B + eps * B)
        tl.store(tQ_tiled_ptr, tQ_tiled_data, boundary_check=(1,))
        tQ_tiled_ptr = tl.advance(tQ_tiled_ptr, (REDUCE_BLK, 0))

def optimal_transport_tl(Q: torch.Tensor, n_iters: int=5, eps: float=1e-8):
    K, B = Q.shape
    Q = Q.float()
    REDUCE_BLK = 128

    for _ in range(n_iters):
        row_normalization[(K,)](Q, K, B, eps, REDUCE_BLK)
        col_normalization[(B,)](Q, K, B, eps, REDUCE_BLK)
    
    return Q


def optimal_transport_torch(Q: torch.Tensor, n_iters: int=5, eps: float=1e-8):
    K, B = Q.shape
    Q = Q.float()

    for _ in range(n_iters):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= (sum_of_rows + 1e-8)
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= (torch.sum(Q, dim=0, keepdim=True) + 1e-8)
        Q /= B
    
    return Q

if __name__ == '__main__':
    device = "cuda:0"
    Q = torch.randn(1023, 16385, dtype=torch.float, device=device)

    torch_res = optimal_transport_torch(Q)
    tl_res = optimal_transport_tl(Q)

    # test is correct
    assert torch.allclose(torch_res, tl_res, rtol=1e-05, atol=1e-05)

    # test time
    for _ in range(10, 16):
        Q = torch.randn(2 ** _, 16384, dtype=torch.float, device=device)

        torch_fun = partial(lambda : optimal_transport_torch(Q))
        tl_fun = partial(lambda : optimal_transport_tl(Q))

        torch_time = do_bench(torch_fun)
        tl_time = do_bench(tl_fun)

        print(f"loop {_}: torch time: {torch_time}, tl time: {tl_time}")