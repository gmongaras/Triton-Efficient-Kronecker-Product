import torch
import triton
import triton.language as tl


# x: [B, H, N, d]  -> out: [B, H, N, d(d+1)/2]
# Note that the kernel works over the entire batch and head dimension
# each kernel call works on a seperate block along the sequence and
# a single index for the output vector.
@triton.jit
def kron_kernel(x_ptr, out_ptr,
                    B, H, N,
                    d: tl.constexpr,
                    num_terms: tl.constexpr,
                    BLOCK_M: tl.constexpr):
    # program ids
    pid   = tl.program_id(0)                          # 0 .. num_terms-1  (upper-triangular term id)
    bid_n = tl.program_id(1)                          # tile along N (sequence) dimension
    off_hz = tl.program_id(2)                         # fused (B,H) index
    off_b  = off_hz // H
    off_h  = off_hz %  H

    # map pid -> (i,j) with i <= j (upper-triangular, row-major)
    # This maps an index we wanna calculate and converts it to
    # two indices we are going to multiply together. Since this
    # is a 2-degree polynomial, we are only dealing with
    # two indices at a time.
    acc = tl.zeros((), dtype=tl.int32)
    i   = tl.zeros((), dtype=tl.int32)
    skipping = tl.full((), True, dtype=tl.int1)
    for k in range(d):                                # unrolled
        count   = d - k
        new_acc = acc + count
        take    = skipping & (pid >= new_acc)
        acc     = tl.where(take, new_acc, acc)
        i       = tl.where(take, k + 1, i)
        skipping = skipping & (pid >= new_acc)
    j = i + (pid - acc)                               # column offset within row i

    # row offsets for this (b,h) and N tile
    n0   = bid_n * BLOCK_M
    offs_n = n0 + tl.arange(0, BLOCK_M)               # [BLOCK_M]
    mask_n = offs_n < N

    # base pointers
    # x layout:  [B, H, N, d]    -> linear idx = b*(H*N*d) + h*(N*d) + n*d + col
    # out layout:[B, H, N, T]    -> linear idx = b*(H*N*T) + h*(N*T) + n*T + pid
    base_x   = off_b * (H * N * d) + off_h * (N * d) + offs_n * d
    base_out = off_b * (H * N * num_terms) + off_h * (N * num_terms) + offs_n * num_terms

    # Load the two values to multiply together.
    xi = tl.load(x_ptr + base_x + i, mask=mask_n, other=0.0)
    xj = tl.load(x_ptr + base_x + j, mask=mask_n, other=0.0)

    # 1 coefficient if indices equal, √2 coefficient otherwise
    coef = tl.where(j == i, 1.0, tl.full((), 1.4142135623730951, dtype=tl.float32))  # √2 (fp32)
    val  = coef * xi * xj

    tl.store(out_ptr + base_out + pid, val, mask=mask_n)



def kron(x: torch.Tensor, block_m: int = 64):
    """
    Symmetric degree-2 feature map per (B,H,N) row with √2 on off-diagonals.
    x:   [B, H, N, d]
    out: [B, H, N, d(d+1)//2]
    Ensures that for each (b,h,n):  dot(phi(x), phi(y)) == (x·y)^2
    """
    assert x.ndim == 4, "x must be [B, H, N, d]"
    B, H, N, d = x.shape
    num_terms = d * (d + 1) // 2
    out = torch.empty((B, H, N, num_terms), device=x.device, dtype=x.dtype)

    grid = (num_terms, triton.cdiv(N, block_m), B * H)
    kron_kernel[grid](
        x, out, B, H, N,
        d=d, num_terms=num_terms, BLOCK_M=block_m
    )
    return out



if __name__ == "__main__":
    # Testing. This will likely be done with N=M=1 but whatever

    torch.manual_seed(0)
    B = 16
    H = 16
    N = 128
    M = 256
    d = 64
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    x = torch.rand(B, H, N, d, device=DEVICE)
    y = torch.rand(B, H, M, d, device=DEVICE)
    BLOCK_M = min(N, 64)
    output_triton = kron(x, BLOCK_M) @ kron(y, BLOCK_M).mT
    output_torch = (x @ y.mT)**2
    print(output_torch)
    print(output_triton)
    print(f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(output_torch - output_triton))}')


