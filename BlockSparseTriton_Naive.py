from typing import Optional
from utils import *
from triton_kernels.BlockSparse_Naive import _attn_fwd_block_sparse
import math
import triton

REPEATS_WARMUP = 10
REPEATS_TIMING = 25

def run_experiments_BSTriton_Naive(
    args: Args,
    block_mask: torch.Tensor,
    seq_len: int,
    row_perm: Optional[torch.Tensor] = None,
    col_perm: Optional[torch.Tensor] = None,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float16,
):
    """
    Empty execution stub (wire your actual logic here).
    """
    torch.manual_seed(42)
    batch_size = 1
    nheads = args.num_heads
    d = args.head_dim
    block_size = args.block_size
    # -----------------------------
    # Generate query, key, value
    # -----------------------------
    q = torch.randn(batch_size, nheads, seq_len, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, nheads, seq_len, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, nheads, seq_len, d, device=device, dtype=dtype)

    # Create a softmax statistics tensor
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

    # Create an output tensor
    o = torch.empty_like(q)

    # Change the permutation shapes if they are provided
    if row_perm is not None:
        row_perm = row_perm.view(1, 1, q.shape[2], 1).expand(q.shape[0], q.shape[1], q.shape[2], q.shape[3])
    if col_perm is not None:
        col_perm = col_perm.view(1, 1, q.shape[2], 1).expand(q.shape[0], q.shape[1], q.shape[2], q.shape[3])

    # Set some parameters for the kernel
    BLOCK_M = 64 # 128, 64 is used by the original kernel's best config
    BLOCK_N = 64 # 128, 64 is used by the original kernel's best config
    num_warps = 4 # used by the original kernel's best config
    num_stages = 3 # used by the original kernel's best config
    sm_scale = (1.0 / math.sqrt(k.size(-1)))
    STAGE = 1 # Not causal.
    warp_specialize = False
    causal = False

    # Set the kernel launch grid
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)


    # -----------------------------
    # Cumulative sequence lengths
    # -----------------------------
    cu_seqlens_q = torch.arange(0, (batch_size + 1) * seq_len, step=seq_len, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.arange(0, (batch_size + 1) * seq_len, step=seq_len, dtype=torch.int32, device=device)
    
    
    
    # -----------------------------
    # Attention head types & streaming info
    # -----------------------------
    head_mask_type = torch.ones(nheads, device=device, dtype=torch.int32)  # full attention
    streaming_info = torch.tensor([0, 0] * nheads, device=device, dtype=torch.int32)
    
    # -----------------------------
    # Block mask (all True)
    # -----------------------------
    nrow = (seq_len + block_size - 1) // block_size
    ncol = (seq_len + block_size - 1) // block_size

    # Measure time for sparse attention with reordering

    # -----------------------------
    # Warm-up
    # -----------------------------
    torch.cuda.synchronize()
    for _ in range(REPEATS_WARMUP):
        #Reorder q, k, v if permutations are provided
        if row_perm is not None:
            q_unpad = torch.gather(q, dim = 2, index = row_perm)
        else:
            q_unpad = q
        if col_perm is not None:
            k_unpad = torch.gather(k, dim = 2, index = col_perm)
            v_unpad = torch.gather(v, dim = 2, index = col_perm)
        else:
            k_unpad = k
            v_unpad = v
        # Call the kernel.
        _attn_fwd_block_sparse[grid](
            sm_scale, M,  #
            q.shape[0], q.shape[1],  #
            q_unpad, k_unpad, v_unpad, o,  #
            block_mask,
            N_CTX=q.shape[2],  #
            HEAD_DIM=q.shape[-1],  #
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            STAGE=STAGE,  #
            warp_specialize=warp_specialize,  #
            IS_HOPPER=False,  #
            num_warps=4,
            num_stages=3
            )
        
        if row_perm is not  None:
            # Undo reordering of output
            out = torch.empty_like(o)
            out.scatter_(dim = 2, index = row_perm, src=o)

    torch.cuda.synchronize()


    # -----------------------------
    # Timing
    # -----------------------------
    torch.cuda.synchronize()
    start_ns = time.monotonic_ns()

    for _ in range(REPEATS_TIMING):
        #Reorder q, k, v if permutations are provided
        if row_perm is not None:
            q_unpad = torch.gather(q, dim = 2, index = row_perm)
        else:
            q_unpad = q
        if col_perm is not None:
            k_unpad = torch.gather(k, dim = 2, index = col_perm)
            v_unpad = torch.gather(v, dim = 2, index = col_perm)
        else:
            k_unpad = k
            v_unpad = v
        # Call the kernel.
        _attn_fwd_block_sparse[grid](
            sm_scale, M,  #
            q.shape[0], q.shape[1],  #
            q_unpad, k_unpad, v_unpad, o,  #
            block_mask,
            N_CTX=q.shape[2],  #
            HEAD_DIM=q.shape[-1],  #
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            STAGE=STAGE,  #
            warp_specialize=warp_specialize,  #
            IS_HOPPER=False,  #
            num_warps=4,
            num_stages=3
            )
        
        if row_perm is not  None:
            # Undo reordering of output
            out = torch.empty_like(o)
            out.scatter_(dim = 2, index = row_perm, src=o)

    torch.cuda.synchronize()

    end_ns = time.monotonic_ns()
    total_time_s = (end_ns - start_ns) / 1e9
    avg_time_ms_with_reordering = (total_time_s / REPEATS_TIMING) * 1e3

    print(f"Average time with reordering: {avg_time_ms_with_reordering:.3f} ms")


    # -----------------------------
    # Warm-up
    # -----------------------------

    #Reorder q, k, v if permutations are provided
    if row_perm is not None:
        q_unpad = torch.gather(q, dim = 2, index = row_perm)
    if col_perm is not None:
        k_unpad = torch.gather(k, dim = 2, index = col_perm)
        v_unpad = torch.gather(v, dim = 2, index = col_perm)
    torch.cuda.synchronize()
    for _ in range(REPEATS_WARMUP):
        _attn_fwd_block_sparse[grid](
            sm_scale, M,  #
            q.shape[0], q.shape[1],  #
            q_unpad, k_unpad, v_unpad, o,  #
            block_mask,
            N_CTX=q.shape[2],  #
            HEAD_DIM=q.shape[-1],  #
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            STAGE=STAGE,  #
            warp_specialize=warp_specialize,  #
            IS_HOPPER=False,  #
            num_warps=4,
            num_stages=3
            )
    torch.cuda.synchronize()

    # -----------------------------
    # Timing
    # -----------------------------
    torch.cuda.synchronize()
    start_ns = time.monotonic_ns()

    
        
    for _ in range(REPEATS_TIMING):
        _attn_fwd_block_sparse[grid](
            sm_scale, M,  #
            q.shape[0], q.shape[1],  #
            q_unpad, k_unpad, v_unpad, o,  #
            block_mask,
            N_CTX=q.shape[2],  #
            HEAD_DIM=q.shape[-1],  #
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            STAGE=STAGE,  #
            warp_specialize=warp_specialize,  #
            IS_HOPPER=False,  #
            num_warps=4,
            num_stages=3
            )
    torch.cuda.synchronize()

    end_ns = time.monotonic_ns()
    total_time_s = (end_ns - start_ns) / 1e9
    avg_time_ms = (total_time_s / REPEATS_TIMING) * 1e3

    print(f"Average time without reordering: {avg_time_ms:.3f} ms")