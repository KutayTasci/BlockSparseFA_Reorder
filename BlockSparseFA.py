from typing import Optional
from utils import *

REPEATS_WARMUP = 10
REPEATS_TIMING = 25

def run_experiments_BSFA(
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
    q = torch.randn(batch_size, seq_len, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, nheads, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, nheads, d, device=device, dtype=dtype)

    # Flatten batch and sequence for unpadded format
    q_unpad_tmp = rearrange(q, "b s h d -> (b s) h d")
    k_unpad_tmp = rearrange(k, "b s h d -> (b s) h d")
    v_unpad_tmp = rearrange(v, "b s h d -> (b s) h d")

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

    for _ in range(REPEATS_WARMUP):
        #Reorder q, k, v if permutations are provided
        if row_perm is not None:
            q_unpad = apply_reordering(q_unpad_tmp, row_perm)
        else:
            q_unpad = q_unpad_tmp
        if col_perm is not None:
            k_unpad = apply_reordering(k_unpad_tmp, col_perm)
            v_unpad = apply_reordering(v_unpad_tmp, col_perm)
        else:
            k_unpad = k_unpad_tmp
            v_unpad = v_unpad_tmp
        out_unpad, sm_lse, S_dmask = block_sparse_attn_func(
            q_unpad, k_unpad, v_unpad,
            cu_seqlens_q, cu_seqlens_k,
            head_mask_type, streaming_info,
            block_mask,
            seq_len, seq_len,
            0.0,
            deterministic=True,
            softmax_scale=None,
            is_causal=False,
            exact_streaming=False,
            return_attn_probs=True,
        )
    torch.cuda.synchronize()

    # -----------------------------
    # Timing
    # -----------------------------
    start_ns = time.monotonic_ns()

    for _ in range(REPEATS_TIMING):
        #Reorder q, k, v if permutations are provided
        if row_perm is not None:
            q_unpad = apply_reordering(q_unpad_tmp, row_perm)
        if col_perm is not None:
            k_unpad = apply_reordering(k_unpad_tmp, col_perm)
            v_unpad = apply_reordering(v_unpad_tmp, col_perm)

        out_unpad, sm_lse, S_dmask = block_sparse_attn_func(
            q_unpad, k_unpad, v_unpad,
            cu_seqlens_q, cu_seqlens_k,
            head_mask_type, streaming_info,
            block_mask,
            seq_len, seq_len,
            0.0,
            deterministic=True,
            softmax_scale=None,
            is_causal=False,
            exact_streaming=False,
            return_attn_probs=True,
        )

        if row_perm is not  None:
            # Undo reordering of output
            out_unpad = reverse_reordering(out_unpad, row_perm)

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
        q_unpad = apply_reordering(q_unpad_tmp, row_perm)
    if col_perm is not None:
        k_unpad = apply_reordering(k_unpad_tmp, col_perm)
        v_unpad = apply_reordering(v_unpad_tmp, col_perm)
    for _ in range(REPEATS_WARMUP):
        

        out_unpad, sm_lse, S_dmask = block_sparse_attn_func(
            q_unpad, k_unpad, v_unpad,
            cu_seqlens_q, cu_seqlens_k,
            head_mask_type, streaming_info,
            block_mask,
            seq_len, seq_len,
            0.0,
            deterministic=True,
            softmax_scale=None,
            is_causal=False,
            exact_streaming=False,
            return_attn_probs=True,
        )
    torch.cuda.synchronize()

    # -----------------------------
    # Timing
    # -----------------------------
    start_ns = time.monotonic_ns()

    
        
    for _ in range(REPEATS_TIMING):
        
        
        out_unpad, sm_lse, S_dmask = block_sparse_attn_func(
            q_unpad, k_unpad, v_unpad,
            cu_seqlens_q, cu_seqlens_k,
            head_mask_type, streaming_info,
            block_mask,
            seq_len, seq_len,
            0.0,
            deterministic=True,
            softmax_scale=None,
            is_causal=False,
            exact_streaming=False,
            return_attn_probs=True,
        )
    torch.cuda.synchronize()

    end_ns = time.monotonic_ns()
    total_time_s = (end_ns - start_ns) / 1e9
    avg_time_ms = (total_time_s / REPEATS_TIMING) * 1e3

    print(f"Average time without reordering: {avg_time_ms:.3f} ms")