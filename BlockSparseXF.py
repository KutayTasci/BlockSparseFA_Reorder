from typing import Optional
from utils import *
from torch.nn.attention.flex_attention import flex_attention

REPEATS_WARMUP = 10
REPEATS_TIMING = 25


def run_experiments_BSFlexAttention(
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


    # -----------------------------
    # Block mask (all True)
    # -----------------------------
    nrow = (seq_len + block_size - 1) // block_size
    ncol = (seq_len + block_size - 1) // block_size



    # Compile flex attention so it doesn't materialize full scores
    flex_attn_compiled = torch.compile(flex_attention, fullgraph=True)
    # Measure time for sparse attention with reordering
    for _ in range(REPEATS_WARMUP):
        #Reorder q, k, v if permutations are provided
        if row_perm is not None:
            q_unpad = apply_reordering(q, row_perm, dim=2)
        else:
            q_unpad = q
        if col_perm is not None:
            k_unpad = apply_reordering(k, col_perm, dim=2)
            v_unpad = apply_reordering(v, col_perm, dim=2)
        else:
            k_unpad = k
            v_unpad = v
        out = flex_attn_compiled(
            q_unpad,
            k_unpad,
            v_unpad,
            block_mask=block_mask,
        )

        if row_perm is not None:
            out = reverse_reordering(out, row_perm, dim=2)

    torch.cuda.synchronize()


    start_ns = time.monotonic_ns()

    for _ in range(REPEATS_TIMING):
        #Reorder q, k, v if permutations are provided
        if row_perm is not None:
            q_unpad = apply_reordering(q, row_perm, dim=2)
        else:
            q_unpad = q
        if col_perm is not None:
            k_unpad = apply_reordering(k, col_perm, dim=2)
            v_unpad = apply_reordering(v, col_perm, dim=2)
        else:
            k_unpad = k
            v_unpad = v
        out = flex_attn_compiled(
            q_unpad,
            k_unpad,
            v_unpad,
            block_mask=block_mask,
        )

        if row_perm is not None:
            out = reverse_reordering(out, row_perm, dim=2)

    torch.cuda.synchronize()

    end_ns = time.monotonic_ns()
    total_time_s = (end_ns - start_ns) / 1e9
    avg_time_ms_with_reordering = (total_time_s / REPEATS_TIMING) * 1e3

    print(f"Average time with reordering: {avg_time_ms_with_reordering:.3f} ms")

    if row_perm is not None:
        q_unpad = apply_reordering(q, row_perm, dim=2)
    if col_perm is not None:
        k_unpad = apply_reordering(k, col_perm, dim=2)
        v_unpad = apply_reordering(v, col_perm, dim=2)
    for _ in range(REPEATS_WARMUP):
        out = flex_attn_compiled(
            q_unpad,
            k_unpad,
            v_unpad,
            block_mask=block_mask,
        )
    torch.cuda.synchronize()

    start_ns = time.monotonic_ns()

    for _ in range(REPEATS_TIMING):
        out = flex_attn_compiled(
            q_unpad,
            k_unpad,
            v_unpad,
            block_mask=block_mask,
        )

    torch.cuda.synchronize()
    
    end_ns = time.monotonic_ns()
    total_time_s = (end_ns - start_ns) / 1e9
    avg_time_ms = (total_time_s / REPEATS_TIMING) * 1e3

    print(f"Average time without reordering: {avg_time_ms:.3f} ms")
