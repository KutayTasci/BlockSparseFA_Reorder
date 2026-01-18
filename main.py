#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from utils import *


REPEATS_WARMUP = 10
REPEATS_TIMING = 25


@dataclass(frozen=True)
class Args:
    matrix_mtx: Path
    block_size: int = 128
    head_dim: int = 128
    num_heads: int = 1
    row_reorder: Optional[Path] = None
    col_reorder: Optional[Path] = None


def run_experiments(
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

    return


def run_sparse_attention_reordering_test(args: Args) -> None:
    """
    Empty execution stub (wire your actual logic here).
    """
    block_mask, seq_len = generate_blockmask_from_mtx_coords(
        mtx_path=str(args.matrix_mtx),
        block_size=args.block_size,
        batch_size=1,
        nheads=args.num_heads,
        device="cuda:0",
    )

    

    if args.row_reorder is not None:
        row_perm = read_reordering_file_to_tensor(
            path=str(args.row_reorder),
            device="cuda:0",
            zero_based=None,
        )
    else:
        row_perm = None
    if args.col_reorder is not None:
        col_perm = read_reordering_file_to_tensor(
            path=str(args.col_reorder),
            device="cuda:0",
            zero_based=None,
        )
    else:
        col_perm = None

    

    run_experiments(args, block_mask, seq_len, row_perm=row_perm, col_perm=col_perm)

    return


def _positive_int(x: str) -> int:
    v = int(x)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return v


def parse_args(argv: Optional[list[str]] = None) -> Args:
    p = argparse.ArgumentParser(
        description="Sparse attention matrix reordering test (MTX input)."
    )

    p.add_argument(
        "matrix_mtx",
        type=Path,
        help="Input sparse attention matrix in Matrix Market (.mtx) format.",
    )
    p.add_argument(
        "--block-size",
        type=_positive_int,
        default=128,
        help="Block size (default: 128).",
    )
    p.add_argument(
        "--head-dim",
        type=_positive_int,
        default=128,
        help="Head dimension (default: 128).",
    )
    p.add_argument(
        "--num-heads",
        type=_positive_int,
        default=1,
        help="Number of heads (default: 1). Values >1 are not supported in this version.",
    )
    p.add_argument(
        "--row-reorder",
        type=Path,
        default=None,
        help="Optional input file for row reorderings (default: None).",
    )
    p.add_argument(
        "--col-reorder",
        type=Path,
        default=None,
        help="Optional input file for column reorderings (default: None).",
    )

    ns = p.parse_args(argv)

    if ns.num_heads != 1:
        p.error("Only --num-heads=1 is supported in this version.")

    # Optional: basic extension check (not required, but handy)
    if ns.matrix_mtx.suffix.lower() != ".mtx":
        p.error("Input matrix file must have .mtx extension.")

    return Args(
        matrix_mtx=ns.matrix_mtx,
        block_size=ns.block_size,
        head_dim=ns.head_dim,
        num_heads=ns.num_heads,
        row_reorder=ns.row_reorder,
        col_reorder=ns.col_reorder,
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # Expose constants for your benchmark harness
    warmup = REPEATS_WARMUP
    timing = REPEATS_TIMING

    # If you want them inside args, you can also pass separately (kept simple here).
    _ = warmup, timing

    run_sparse_attention_reordering_test(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
