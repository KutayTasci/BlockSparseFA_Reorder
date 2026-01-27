#!/usr/bin/env python3
from __future__ import annotations

import argparse

from pathlib import Path

from utils import *
from BlockSparseFA import run_experiments_BSFA
from BlockSparseXF import run_experiments_BSFlexAttention


REPEATS_WARMUP = 10
REPEATS_TIMING = 25


def run_sparse_attention_reordering_test(args: Args) -> None:
    """
    Empty execution stub (wire your actual logic here).
    """

    rows_idx, cols_idx, seq_len = read_mtx(str(args.matrix_mtx))


    # Permutation shows new index for each old index
    # i.e., old index perm[i] goes to new index i
    if args.row_reorder is not None:
        row_perm = read_reordering_file_to_tensor(
            path=str(args.row_reorder),
            device="cuda:0",
            zero_based=None,
        )

        rows_idx = update_idx_perm(rows_idx, row_perm)

        #print the first 10 entries of the row_perm and rows_idx for debugging
        #print("row_perm:", row_perm[:10])
        #print("rows_idx:", rows_idx[:10])
        
    else:
        row_perm = None
    if args.col_reorder is not None:
        col_perm = read_reordering_file_to_tensor(
            path=str(args.col_reorder),
            device="cuda:0",
            zero_based=None,
        )
        cols_idx = update_idx_perm(cols_idx, col_perm)
    else:
        col_perm = None

    if args.mode == "BSFA":
        block_mask, seq_len = generate_blockmask_from_mtx_coords(
            rows_idx=rows_idx,
            cols_idx=cols_idx,
            seq_len=seq_len,
            block_size=args.block_size,
            batch_size=1,
            nheads=args.num_heads,
            device="cuda:0",
        )
    elif args.mode == "BSFlexAttention":
        block_mask, seq_len = generate_flex_blockmask_from_mtx_coords(
            rows_idx=rows_idx,
            cols_idx=cols_idx,
            seq_len=seq_len,
            block_size=args.block_size,
            device="cuda:0",
        )
    else:
        raise ValueError(f"Unsupported MODE: {args.mode}")
    
    if args.mode == "BSFlexAttention":
        run_experiments_BSFlexAttention(args, block_mask, seq_len, row_perm=row_perm, col_perm=col_perm)
    elif args.mode == "BSFA":
        run_experiments_BSFA(args, block_mask, seq_len, row_perm=row_perm, col_perm=col_perm)


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
    p.add_argument(
        "--mode",
        type=str,
        choices=["BSFA", "BSFlexAttention"],
        default="BSFA",
        help="Attention mode: BSFA or BSFlexAttention (default: BSFA).",
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
        mode=ns.mode,
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