from __future__ import annotations
import torch
from einops import rearrange
from block_sparse_attn import block_sparse_attn_func
import time
import numpy as np
from torch.nn.attention.flex_attention import create_block_mask
from typing import Optional
from dataclasses import dataclass

from pathlib import Path


@dataclass(frozen=True)
class Args:
    matrix_mtx: Path
    block_size: int = 128
    head_dim: int = 128
    num_heads: int = 1
    row_reorder: Optional[Path] = None
    col_reorder: Optional[Path] = None
    mode : str = "BSFA"


def read_mtx(path: str) -> torch.Tensor:
    """
    Reads a sparse matrix from a Matrix Market (.mtx) file and returns it as a dense torch.Tensor.
    Assumes the matrix is in coordinate format (COO) with real values.

    Args:
        path (str): The path to the .mtx file.

    Returns:
        torch.Tensor: A dense tensor representation of the sparse matrix.
    """
    # print(f"Loading blockmask from {path} ...")
    # Read first line for seq_len check (optional)
    with open(path, 'r') as f:
        first_line = f.readline()
        rows, cols, nnz = map(int, first_line.strip().split())

    seq_len = rows  # assuming square matrix for attention
    # Load the remaining coordinates
    coords = np.loadtxt(path, skiprows=1, dtype=int)
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)
    rows_idx = coords[:, 0] - 1  # convert 1-based -> 0-based
    cols_idx = coords[:, 1] - 1
    return rows_idx, cols_idx, seq_len

def generate_blockmask_from_mtx_coords(
    rows_idx: np.ndarray,
    cols_idx: np.ndarray,
    seq_len: int,
    block_size: int,
    batch_size: int,
    nheads: int,
    device="cuda"
):
    """
    Generate block mask from a MTX file where:
      - First line: rows cols nnz
      - Remaining lines: row col (1-based, no values)
    """

    nrow = (seq_len + block_size - 1) // block_size
    ncol = (seq_len + block_size - 1) // block_size

    blockmask = torch.zeros(batch_size, nheads, nrow, ncol, dtype=torch.bool, device=device)

    row_blocks = rows_idx // block_size
    col_blocks = cols_idx // block_size

    # Clip in case some indices exceed nrow/ncol
    valid = (row_blocks < nrow) & (col_blocks < ncol)
    row_blocks = row_blocks[valid]
    col_blocks = col_blocks[valid]

    # Vectorized assignment for speed
    idx = torch.tensor(row_blocks * ncol + col_blocks, device=device)
    blockmask_flat = blockmask.view(batch_size, nheads, -1)
    blockmask_flat[:, :, idx] = True
    print(f"NNZ blocks: {blockmask.sum().item()}")
    return blockmask, seq_len

def generate_flex_blockmask_from_mtx_coords(
    rows_idx: np.ndarray,
    cols_idx: np.ndarray,
    seq_len: int,
    block_size: int,
    device: str = "cuda",
    *,
    compile_mask: bool = True,
): 
    """
    Build a FlexAttention BlockMask from an MTX file of allowed (row, col) coords.

    MTX format assumed:
      - First line: rows cols nnz
      - Remaining lines: row col (1-based, no values)
    Interprets entries as "allow attention from row -> col".

    Returns:
      block_mask: torch.nn.attention.flex_attention.BlockMask
      seq_len: int
      nblocks: int  (seq_len // block_size, rounded up)
    """

    nrow = (seq_len + block_size - 1) // block_size
    ncol = (seq_len + block_size - 1) // block_size

    allowed_blocks = torch.zeros((nrow, ncol), dtype=torch.bool, device=device)

    row_blocks = rows_idx // block_size
    col_blocks = cols_idx // block_size

    # Clip in case some indices exceed nrow/ncol
    valid = (row_blocks < nrow) & (col_blocks < ncol)
    row_blocks = row_blocks[valid]
    col_blocks = col_blocks[valid]
    allowed_blocks[row_blocks, col_blocks] = True

    def mask_mod(b, h, q_idx, kv_idx):
        qb = torch.div(q_idx, block_size, rounding_mode="floor")
        kb = torch.div(kv_idx, block_size, rounding_mode="floor")
        return allowed_blocks[qb, kb]
    
    block_mask = create_block_mask(
        mask_mod,
        B=1, H=1,
        Q_LEN=seq_len, KV_LEN=seq_len,
        device=device,
        BLOCK_SIZE=block_size,
        _compile=compile_mask,
    )

    return block_mask, seq_len

def read_reordering_file_to_tensor(
    path: str | Path,
    *,
    dtype: torch.dtype = torch.int64,
    device: torch.device | str = "cpu",
    zero_based: bool | None = None,
) -> torch.Tensor:
    """
    Reads a reordering file of the form:
        N
        idx_0
        idx_1
        ...
        idx_{N-1}

    Returns:
        perm: shape [N] torch tensor (dtype=int64 by default)

    Args:
        zero_based:
            - True  => treat indices as 0-based
            - False => treat indices as 1-based and convert to 0-based
            - None  => auto-detect: if min(idx) == 0 assume 0-based else 1-based

    // Permutation shows new index for each old index
    // i.e., old index perm[i] goes to new index i
    """
    path = Path(path)

    # Read all non-empty, non-comment lines
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)

    if not lines:
        raise ValueError(f"{path}: empty reordering file")

    try:
        n = int(lines[0])
    except ValueError as e:
        raise ValueError(f"{path}: first line must be an integer N") from e

    if n < 0:
        raise ValueError(f"{path}: N must be non-negative, got {n}")

    data_lines = lines[1:]
    if len(data_lines) != n:
        raise ValueError(
            f"{path}: expected {n} indices after first line, got {len(data_lines)}"
        )

    # Parse indices
    try:
        idx_list = [int(x) for x in data_lines]
    except ValueError as e:
        raise ValueError(f"{path}: all indices must be integers") from e

    perm = torch.tensor(idx_list, dtype=dtype, device=device)

    # Decide 0-based vs 1-based handling
    if n == 0:
        return perm  # empty

    if perm.numel() > 0:
        minv = int(perm.min().item())
        maxv = int(perm.max().item())
    else:
        minv = 0
        maxv = -1

    if zero_based is None:
        # Common heuristics: presence of 0 implies 0-based; otherwise assume 1-based.
        zero_based = (minv == 0)

    if not zero_based:
        perm = perm - 1
        minv = minv - 1
        maxv = maxv - 1

    # Basic validity checks for permutation-like content
    if minv < 0:
        raise ValueError(f"{path}: indices contain negatives after base adjustment (min={minv})")
    if maxv >= n:
        raise ValueError(f"{path}: index out of range after base adjustment (max={maxv}, N={n})")

    # Optional: enforce it's a proper permutation (no duplicates / missing)
    # Comment out if your files aren't guaranteed to be permutations.
    if torch.unique(perm).numel() != n:
        raise ValueError(f"{path}: indices are not a permutation (duplicates found)")

    return perm


def update_idx_perm(idx: np.ndarray, perm: torch.Tensor) -> np.ndarray:
    """
    Update indices according to the given permutation.

    Args:
        idx: numpy array of indices to be permuted
        perm: torch tensor representing the permutation

    Returns:
        numpy array of permuted indices
        # i.e., old index perm[i] goes to new index i
    """
    # Convert perm to numpy
    perm_np = perm.cpu().numpy()


    # Compute inverse permutation: inv_perm[old] = new
    inv_perm = np.empty_like(perm_np)
    inv_perm[perm_np] = np.arange(len(perm_np))


    # Apply inverse permutation to indices
    return inv_perm[idx]

def apply_reordering(x: torch.Tensor, perm: torch.Tensor, dim: int = 0) -> torch.Tensor:
    if perm.dtype != torch.int64:
        perm = perm.to(torch.int64)
    if perm.device != x.device:
        perm = perm.to(x.device)
    perm = perm.contiguous()
    return x.index_select(dim, perm)


def reverse_reordering(
    x: torch.Tensor,
    perm: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    """
    Reverse reordering of tensor x along dimension dim using permutation perm.

    Args:
        x: input tensor
        perm: permutation tensor of shape [N]
        dim: dimension along which to reverse the permutation

    Returns:
        Tensor with reversed reordering
    """
    # Create inverse permutation
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=perm.device)
    return x.index_select(dim, inv_perm)


    