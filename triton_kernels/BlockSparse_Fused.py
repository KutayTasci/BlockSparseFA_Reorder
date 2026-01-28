import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    desc_k, desc_v,  #
                    block_mask, H,
                    kv_perm,
                    offset_y, dtype: tl.constexpr, start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, warp_specialize: tl.constexpr, IS_HOPPER: tl.constexpr):
    # # range of values handled by this stage
    lo, hi = 0, N_CTX

    offsetk_y = offset_y + lo
    offsetv_y = offset_y + lo

    n_block = N_CTX // BLOCK_M

    m_off = ((tl.program_id(1) // H) * H) + (tl.program_id(1) % H)
    m_off = m_off * n_block * n_block
    m_off = m_off + (tl.program_id(0) * n_block)

    row_ids = tl.arange(0, BLOCK_M)


    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # Verify that the block should be calculated.
        b_mask_ptr = m_off + block_mask
        b_ind = tl.load(b_mask_ptr)
        # Triton does not allow the continue statement.
        if b_ind == True:
            # -- compute qk ----
            # off_m = offsetk_y + tl.arange(0, BLOCK_M)
            # off_k = tl.arange(0, HEAD_DIM)

            # k_ptrs = desc_k + (off_m[:, None] * HEAD_DIM) + off_k[None, :]

            pid_m  = tl.program_id(0)
            pid_bh = tl.program_id(1)

            b = pid_bh // H
            h = pid_bh % H

            # Block row IDs within N
            # row_ids = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            # row_ids = tl.arange(0, BLOCK_M)

            # Load permutation indices for this block
            perm_ptrs = (kv_perm
                    + b * H * N_CTX
                    + h * N_CTX
                    + row_ids)
            perm = tl.load(perm_ptrs)

            # Head dimension
            kte = tl.arange(0, HEAD_DIM)

            # Compute pointers into K using permuted row indices
            k_ptrs = (
                desc_k
                + b * H * N_CTX * HEAD_DIM
                + h * N_CTX * HEAD_DIM
                + perm[:, None] * HEAD_DIM
                + kte[None, :]
            )

            # Load K block with masking
            # k = tl.load(k_ptrs)  # shape [BLOCK_M, HEAD_DIM]

            k = tl.load(k_ptrs).T
            qk = tl.dot(q, k)
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = tl.math.exp2(qk)
            # -- compute correction factor
            alpha = tl.math.exp2(m_i - m_ij)
            l_ij = tl.sum(p, 1)
            # -- update output accumulator --
            acc = acc * alpha[:, None]
            # prepare p and v for the dot
            off_m = offsetv_y + tl.arange(0, BLOCK_M)
            off_k = tl.arange(0, HEAD_DIM)

            # v_ptrs = desc_v + (off_m[:, None] * HEAD_DIM) + off_k[None, :]
            # Compute pointers into K using permuted row indices
            v_ptrs = (
                desc_v
                + b * H * N_CTX * HEAD_DIM
                + h * N_CTX * HEAD_DIM
                + perm[:, None] * HEAD_DIM
                + kte[None, :]
            )

            v = tl.load(v_ptrs)
            p = p.to(dtype)
            # note that this non transposed v for FP8 is only supported on Blackwell
            acc = tl.dot(p, v, acc)
            # update m_i and l_i
            # place this at the end of the loop to reduce register pressure
            l_i = l_i * alpha + l_ij
            m_i = m_ij
        # These will always increment.
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
        m_off += 1
        row_ids += BLOCK_M
    return acc, l_i, m_i


@triton.jit
def _attn_fwd_block_sparse_perm(sm_scale, M,  #
              Z, H, desc_q, desc_k, desc_v, desc_o, 
              block_mask, N_CTX,  #
              q_perm,
              kv_perm,
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              warp_specialize: tl.constexpr,  #
              IS_HOPPER: tl.constexpr,  #
              ):
    dtype = tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    tl.multiple_of(qo_offset_y, BLOCK_M)
    tl.assume(HEAD_DIM % 16 == 0)

    # Calculate the position in the permutation matrix.
    # perm_off = ((tl.program_id(1) // H) * H) + (tl.program_id(1) % H)
    # perm_off = perm_off * N_CTX
    # perm_off = perm_off + (tl.program_id(0) * BLOCK_M)

    # # Read in the permutation indices.
    # q_perm_ptrs = q_perm + perm_off + tl.arange(0, BLOCK_M)
    # q_p_ind = tl.load(q_perm_ptrs)

    off_m = qo_offset_y + tl.arange(0, BLOCK_M)
    # off_te = qo_offset_y + q_p_ind
    off_k = tl.arange(0, HEAD_DIM)

    # q_ptrs = desc_q + (off_te[:, None] * HEAD_DIM) + off_k[None, :]
    # q = tl.load(q_ptrs)

    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    row_ids = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    perm_ptrs = (
        q_perm
        + b * H * N_CTX
        + h * N_CTX
        + row_ids
    )

    perm = tl.load(perm_ptrs)

    kte = tl.arange(0, HEAD_DIM)

    q_ptrs = (
        desc_q
        + b * H * N_CTX * HEAD_DIM
        + h * N_CTX * HEAD_DIM
        + perm[:, None] * HEAD_DIM
        + kte[None, :]
    )

    q = tl.load(q_ptrs)

    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        block_mask, H,
                                        kv_perm,
                                        offset_y, dtype, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX,  #
                                        warp_specialize, IS_HOPPER)

    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    # m_ptrs = M + off_hz * N_CTX + offs_m
    m_ptrs = M + (off_hz * N_CTX) + perm
    tl.store(m_ptrs, m_i)
    # desc_o.store([qo_offset_y, 0], acc.to(dtype))
    # o_ptrs = desc_o + (off_m[:, None] * HEAD_DIM) + off_k[None, :] + (perm[:, None] * HEAD_DIM)
    o_ptrs = (
        desc_o
        + b * H * N_CTX * HEAD_DIM
        + h * N_CTX * HEAD_DIM
        + perm[:, None] * HEAD_DIM
        + kte[None, :]
    )
    tl.store(o_ptrs, acc.to(dtype))