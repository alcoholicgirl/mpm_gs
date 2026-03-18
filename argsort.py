"""
GPU stable radix sort via Taichi.

LSD, 8 bits/pass, 4 passes → 32-bit keys.

Each pass runs four GPU kernels:
  1. _k_block_hist  – one thread per block; counts digits in its BLOCK elements
  2. _k_digit_total – one thread per digit; sums counts across all blocks
  3. _k_blk_pfx     – one thread per digit; builds per-(block,digit) start offset
  4. _k_scatter     – one thread per block; stable scatter (sequential within block)

Only the 256-entry digit total (1 KB) crosses the GPU↔CPU boundary per pass to
let the CPU compute the global exclusive prefix sum for each digit.
"""

import numpy as np
import taichi as ti

_BLOCK   = 512                               # elements per processing block
_RADIX   = 256                               # 2^8
_MAX_N   = 8 * 1024 * 1024                  # 8 M maximum elements
_MAX_BLK = (_MAX_N + _BLOCK - 1) // _BLOCK  # 16 384

_ready = False
_keys_a = _keys_b = _idx_a = _idx_b = None
_blk_hist = _blk_pfx = _digit_tot = _global_pfx = None


def _ensure():
    global _ready
    global _keys_a, _keys_b, _idx_a, _idx_b
    global _blk_hist, _blk_pfx, _digit_tot, _global_pfx
    if _ready:
        return
    _keys_a    = ti.field(ti.u32, shape=_MAX_N)
    _keys_b    = ti.field(ti.u32, shape=_MAX_N)
    _idx_a     = ti.field(ti.i32, shape=_MAX_N)
    _idx_b     = ti.field(ti.i32, shape=_MAX_N)
    _blk_hist  = ti.field(ti.i32, shape=_MAX_BLK * _RADIX)
    _blk_pfx   = ti.field(ti.i32, shape=_MAX_BLK * _RADIX)
    _digit_tot = ti.field(ti.i32, shape=_RADIX)
    _global_pfx= ti.field(ti.i32, shape=_RADIX)
    _ready = True


# ── kernels ───────────────────────────────────────────────────────────────────

@ti.kernel
def _k_load(src: ti.types.ndarray(ti.u32, ndim=1),
            dst: ti.template(), n: int):
    for i in range(n):
        dst[i] = src[i]


@ti.kernel
def _k_iota(dst: ti.template(), n: int):
    for i in range(n):
        dst[i] = i


@ti.kernel
def _k_block_hist(keys: ti.template(), hist: ti.template(),
                  n_blocks: int, n: int, shift: int):
    """Parallel over blocks; sequential within each block."""
    for b in range(n_blocks):
        base = b * _RADIX
        for d in range(_RADIX):
            hist[base + d] = 0
        start = b * _BLOCK
        end   = ti.min(start + _BLOCK, n)
        for i in range(start, end):
            digit = (keys[i] >> ti.u32(shift)) & ti.u32(0xFF)
            hist[base + digit] += 1


@ti.kernel
def _k_digit_total(hist: ti.template(), tot: ti.template(), n_blocks: int):
    """Parallel over digits; sums each digit's count across all blocks."""
    for d in range(_RADIX):
        s = 0
        for b in range(n_blocks):
            s += hist[b * _RADIX + d]
        tot[d] = s


@ti.kernel
def _k_global_pfx_scan(tot: ti.template(), gp: ti.template()):
    """Sequential exclusive prefix scan of 256 digit totals (single GPU thread)."""
    for _ in range(1):   # one iteration → inner loop is sequential
        running = 0
        for d in range(_RADIX):
            gp[d] = running
            running += tot[d]


@ti.kernel
def _k_blk_pfx(hist: ti.template(), pfx: ti.template(),
               gp: ti.template(), n_blocks: int):
    """Parallel over digits; builds per-(block,digit) exclusive start offset."""
    for d in range(_RADIX):
        running = gp[d]
        for b in range(n_blocks):
            pfx[b * _RADIX + d] = running
            running += hist[b * _RADIX + d]


@ti.kernel
def _k_scatter(keys_in:  ti.template(), keys_out: ti.template(),
               idx_in:   ti.template(), idx_out:  ti.template(),
               pfx:      ti.template(), n_blocks: int, n: int, shift: int):
    """Parallel over blocks; sequential within each block → stable."""
    for b in range(n_blocks):
        base  = b * _RADIX
        start = b * _BLOCK
        end   = ti.min(start + _BLOCK, n)
        for i in range(start, end):
            digit = (keys_in[i] >> ti.u32(shift)) & ti.u32(0xFF)
            pos   = pfx[base + digit]
            pfx[base + digit] += 1      # safe: only thread b writes to pfx[base + *]
            keys_out[pos] = keys_in[i]
            idx_out[pos]  = idx_in[i]


# ── key conversion ────────────────────────────────────────────────────────────

def _to_u32(a: np.ndarray) -> np.ndarray:
    """Map keys to uint32 that sort in the same ascending order."""
    if a.dtype == np.float64:
        a = a.astype(np.float32)
    if a.dtype == np.float32:
        u   = a.view(np.uint32).copy()
        neg = (u >> np.uint32(31)).astype(np.uint32)
        # positive float: flip sign bit only; negative float: flip all bits
        u  ^= neg * np.uint32(0x7FFFFFFF) + np.uint32(0x80000000)
        return u
    # integer types — assumed non-negative, fitting in uint32
    return a.astype(np.uint32)


# ── public API ────────────────────────────────────────────────────────────────

def argsort(a: np.ndarray, *, stable: bool = False) -> np.ndarray:
    """Return indices that sort *a* ascending (stable GPU radix sort, 32-bit keys)."""
    _ensure()
    n = len(a)
    if n > _MAX_N:
        raise ValueError(f"argsort: n={n} > _MAX_N={_MAX_N}")

    n_blocks = (n + _BLOCK - 1) // _BLOCK

    _k_load(_to_u32(a), _keys_a, n)
    _k_iota(_idx_a, n)

    src_k, dst_k = _keys_a, _keys_b
    src_i, dst_i = _idx_a,  _idx_b

    for shift in (0, 8, 16, 24):
        # 1. per-block digit histograms (GPU)
        _k_block_hist(src_k, _blk_hist, n_blocks, n, shift)
        # 2. per-digit totals then global exclusive prefix — all on GPU, no CPU sync
        _k_digit_total(_blk_hist, _digit_tot, n_blocks)
        _k_global_pfx_scan(_digit_tot, _global_pfx)
        # 3. per-(block,digit) start offsets (GPU)
        _k_blk_pfx(_blk_hist, _blk_pfx, _global_pfx, n_blocks)
        # 4. stable scatter (GPU)
        _k_scatter(src_k, dst_k, src_i, dst_i, _blk_pfx, n_blocks, n, shift)
        src_k, dst_k = dst_k, src_k
        src_i, dst_i = dst_i, src_i

    # 4 swaps → result is in original src (_idx_a)
    return src_i.to_numpy()[:n]
