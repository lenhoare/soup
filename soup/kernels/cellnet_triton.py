"""Segmented per-genome cell-net kernel (M6, spec 5.4 "weight locality").

The eager path gathers every cell's own copy of its genome's weights (~4.7 KB per
cell per tick in fp32). Here live cells are sorted by genome slot and each genome's
run is cut into blocks of BLOCK cells. One Triton program handles one block: it
loads that genome's weights once and runs the 2-layer MLP for all its cells as two
small matrix multiplies:

    h   = tanh(X[BLOCK, IN] @ W1[IN, H] + b1)
    out = h[BLOCK, H] @ W2[H, OUT] + b2

IN, H and OUT are padded to powers of two (>= 16, as tl.dot requires). Weight traffic
falls from N cells x P params to n_blocks x P.

Runs on CUDA. For correctness checks without a GPU, set TRITON_INTERPRET=1 before
importing triton (tests do this); the interpreter is slow and only for testing.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # pragma: no cover - triton missing
    HAVE_TRITON = False


def _pow2(n: int, lo: int = 16) -> int:
    p = lo
    while p < n:
        p *= 2
    return p


if HAVE_TRITON:
    @triton.jit
    def _tanh(x):
        # tanh via exp (portable across backends and the interpreter)
        e = tl.exp(-2.0 * tl.abs(x))
        t = (1.0 - e) / (1.0 + e)
        return tl.where(x >= 0, t, -t)

    @triton.jit
    def _segmented_mlp(x_ptr, w_ptr, out_ptr, seg_gid, seg_start, seg_len,
                       stride_x, stride_w, stride_o,
                       IN: tl.constexpr, H: tl.constexpr, OUT: tl.constexpr,
                       IN_P: tl.constexpr, H_P: tl.constexpr, OUT_P: tl.constexpr,
                       BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        g = tl.load(seg_gid + pid).to(tl.int64)
        start = tl.load(seg_start + pid).to(tl.int64)
        n = tl.load(seg_len + pid)

        rows = tl.arange(0, BLOCK)
        rmask = rows < n
        ki = tl.arange(0, IN_P)
        kh = tl.arange(0, H_P)
        ko = tl.arange(0, OUT_P)

        # this block's inputs (cells already sorted by genome)
        x = tl.load(x_ptr + (start + rows)[:, None] * stride_x + ki[None, :],
                    mask=rmask[:, None] & (ki[None, :] < IN), other=0.0).to(tl.float32)

        # genome weights, loaded once per block. Flat layout: W1 (IN*H, row-major),
        # b1 (H), W2 (H*OUT, row-major), b2 (OUT), matching CellNetLayout.
        wbase = w_ptr + g * stride_w
        w1 = tl.load(wbase + ki[:, None] * H + kh[None, :],
                     mask=(ki[:, None] < IN) & (kh[None, :] < H), other=0.0).to(tl.float32)
        b1 = tl.load(wbase + IN * H + kh, mask=kh < H, other=0.0).to(tl.float32)
        o2 = IN * H + H
        w2 = tl.load(wbase + o2 + kh[:, None] * OUT + ko[None, :],
                     mask=(kh[:, None] < H) & (ko[None, :] < OUT), other=0.0).to(tl.float32)
        b2 = tl.load(wbase + o2 + H * OUT + ko, mask=ko < OUT, other=0.0).to(tl.float32)

        h = _tanh(tl.dot(x, w1) + b1[None, :])
        out = tl.dot(h, w2) + b2[None, :]
        tl.store(out_ptr + (start + rows)[:, None] * stride_o + ko[None, :], out,
                 mask=rmask[:, None] & (ko[None, :] < OUT))


def segments(gid_sorted: torch.Tensor, block: int):
    """Cut runs of equal genome id (in an already sorted gid vector) into blocks of at
    most `block` cells. Returns (seg_gid, seg_start, seg_len) as int32 tensors."""
    uniq, counts = torch.unique_consecutive(gid_sorted, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    nchunks = (counts + block - 1) // block
    seg_run = torch.repeat_interleave(torch.arange(len(uniq), device=gid_sorted.device), nchunks)
    first_chunk = torch.cumsum(nchunks, 0) - nchunks
    k = torch.arange(len(seg_run), device=gid_sorted.device) - first_chunk[seg_run]
    seg_start = starts[seg_run] + k * block
    seg_len = torch.clamp(counts[seg_run] - k * block, max=block)
    return uniq[seg_run].int(), seg_start.int(), seg_len.int()


def forward(layout, x: torch.Tensor, weights: torch.Tensor, gid: torch.Tensor,
            block: int = 32) -> torch.Tensor:
    """Drop-in for cellnet.forward: x (N, n_in), weights (S, P), gid (N,) -> (N, n_out)."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not installed")
    n = x.shape[0]
    out = torch.empty(n, layout.n_out, device=x.device, dtype=torch.float32)
    if n == 0:
        return out
    order = torch.argsort(gid)
    gs = gid[order]
    xs = x[order].contiguous()
    seg_gid, seg_start, seg_len = segments(gs, block)
    outs = torch.empty_like(out)
    w = weights.contiguous()
    _segmented_mlp[(seg_gid.numel(),)](
        xs, w, outs, seg_gid, seg_start, seg_len,
        xs.stride(0), w.stride(0), outs.stride(0),
        IN=layout.n_in, H=layout.hidden, OUT=layout.n_out,
        IN_P=_pow2(layout.n_in), H_P=_pow2(layout.hidden), OUT_P=_pow2(layout.n_out),
        BLOCK=block)
    out[order] = outs
    return out
