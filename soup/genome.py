"""Genome strings: creation, effective length and spore-time mutation.

A genome is a row of `genome_len` uint8 tokens in [0, alphabet). Token 0 is PAD.
Pads may sit anywhere, so deletions don't need to shift the array.
"""
from __future__ import annotations

import torch

PAD = 0


def random_genomes(n: int, length: int, alphabet: int, eff_len: int,
                   generator: torch.Generator, exclude: int | None = None) -> torch.Tensor:
    """n random genomes (on CPU) whose first `eff_len` tokens are non-pad. `exclude`
    (e.g. a STOP token) is never drawn, so founders are fully expressed."""
    if exclude is None:
        g = torch.randint(1, alphabet, (n, length), generator=generator, dtype=torch.uint8)
    else:
        allowed = torch.tensor([t for t in range(1, alphabet) if t != exclude],
                               dtype=torch.uint8)
        g = allowed[torch.randint(0, len(allowed), (n, length), generator=generator)]
    if eff_len < length:
        g[:, eff_len:] = PAD
    return g


def effective_length(genomes: torch.Tensor) -> torch.Tensor:
    return (genomes != PAD).sum(-1)


def _insert(row: torch.Tensor, pos: int, tok: int) -> torch.Tensor:
    """Insert `tok` at `pos`, shifting right. Absorbs the shift into the first pad
    at or after `pos`; if none, the last token falls off the end."""
    pads = (row[pos:] == PAD).nonzero()
    end = pos + int(pads[0]) if len(pads) else len(row) - 1
    out = row.clone()
    out[pos + 1:end + 1] = row[pos:end]
    out[pos] = tok
    return out


def mutate(genomes: torch.Tensor, p_sub: float, p_indel: float, p_dup: float,
           p_del: float, alphabet: int, generator: torch.Generator) -> torch.Tensor:
    """Mutated copies of a batch of genomes (CPU). Rates are per non-pad token."""
    g = genomes.clone()
    n, length = g.shape
    nonpad = g != PAD

    # Point substitution on non-pad positions (to another non-pad token).
    sub = (torch.rand(g.shape, generator=generator) < p_sub) & nonpad
    if sub.any():
        g[sub] = torch.randint(1, alphabet, (int(sub.sum()),), generator=generator,
                               dtype=torch.uint8)

    # Rare structural events: per-genome counts, applied in a small Python loop.
    L = nonpad.sum(-1).float()
    n_indel = torch.poisson(L * p_indel, generator=generator).long()
    n_dup = torch.poisson(L * p_dup, generator=generator).long()
    n_del = torch.poisson(L * p_del, generator=generator).long()
    busy = ((n_indel + n_dup + n_del) > 0).nonzero().flatten().tolist()

    def r(hi: int) -> int:
        return int(torch.randint(0, hi, (1,), generator=generator))

    for i in busy:
        row = g[i]
        for _ in range(int(n_indel[i])):
            if r(2):   # insertion
                row = _insert(row, r(length), r(alphabet - 1) + 1)
            else:      # single-token deletion
                row = row.clone()
                row[r(length)] = PAD
        for _ in range(int(n_dup[i])):   # segment duplication (overwrite at target)
            w = 2 + r(7)
            src, dst = r(length - w + 1), r(length - w + 1)
            row = row.clone()
            row[dst:dst + w] = row[src:src + w].clone()
        for _ in range(int(n_del[i])):   # segment deletion (pad out a window)
            w = 2 + r(7)
            s = r(length - w + 1)
            row = row.clone()
            row[s:s + w] = PAD
        g[i] = row
    return g
