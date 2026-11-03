"""M6 Triton kernel: correctness vs eager, via the Triton CPU interpreter (runs in a
subprocess because TRITON_INTERPRET must be set before triton is imported)."""
import os
import subprocess
import sys

import pytest

pytest.importorskip("triton")

SCRIPT = r"""
import torch
from soup import cellnet
from soup.cellnet import CellNetLayout
from soup.config import load_config
from soup.kernels import cellnet_triton as K
from soup.world import World
torch.set_grad_enabled(False)
lay = CellNetLayout()
g = torch.Generator().manual_seed(0)
S = 40
w = torch.stack([cellnet.base_weights(lay, i) for i in range(S)]) + 0.1 * torch.randn(S, lay.n_params, generator=g)
gid = torch.repeat_interleave(torch.randperm(S, generator=g)[:15], torch.randint(1, 70, (15,), generator=g))
gid = gid[torch.randperm(len(gid), generator=g)]
x = torch.randn(len(gid), lay.n_in, generator=g)
for block, dt in ((16, torch.float32), (32, torch.float32), (32, torch.float16)):
    a = K.forward(lay, x, w.to(dt), gid, block=block)
    b = cellnet.forward(lay, x, w.to(dt), gid)
    tol = 1e-4 if dt == torch.float32 else 2e-2
    assert (a - b).abs().max() < tol, (block, dt, float((a - b).abs().max()))
# segments cover every cell exactly once, one genome per segment
gs = torch.sort(gid).values
sg, ss, sl = K.segments(gs, 32)
assert int(sl.sum()) == len(gid) and int(sl.max()) <= 32
for a_, b_, c_ in zip(sg.tolist(), ss.tolist(), sl.tolist()):
    assert (gs[b_:b_ + c_] == a_).all()
# whole-world: triton and eager worlds stay identical over a few ticks
cfg = dict(grid=[32, 32], genome_slots=256, device="cpu", init=dict(founders=8, genome_len=24), epoch_ticks=8)
A = World(load_config(None, {**cfg, "kernel": "eager"})); A.seed_founders(); A.run(12)
B = World(load_config(None, {**cfg, "kernel": "triton"})); B.seed_founders(); B.run(12)
assert B.kernel == "triton"
assert torch.equal(A.gid, B.gid), "worlds diverged"
print("OK")
"""


def test_triton_kernel_matches_eager():
    env = {**os.environ, "TRITON_INTERPRET": "1", "OMP_NUM_THREADS": "1"}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "-c", SCRIPT], cwd=root, env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout + r.stderr
