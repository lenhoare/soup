"""Which heat idea makes fidelity (or redundancy) adapt? Each option alone on a
static mid-heat baseline, then all together.

    python -m experiments.heat_options --out runs/heat_options --ticks 8192 --seeds 1,2

Options (all off by default in soup/config.py):
  stop    ribosome.stop_token=15: translation stops at the first STOP (fragile code)
  power   heat.fid_power=3 with base_flip raised x8 so a typical organism (fidelity
          ~0.65) takes the same damage as baseline; only the spread grows (~16x)
  zones   fields.heat_max_wavenumber=1, heat_contrast=4: few, large, sharp zones
  direct  heat.state_noise=0.1, heat.energy_cost=0.01: heat acts on cells directly
"""
from __future__ import annotations

import copy

from experiments.heat_sweep import run_conditions

BASE = {"heat": {"base_flip": 0.03}, "fields": {"heat_drift": 0.0}}
OPTIONS = {
    "stop": {"ribosome": {"stop_token": 15}},
    "power": {"heat": {"fid_power": 3.0, "base_flip": 0.25}},
    "zones": {"fields": {"heat_max_wavenumber": 1, "heat_contrast": 4.0}},
    "direct": {"heat": {"state_noise": 0.1, "energy_cost": 0.01}},
}


def merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


CONDITIONS = {"baseline": BASE}
for name, opt in OPTIONS.items():
    CONDITIONS[name] = merge(BASE, opt)
ALL = BASE
for opt in OPTIONS.values():
    ALL = merge(ALL, opt)
CONDITIONS["all"] = ALL

SHOW = ["organisms", "shannon", "mean_len", "mean_expressed", "flips_per_org_epoch",
        "mean_fidelity", "fid_heat_corr", "fid_hot_minus_cold", "repeat_frac",
        "repeat_hot_minus_cold"]


# follow-ups: is STOP what blocks local adaptation in "all"? does junk build up
# behind the STOP even without heat?
CONDITIONS["all_but_stop"] = merge(merge(merge(BASE, OPTIONS["power"]), OPTIONS["zones"]),
                                   OPTIONS["direct"])
CONDITIONS["stop_noheat"] = merge(merge(BASE, OPTIONS["stop"]), {"features": {"heat": False}})

if __name__ == "__main__":
    run_conditions(CONDITIONS, SHOW, "runs/heat_options")
