# %% [markdown]
# # Large network (Exp 3): how close are the stored labels to a tighter equilibrium?

# %%
import os

os.environ.setdefault("AEQ_SHOW_PROGRESS", "FALSE")

import argparse
import glob
from dataclasses import fields

import numpy as np
import pandas as pd
import yaml

from ta import ROOT
from ta.network import Network
from ta.scenarios import ScenarioConfig, demand_rng, sample_demand
from ta.solver import solve
from ta.splits import load_set

# %% [markdown]
# ## Setup

# %%
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--config", default="configs/data_sydney.yaml")
parser.add_argument("--set", default="main")
parser.add_argument("--index", type=int, default=0)
parser.add_argument("--scenarios", type=int, default=12)
parser.add_argument("--reference-gap", type=float, default=1e-4)
parser.add_argument("--max-iter", type=int, default=3000)
parser.add_argument("--cores", type=int, default=os.cpu_count())
parser.add_argument("--summary", action="store_true")
args = parser.parse_known_args()[0]

config = yaml.safe_load((ROOT / args.config).read_text())
set_cfg = config["sets"][args.set]
NETWORK, SEED = config["network"], config["seed"]
names = {f.name for f in fields(ScenarioConfig)}
cfg = ScenarioConfig(**config["demand"], **{k: v for k, v in set_cfg.items() if k in names})
OUT = ROOT / "runs" / "results" / NETWORK / "exp3" / "label_accuracy"
OUT.mkdir(parents=True, exist_ok=True)

net = Network.from_tntp(NETWORK)
data, links, splits = load_set(NETWORK, args.set)
real = ~links.is_connector.to_numpy()


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


# %% [markdown]
# ## One test scenario per quantile of the iterations its label needed: solve it again to the tighter gap

# %%
if not args.summary:
    test = np.flatnonzero((splits.split == "test").to_numpy())
    ranked = test[np.argsort(data["iterations"][test], kind="stable")]
    r = int(ranked[int((args.index + 0.5) / args.scenarios * len(ranked))])
    closed = data["closed"][r]
    od = sample_demand(demand_rng(net.name, SEED, int(data["closure_set"][r]), int(data["variant"][r])), net.demand, cfg)[0]
    sol = solve(net.links[~closed].reset_index(drop=True), od, net.block_centroids, rgap_target=args.reference_gap,
                max_iter=args.max_iter, cores=args.cores)
    reference = np.zeros(net.num_links)
    reference[~closed] = sol.flows
    m = real & ~closed
    y = data["flows"][r]
    print(f"row {r}: label gap {data['rgap'][r]:.2e} ({data['iterations'][r]} iterations) | reference gap {sol.rgap:.2e} "
          f"({sol.iterations} iterations, {sol.runtime:.0f} s) | R² {r2(reference[m], y[m]):.5f} | MAE {np.abs(reference - y)[m].mean():.2f}", flush=True)
    np.savez_compressed(OUT / f"row_{args.index}.npz", row=r, reference=reference, rgap=sol.rgap, iterations=sol.iterations, runtime=sol.runtime)

# %% [markdown]
# ## Summary over the scenarios

# %%
if args.summary:
    rows, ys, ps = [], [], []
    for path in sorted(glob.glob(str(OUT / "row_*.npz"))):
        with np.load(path) as f:
            r, reference = int(f["row"]), f["reference"]
            m = real & ~data["closed"][r]
            y = data["flows"][r]
            rows.append({"row": r, "label_gap": data["rgap"][r], "label_iterations": data["iterations"][r], "reference_gap": float(f["rgap"]),
                         "reference_iterations": int(f["iterations"]), "r2": r2(reference[m], y[m]), "mae": np.abs(reference - y)[m].mean(),
                         "closed_share": data["closed"][r][real].mean(), "demand_scale": data["demand_scale"][r]})
            ys.append(reference[m])
            ps.append(y[m])
    table = pd.DataFrame(rows).sort_values("label_iterations")
    table.to_csv(OUT / "summary.csv", index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    y, p = np.concatenate(ys), np.concatenate(ps)
    print(f"pooled over {len(rows)} scenarios: R² {r2(y, p):.5f} | MAE {np.abs(y - p).mean():.2f} | per-scenario R² min {table.r2.min():.5f}, "
          f"median {table.r2.median():.5f} | reached reference gap {(table.reference_gap <= args.reference_gap).sum()}/{len(rows)}")
