# %% [markdown]
# # Solver speed / accuracy curve

# %%
import os

for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(var, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ta import ROOT
from ta.metrics import evaluate, select
from ta.network import Network, solve_scenario
from ta.splits import load_set

NETWORK = "Berlin-Friedrichshain"
ITERATIONS = [1, 2, 3, 5, 10, 20, 50, 100]
OUT = ROOT / "runs" / "baselines" / NETWORK

# %% [markdown]
# ## One truncated solve (in a worker process)

# %%
WORKER = {}


def init_worker(network):
    WORKER["net"] = Network.from_tntp(network)


def truncated(args):
    demand, closed, k = args
    sol, _ = solve_scenario(WORKER["net"], demand, closed, rgap_target=1e-12, max_iter=k, retries=1)
    return sol.flows, sol.runtime, sol.rgap


# %% [markdown]
# ## Curve for one dataset


# %%
def curve(name, workers):
    data, links, splits = load_set(NETWORK, name)
    conn, grid = links.is_connector.to_numpy(), splits.gridlock.to_numpy()
    test = select(splits, "test")
    tasks = [(data["demand"][r], data["closed"][r], k) for k in ITERATIONS for r in test]
    with ProcessPoolExecutor(workers, initializer=init_worker, initargs=(NETWORK,)) as pool:
        results = list(pool.map(truncated, tasks, chunksize=8))
    rows = []
    for j, k in enumerate(ITERATIONS):
        chunk = results[j * len(test):(j + 1) * len(test)]
        pred = np.stack([c[0] for c in chunk])
        table = evaluate(pred, data, links, conn, test, grid)
        real = table.xs("real", level="links")
        rows.append({"iterations": k, "sec_median": np.median([c[1] for c in chunk]), "rgap_median": np.median([c[2] for c in chunk]),
                     "r2_all": real.loc["all", "r2"], "r2_no_gridlock": real.loc["no_gridlock", "r2"],
                     "r2_gridlock": real.loc["gridlock", "r2"], "mae": real.loc["all", "mae"], "wmape": real.loc["all", "wmape"],
                     "conservation": real.loc["all", "conservation"]})
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / f"solver_curve_{name}.csv", index=False)
    return out, float(np.median(data["runtime"][test]))


# %% [markdown]
# ## Run

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--sets", nargs="+", default=["main", "old_like"])
    parser.add_argument("--workers", type=int, default=min(28, os.cpu_count()))
    args = parser.parse_known_args()[0]
    curves, full_time = {}, {}
    for name in args.sets:
        curves[name], full_time[name] = curve(name, args.workers)
        print(f"== {name} (full equilibrium solve, gap 1e-5: median {full_time[name]:.2f} s)")
        print(curves[name].to_string(index=False, float_format=lambda v: f"{v:.3g}"))

# %% [markdown]
# ## Figure: accuracy vs time, with the no-learning baselines from step 2.2

# %%
if __name__ == "__main__":
    fig, axes = plt.subplots(1, len(curves), figsize=(6.5 * len(curves), 4.8), facecolor="white", squeeze=False)
    for ax, (name, c) in zip(axes[0], curves.items()):
        ax.plot(c.sec_median, c.r2_all, "o-", color="#2e86c1", label="solver stopped after k iterations")
        for _, r in c.iterrows():
            ax.annotate(f"k={int(r.iterations)}", (r.sec_median, r.r2_all), xytext=(4, -10), textcoords="offset points", fontsize=7)
        ax.scatter([full_time[name]], [1.0], color="black", zorder=5, label="full solve (gap 1e-5)")
        base = pd.read_csv(OUT / f"{name}.csv")
        base = base[(base.links == "real") & (base.subset == "all") & base.evaluation.str.startswith("test")]
        for _, r in base.iterrows():
            ax.scatter([max(r.sec_per_scenario, 1e-5)], [r.r2], marker="s", s=30, zorder=4)
            ax.annotate(r.baseline, (max(r.sec_per_scenario, 1e-5), r.r2), xytext=(4, 3), textcoords="offset points", fontsize=7)
        ax.set_xscale("log"); ax.set_xlabel("seconds per scenario (single core, log)"); ax.set_ylabel("R² on real links (test)")
        ax.set_title(f"{name}: speed vs accuracy"); ax.set_ylim(0.3, 1.02); ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "runs" / "figures" / "solver_curve.png", dpi=130)
    print("saved runs/figures/solver_curve.png")
