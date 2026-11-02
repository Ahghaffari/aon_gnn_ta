# %% [markdown]
# # Throughput: AON-GNN on GPU vs solver on CPU

# %%
import os

for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[var] = "1"
os.environ.setdefault("AEQ_SHOW_PROGRESS", "FALSE")

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Batch

from ta import ROOT
from ta.baselines import aon_flows
from ta.dataset import GraphDataset
from ta.network import Network, solve_scenario

NETWORK, SET = "Berlin-Friedrichshain", "main"
GNN_RUN = ROOT / "runs" / "aon_gnn" / "20261002_143538_main"
OUT = ROOT / "runs" / "results" / NETWORK / "throughput_node.csv"
WORKER = {}


def init_worker():
    WORKER["net"] = Network.from_tntp(NETWORK)


def aon_one(args):
    demand, closed = args
    return aon_flows(WORKER["net"], demand, closed)


def solve_one(args):
    demand, closed, max_iter, target = args
    sol, _ = solve_scenario(WORKER["net"], demand, closed, rgap_target=target, max_iter=max_iter, retries=1 if target < 1e-9 else 3)
    return sol.iterations


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_known_args()[0]
    device = torch.device("cuda:0")
    ds = GraphDataset(NETWORK, SET)
    data, test = ds.data, ds.rows("test")
    jobs = [(data["demand"][r], data["closed"][r]) for r in test]
    rows = []

    def record(name, n, sec, **extra):
        rows.append({"method": name, "scenarios": n, "seconds": sec, "scenarios_per_s": n / sec, "ms_per_scenario": 1e3 * sec / n} | extra)
        print(f"{name:40} {n / sec:9.1f} scenarios/s | {1e3 * sec / n:8.3f} ms per scenario", flush=True)

    with ProcessPoolExecutor(args.workers, initializer=init_worker) as pool:
        list(pool.map(aon_one, jobs[: args.workers], chunksize=1))
        t0 = time.perf_counter()
        aon = list(pool.map(aon_one, jobs, chunksize=4))
        record(f"all-or-nothing, {args.workers} CPU cores", len(jobs), time.perf_counter() - t0)

        cfg = yaml.safe_load((GNN_RUN / "config.yaml").read_text())
        from models.aon_gnn import Model

        model = Model(num_zones=None, **cfg.get("model_args", {}))
        model.load_state_dict(torch.load(GNN_RUN / "best.pt", map_location="cpu"))
        model = model.to(device).eval()
        items = []
        for r, a in zip(test, aon):
            item = ds[r]
            item.aon = torch.as_tensor(a, dtype=torch.float32)
            items.append(item)
        with torch.inference_mode():
            for bs in [1, 64, 512]:
                for _ in range(3):
                    model.predict_flows(Batch.from_data_list(items[:bs]).to(device))
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for i in range(0, len(items), bs):
                    model.predict_flows(Batch.from_data_list(items[i:i + bs]).to(device)).cpu()
                torch.cuda.synchronize()
                record(f"AON-GNN forward on GPU, batch {bs}", len(items), time.perf_counter() - t0, batch=bs)

        for name, max_iter, target in [("solver, 5 iterations", 5, 1e-12), ("solver, converged", 2000, 1e-5)]:
            t0 = time.perf_counter()
            iters = list(pool.map(solve_one, [(d, c, max_iter, target) for d, c in jobs], chunksize=1))
            record(f"{name}, {args.workers} CPU cores", len(jobs), time.perf_counter() - t0, median_iterations=float(np.median(iters)))

    table = pd.DataFrame(rows)
    aon_ms = table.ms_per_scenario.iloc[0]
    for bs in [1, 64, 512]:
        fwd = table[(table.method.str.startswith("AON-GNN")) & (table.batch == bs)].ms_per_scenario.iloc[0]
        record(f"AON-GNN pipeline (AON + GPU forward), batch {bs}", 1, (aon_ms + fwd) / 1e3, batch=bs)
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
