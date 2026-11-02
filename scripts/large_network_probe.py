# %% [markdown]
# # Large-network probe: pruning, AON, solver curve, AON-GNN cost (Stage S0)

# %%
import os

os.environ.setdefault("AEQ_SHOW_PROGRESS", "FALSE")

import argparse
import json
import resource
import time

import numpy as np
import torch

from models.aon_gnn import DEFAULTS, AonGNN
from ta import ROOT
from ta.dataset import ScenarioData, static_graph
from ta.network import Network, unreachable_demand
from ta.solver import conservation_error, prepare_links, prune_dead_ends, relative_gap, run_aequilibrae

# %% [markdown]
# ## Setup

# %%
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--network", default="Sydney")
parser.add_argument("--cores", type=int, default=os.cpu_count())
parser.add_argument("--truncate", type=int, nargs="*", default=[2, 5, 10, 20])
parser.add_argument("--rgap", type=float, default=1e-4)
parser.add_argument("--max-iter", type=int, default=1000)
parser.add_argument("--no-converged", action="store_true")
parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
args = parser.parse_known_args()[0]
OUT = ROOT / "runs" / "results" / args.network / "probe"
OUT.mkdir(parents=True, exist_ok=True)
results = {"args": vars(args)}


def log(key, value):
    results[key] = value
    print(f"{key}: {value}", flush=True)
    (OUT / "probe.json").write_text(json.dumps(results, indent=1, default=float))


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


# %% [markdown]
# ## Network, pruning, reachability

# %%
t0 = time.perf_counter()
net = Network.from_tntp(args.network)
log("load_s", time.perf_counter() - t0)
z, block = net.num_zones, net.block_centroids
od = np.array(net.demand, dtype=np.float64)
np.fill_diagonal(od, 0.0)
log("size", {"zones": z, "links": net.num_links, "connectors": int(net.is_connector.sum()), "real_streets": len(net.real_streets()),
             "od_pairs": int((od > 0).sum()), "demand": od.sum(), "intrazonal": float(np.trace(net.demand))})

prep = prepare_links(net.links)
t0 = time.perf_counter()
keep = prune_dead_ends(prep, z)
log("prune_s", time.perf_counter() - t0)
log("pruned_links", int((~keep).sum()))
active = prep[keep].reset_index(drop=True)
active["link_id"] = np.arange(1, len(active) + 1)

t0 = time.perf_counter()
lost = unreachable_demand(net, net.default_closed)
log("reachability_s", time.perf_counter() - t0)
log("unreachable_demand", lost)
real = ~net.is_connector
flows = {}


def assign(name, algorithm, max_iter, cores, rgap_target=1e-12):
    t0 = time.perf_counter()
    f, report = run_aequilibrae(active, od, block, rgap_target, max_iter, algorithm, cores)
    secs = time.perf_counter() - t0
    full = np.zeros(len(prep))
    full[keep] = f
    t0 = time.perf_counter()
    gap = relative_gap(prep, od, full, block)
    gap_s = time.perf_counter() - t0
    iterations = max(len(report), 1)
    log(name, {"s": secs, "iterations": iterations, "s_per_iter": secs / iterations, "rgap": gap,
               "rgap_reported": float(report["rgap"].iloc[-1]) if len(report) else None,
               "conservation": conservation_error(prep, full, od),
               "gap_eval_s": gap_s, "peak_rss_gb": peak_gb()})
    flows[name] = full
    np.savez_compressed(OUT / "flows.npz", **flows)
    return full, report


# %% [markdown]
# ## All-or-nothing (AequilibraE, one core and all cores)

# %%
aon, _ = assign("aon_1core", "all-or-nothing", 1, 1)
assign(f"aon_{args.cores}cores", "all-or-nothing", 1, args.cores)

# %% [markdown]
# ## AON-GNN forward / backward on the full graph (untrained weights: cost only)

# %%
g = static_graph(net)
item = ScenarioData(edge_index=g["edge_index"], node_static=g["node_static"], edge_static=g["edge_static"],
                    zone_node=g["zone_node"], num_nodes=len(g["nodes"]), demand=torch.as_tensor(od, dtype=torch.float32)[None],
                    closed=torch.zeros(net.num_links, dtype=torch.bool), aon=torch.as_tensor(aon, dtype=torch.float32),
                    capacity_factor=torch.ones(net.num_links), fft_factor=torch.ones(net.num_links),
                    y=torch.as_tensor(aon, dtype=torch.float32), graph_scale=g["graph_scale"])
model = AonGNN(**DEFAULTS)
model.norm.c.fill_(float(g["graph_scale"][0, 0]))
model.norm.demand_mean.fill_(float(od.sum()))


def time_model(device, reps=5):
    m, b = model.to(device), item.to(device)
    sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)
    with torch.inference_mode():
        m.predict_flows(b)
        sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            m.predict_flows(b)
        sync()
        fwd = (time.perf_counter() - t0) / reps
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    m.train()
    t0 = time.perf_counter()
    m.loss(b).backward()
    sync()
    out = {"forward_s": fwd, "train_step_s": time.perf_counter() - t0}
    if device.startswith("cuda"):
        out["train_step_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    m.zero_grad()
    m.eval()
    return out


torch.set_num_threads(1)
log("gnn_cpu_1thread", time_model("cpu", reps=2))
if args.device.startswith("cuda"):
    log("gnn_gpu", time_model(args.device))

# %% [markdown]
# ## Solver: truncated runs and one converged run

# %%
for n in args.truncate:
    assign(f"bfw_{n}", "bfw", n, args.cores)
if not args.no_converged:
    conv, report = assign("bfw_converged", "bfw", args.max_iter, args.cores, rgap_target=args.rgap)
    report.to_csv(OUT / "converged_report.csv", index=False)
    r2 = lambda p: float(1 - ((conv[real] - p[real]) ** 2).sum() / ((conv[real] - conv[real].mean()) ** 2).sum())
    log("r2_vs_converged", {k: r2(v) for k, v in flows.items() if k != "bfw_converged"})
