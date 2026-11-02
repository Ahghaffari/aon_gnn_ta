# %% [markdown]
# # Few-shot adaptation to a new network

# %%
import argparse
import importlib

import numpy as np
import pandas as pd
import torch
import yaml

from ta import ROOT
from ta.dataset import GraphDataset
from ta.fewshot import evaluate_fewshot, network_episode, summary
from ta.virtual import expand

KS = [0, 1, 2, 5, 10, 20, 50]
MAIN = (0.1, 5)
LONG_MIN_K = 5
LONG = [(3e-4, 100), (3e-4, 500), (1e-3, 100), (1e-3, 500)]
SCRATCH = [(1e-3, 500), (1e-3, 2000)]
DEFAULTS = {("adapt", "adam"): (3e-4, 500), ("scratch", "adam"): (1e-3, 2000)}
VAL_STRIDE = 5

# %% [markdown]
# ## One run: every new network, every arm


# %%
def load_run(run_dir, device):
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    module = importlib.import_module(f"models.{cfg['model']}")
    model = module.Model(num_zones=None, **cfg.get("model_args", {}))
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    fresh = module.Model(num_zones=None, **cfg.get("model_args", {}))
    fresh.norm.load_state_dict(model.norm.state_dict())
    return cfg, model.to(device).eval(), fresh.to(device).eval()


def arms(k, short_settings):
    out = [("shift", "shift", "-", "sgd", [(0.0, 0)]), ("support_mean", "support_mean", "-", "sgd", [(0.0, 0)])]
    out += [(f"adapt_{scope}", "adapt", scope, "sgd", short_settings) for scope in ["all", "readout"]]
    if k >= LONG_MIN_K:
        out += [("finetune_long", "adapt", "all", "adam", LONG), ("train_on_k", "scratch", "all", "adam", SCRATCH)]
    return out


def choose_settings(models, val, k, method, scope, opt, settings, device):
    if len(settings) == 1:
        return settings[0]
    if not val:
        return DEFAULTS.get((method, opt), MAIN if MAIN in settings else settings[0])
    model = models["scratch" if method == "scratch" else "trained"]
    groups = val[::VAL_STRIDE] if opt == "adam" else val
    k_val = min(k, min(len(ep["support"]) for _, ep in groups))
    mae = [np.mean([evaluate_fewshot(model, ds, [ep], k_val, "adapt", n, lr, scope, device, opt)[2].loc[("all", "real")].mae
                    for ds, ep in groups]) for lr, n in settings]
    return settings[int(np.nanargmin(mae))]


def evaluate_run(run_dir, ks, lrs, steps, device, query_limit=None):
    cfg, model, fresh = load_run(run_dir, device)
    models = {"trained": model, "scratch": fresh}
    load = lambda key: [GraphDataset(n, cfg["dataset"]) for n in expand(cfg.get(key, []))]
    val = [(ds, network_episode(ds, cfg["seed"])) for ds in load("val_networks")]
    short = [(lr, n) for lr in lrs for n in steps]
    chosen = {}
    out = []
    for ds in load("eval_networks"):
        net = ds.network
        ep = network_episode(ds, cfg["seed"])
        ep["query"] = ep["query"][:query_limit]
        info = {"run": run_dir.name, "train_networks": "+".join(cfg["train_networks"]), "target": net,
                "selected_on": "val_networks" if val else "fixed defaults"}
        for k in [k for k in ks if k <= len(ep["support"])]:
            if k == 0:
                _, _, table, _ = evaluate_fewshot(model, ds, [ep], 0, "adapt", 0, 0.0, "all", device)
                out.append(summary(table, **info, method="zero_shot", k=0, opt="-", lr=np.nan, steps=0, main=True, adapt_ms=0.0))
                continue
            for name, method, scope, opt, settings in arms(k, short):
                key = (k, name)
                if key not in chosen:
                    chosen[key] = choose_settings(models, val, k, method, scope, opt, settings, device)
                for lr, n in settings:
                    m, how = (fresh, "adapt") if method == "scratch" else (model, method)
                    _, _, table, sec = evaluate_fewshot(m, ds, [ep], k, how, n, lr, scope, device, opt)
                    out.append(summary(table, **info, method=name, k=k, opt=opt, lr=lr if n else np.nan, steps=n,
                                       main=chosen[key] == (lr, n), adapt_ms=1e3 * sec))
            best = pd.DataFrame([r for r in out if r["target"] == net and r["k"] == k and r["main"]]).sort_values("r2").iloc[-1]
            print(f"{run_dir.name} → {net} | k={k} | best {best.method} (lr {best.lr}, {best.steps} steps) R² {best.r2:.4f}", flush=True)
    return pd.DataFrame(out)


# %% [markdown]
# ## Run

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--k", nargs="+", type=int, default=KS)
    parser.add_argument("--lrs", nargs="+", type=float, default=[0.03, 0.1])
    parser.add_argument("--steps", nargs="+", type=int, default=[5, 20])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_known_args()[0]
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    runs = [ROOT / r for r in args.runs] if args.runs else sorted((ROOT / "runs/aon_gnn").glob("*_m2_own_friedrichshain"))[-1:]
    ks, lrs, steps, limit = (args.k, args.lrs, args.steps, None) if args.full else ([0, 5], [0.1], [5], 32)
    out = ROOT / "runs" / "results" / "m2"
    out.mkdir(parents=True, exist_ok=True)
    for run in runs:
        table = evaluate_run(run, ks, lrs, steps, torch.device(args.device), limit)
        main = table[table.main].pivot_table(index=["target", "method"], columns="k", values="r2")
        print(main.round(4).to_string())
        if args.full:
            table.to_csv(out / f"fewshot_networks_{run.name}.csv", index=False)
            print(f"saved {out / f'fewshot_networks_{run.name}.csv'}")
