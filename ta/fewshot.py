# %% [markdown]
# # Few-shot adaptation: tasks, MAML inner loop, evaluation

# %%
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.func import functional_call
from torch_geometric.data import Batch

from ta.metrics import evaluate

SUPPORT_POOL = list(range(5))
QUERY = list(range(5, 15))

# %% [markdown]
# ## Tasks and episodes


# %%
def task_rows(ds, split):
    rows = ds.rows("train", holdout=False) if split == "train" else ds.rows(split)
    out = {}
    for r in rows:
        out.setdefault(int(ds.data["closure_set"][r]), {})[int(ds.data["variant"][r])] = int(r)
    return dict(sorted(out.items()))


def train_tasks(ds, n_sets=None, seed=42):
    tasks = task_rows(ds, "train")
    if n_sets is not None:
        chosen = np.random.default_rng([seed, 7]).choice(np.array(list(tasks)), n_sets, replace=False)
        tasks = {c: tasks[c] for c in sorted(int(c) for c in chosen)}
    return {c: (ds, list(v.values())) for c, v in tasks.items()}


def network_tasks(datasets):
    return {ds.network: (ds, ds.rows("train", holdout=False).tolist()) for ds in datasets}


def eval_episodes(ds, split, seed=42, limit=None):
    out = []
    for cs, v in list(task_rows(ds, split).items())[:limit]:
        order = np.random.default_rng([seed, cs]).permutation(SUPPORT_POOL)
        out.append({"closure_set": cs, "support": [v[i] for i in order if i in v], "query": [v[i] for i in QUERY if i in v]})
    return out


def network_episode(ds, seed=42):
    tasks = task_rows(ds, "train")
    rng = np.random.default_rng([seed, 17])
    support = [int(rng.choice(list(tasks[int(c)].values()))) for c in rng.permutation(np.array(list(tasks)))]
    return {"network": ds.network, "support": support, "query": ds.rows("test").tolist()}


def sample_episode(rows, k, m, rng):
    rows = rng.permutation(np.array(rows))
    return rows[:k].tolist(), rows[k:k + m].tolist()


def make_batch(ds, rows, device):
    return Batch.from_data_list([ds[r] for r in rows]).to(device)


# %% [markdown]
# ## Functional calls and the inner loop


# %%
class _Call(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch, what):
        return getattr(self.model, what)(batch)


def call(model, params, batch, what="loss"):
    return functional_call(_Call(model), {f"model.{n}": p for n, p in params.items()}, (batch, what))


def flows(model, params, batch):
    return call(model, params, batch, "predict_flows").reshape(batch.num_graphs, -1).cpu().numpy().astype(np.float64)


def base_params(model):
    return {n: p.detach() for n, p in model.named_parameters()}


def adapt(model, batch, params, steps, lr, scope="all", meta=False, first_order=False, opt="sgd"):
    names = [n for n in params if scope == "all" or n.startswith(scope)]
    if opt == "adam":
        assert not meta, "Adam adaptation is for evaluation only"
        leaves = {n: params[n].detach().clone().requires_grad_() for n in names}
        optim = torch.optim.Adam(list(leaves.values()), lr=lr)
        for _ in range(steps):
            optim.zero_grad()
            call(model, params | leaves, batch).backward()
            optim.step()
        return params | {n: p.detach() for n, p in leaves.items()}
    if not meta:
        params = params | {n: params[n].detach().requires_grad_() for n in names}
    for _ in range(steps):
        loss = call(model, params, batch)
        grads = torch.autograd.grad(loss, [params[n] for n in names], create_graph=meta and not first_order, allow_unused=True)
        new = {n: params[n] - lr * (g.detach() if first_order else g) for n, g in zip(names, grads) if g is not None}
        if not meta:
            new = {n: p.detach().requires_grad_() for n, p in new.items()}
        params = params | new
    return params


def meta_loss(model, tasks, rng, k, m, tasks_per_step, steps, lr, scope="all", first_order=False, device="cpu"):
    params = dict(model.named_parameters())
    keys = list(tasks)
    chosen = rng.choice(len(keys), min(tasks_per_step, len(keys)), replace=False)
    total = 0.0
    for i in chosen:
        ds, rows = tasks[keys[i]]
        support, query = sample_episode(rows, k, m, rng)
        adapted = adapt(model, make_batch(ds, support, device), params, steps, lr, scope, meta=True,
                        first_order=first_order) if k else params
        total = total + call(model, adapted, make_batch(ds, query, device))
    return total / len(chosen)


# %% [markdown]
# ## Few-shot evaluation


# %%
def evaluate_fewshot(model, ds, episodes, k, method="adapt", steps=5, lr=0.1, scope="all", device="cpu", opt="sgd"):
    assert method == "adapt" or k > 0, f"{method} needs k ≥ 1"
    model.eval()
    base = base_params(model)
    rows, preds, secs = [], [], []
    for ep in episodes:
        support, query = ep["support"][:k], ep["query"]
        t0 = time.perf_counter()
        if method == "support_mean":
            pred = np.repeat(ds.data["flows"][support].mean(0, keepdims=True), len(query), 0)
        else:
            params = adapt(model, make_batch(ds, support, device), base, steps, lr, scope, opt=opt) if method == "adapt" and k else base
            if method == "shift":
                resid = (ds.data["flows"][support] - flows(model, base, make_batch(ds, support, device))).mean(0)
            secs.append(time.perf_counter() - t0)
            pred = flows(model, params, make_batch(ds, query, device))
            if method == "shift":
                pred = np.where(ds.data["closed"][query], 0.0, np.maximum(pred + resid, 0.0))
        rows += query
        preds.append(pred)
    rows, pred = np.array(rows), np.concatenate(preds)
    table = evaluate(pred, ds.data, ds.links, ds.links.is_connector.to_numpy(), rows, ds.splits.gridlock.to_numpy())
    return pred, rows, table, float(np.mean(secs)) if secs else 0.0


def summary(table, **info):
    r = table.loc[("all", "real")]
    return info | {"r2": r.r2, "mae": r.mae, "wmape": r.wmape, "conservation": r.conservation,
                   "r2_gridlock": table.loc[("gridlock", "real")].r2 if ("gridlock", "real") in table.index else np.nan}


def fewshot_table(model, ds, ks, lrs, steps=5, scopes=("all", "readout"), methods=("shift", "support_mean"),
                  device="cpu", limit=None, label="model", seed=42):
    val, test = eval_episodes(ds, "val", seed, limit), eval_episodes(ds, "test", seed, limit)
    out = []
    for k in ks:
        arms = [("adapt", "all", [0.0])] if k == 0 else [("adapt", s, lrs) for s in scopes] + [(m, "-", [0.0]) for m in methods]
        for method, scope, grid in arms:
            best = grid[0]
            if len(grid) > 1:
                val_mae = [evaluate_fewshot(model, ds, val, k, method, steps, lr, scope, device)[2].loc[("all", "real")].mae for lr in grid]
                best = grid[int(np.nanargmin(val_mae))]
            _, _, table, sec = evaluate_fewshot(model, ds, test, k, method, steps, best, scope, device)
            name = "zero_shot" if k == 0 else (f"adapt_{scope}" if method == "adapt" else method)
            out.append(summary(table, model=label, method=name, k=k, solves_per_task=k, lr=best if method == "adapt" and k else np.nan,
                               steps=steps if method == "adapt" and k else 0, adapt_ms=1e3 * sec))
            print(f"{label:>12} | k={k} | {name:>14} | lr {best:<6g} | R² {out[-1]['r2']:.4f} | MAE {out[-1]['mae']:.1f}", flush=True)
    return pd.DataFrame(out)


# %% [markdown]
# ## Inspect: episodes, and fine-tuning the trained AON-GNN on a few val tasks

# %%
if __name__ == "__main__":
    import os

    from ta import ROOT
    from ta.dataset import GraphDataset
    from models.aon_gnn import Model

    os.nice(10)
    torch.set_num_threads(8)
    ds = GraphDataset("Berlin-Friedrichshain", "main")
    ep = eval_episodes(ds, "test")[0]
    print(f"test tasks {len(eval_episodes(ds, 'test'))} | task {ep['closure_set']}: support variants "
          f"{ds.data['variant'][ep['support']].tolist()}, query variants {ds.data['variant'][ep['query']].tolist()}")
    ckpt = ROOT / "runs/aon_gnn/20261002_143538_main/best.pt"
    if ckpt.exists():
        model = Model(num_zones=ds.net.num_zones).fit_normalizer(ds, ds.rows("train", holdout=False))
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        val = eval_episodes(ds, "val", limit=8)
        print("zero-shot  MAE", round(evaluate_fewshot(model, ds, val, 0)[2].loc[("all", "real")].mae, 1))
        for lr in [0.01, 0.1, 1.0]:
            for k in [1, 4]:
                mae = evaluate_fewshot(model, ds, val, k, "adapt", 5, lr)[2].loc[("all", "real")].mae
                print(f"fine-tune lr {lr:<5} k={k} MAE {mae:.1f}")
        for method in ["shift", "support_mean"]:
            print(method, "k=4 MAE", round(evaluate_fewshot(model, ds, val, 4, method)[2].loc[("all", "real")].mae, 1))
