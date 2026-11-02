# %% [markdown]
# # Trainer for graph models

# %%
import argparse
import importlib
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.loader import DataLoader

from ta import ROOT
from ta.dataset import GraphDataset
from ta.metrics import evaluate
from ta.virtual import expand

# %% [markdown]
# ## Helpers


# %%
def build_model(cfg, dataset):
    module = importlib.import_module(f"models.{cfg['model']}")
    return module.Model(num_zones=dataset.net.num_zones, **cfg.get("model_args", {}))


@torch.no_grad()
def predict(model, dataset, rows, batch_size, device):
    model.eval()
    out = []
    for batch in DataLoader(dataset.subset(rows), batch_size=batch_size):
        out.append(model.predict_flows(batch.to(device)).cpu().reshape(batch.num_graphs, -1))
    return torch.cat(out).numpy().astype(np.float64)


def score(model, dataset, rows, batch_size, device, gap=False):
    pred = predict(model, dataset, rows, batch_size, device)
    conn = dataset.links.is_connector.to_numpy()
    return pred, evaluate(pred, dataset.data, dataset.links, conn, rows, dataset.splits.gridlock.to_numpy(), gap=gap)


# %% [markdown]
# ## Training rows and epoch plan


# %%
def select_training_rows(ds, cfg):
    rows = ds.rows("train", holdout=False)
    n = cfg.get("train_closure_sets")
    if n is None:
        return rows
    sets = np.unique(ds.data["closure_set"][rows])
    assert n <= len(sets), f"only {len(sets)} training closure sets"
    chosen = np.random.default_rng([cfg["seed"], 7]).choice(sets, n, replace=False)
    return rows[np.isin(ds.data["closure_set"][rows], chosen)]


def extra_training_data(cfg):
    extra = cfg.get("extra_train")
    if not extra:
        return None, 0
    ds = GraphDataset(cfg["network"], extra["dataset"])
    rows = ds.rows("train", holdout=False)
    sets = np.random.default_rng([cfg["seed"], 11]).permutation(np.unique(ds.data["closure_set"][rows]))
    assert extra["closure_sets"] <= len(sets), f"only {len(sets)} closure sets in {extra['dataset']}"
    rows = rows[np.isin(ds.data["closure_set"][rows], sets[:extra["closure_sets"]])]
    return ds.subset(rows), extra["closure_sets"]


def network_lists(cfg):
    return expand(cfg.get("train_networks", [cfg.get("network")])), expand(cfg.get("eval_networks", []))


class GroupedBatches(torch.utils.data.Sampler):
    """Shuffled batches that never mix the parts of a ConcatDataset (one part = one network)."""

    def __init__(self, sizes, batch_size, seed):
        self.sizes, self.batch_size, self.rng = list(sizes), batch_size, np.random.default_rng([seed, 5])

    def __iter__(self):
        batches, offset = [], 0
        for n in self.sizes:
            idx = offset + self.rng.permutation(n)
            batches += [idx[i:i + self.batch_size].tolist() for i in range(0, n, self.batch_size)]
            offset += n
        return iter([batches[i] for i in self.rng.permutation(len(batches))])

    def __len__(self):
        return sum(-(-n // self.batch_size) for n in self.sizes)


def plan_epochs(cfg, batches_per_epoch):
    epochs = int(np.ceil(cfg["steps"] / batches_per_epoch)) if "steps" in cfg else cfg["epochs"]
    return epochs, max(1, epochs // cfg.get("val_points", 60))


# %% [markdown]
# ## Training loop


# %%
def train(cfg, run_dir, smoke=False, resume=False):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.set_num_threads(cfg.get("threads", 4))
    device = torch.device(cfg.get("device", "cpu"))
    train_nets, eval_nets = network_lists(cfg)
    multi = len(train_nets) > 1 or bool(eval_nets)
    assert not (multi and cfg.get("extra_train")), "extra_train is for one network"
    cs_aon = cfg.get("model_args", {}).get("aon_input") == "closure_set"
    dss = {n: GraphDataset(n, cfg["dataset"], with_closure_set_aon=cs_aon) for n in train_nets + eval_nets}
    train_rows = {n: select_training_rows(dss[n], cfg) for n in train_nets}
    ds = dss[train_nets[0]]
    model = build_model(cfg, ds)
    if multi:
        model.fit_normalizer([dss[n] for n in train_nets], [train_rows[n] for n in train_nets])
    else:
        model.fit_normalizer(ds, train_rows[train_nets[0]])
    model = model.to(device)
    extra, extra_sets = extra_training_data(cfg)
    parts = [dss[n].subset(train_rows[n]) for n in train_nets] + ([extra] if extra is not None else [])
    train_data = parts[0] if len(parts) == 1 else torch.utils.data.ConcatDataset(parts)
    if len(train_nets) > 1:
        loader = DataLoader(train_data, batch_sampler=GroupedBatches([len(p) for p in parts], cfg["batch_size"], cfg["seed"]))
    else:
        loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True,
                            generator=torch.Generator().manual_seed(cfg["seed"]))
    val_rows = {n: dss[n].rows("val") for n in train_nets}
    test_rows = {n: dss[n].rows("test") for n in train_nets + eval_nets}
    holdout_rows = {n: dss[n].rows("train", holdout=True) for n in train_nets}
    epochs, val_every = plan_epochs(cfg, len(loader))
    max_batches = None
    if smoke:
        epochs, val_every, max_batches = 2, 1, 2
        cut = lambda rows: {n: r[:16] for n, r in rows.items()}
        val_rows, test_rows, holdout_rows = cut(val_rows), cut(test_rows), cut(holdout_rows)
    n_sets = sum(len(np.unique(dss[n].data["closure_set"][train_rows[n]])) for n in train_nets) + extra_sets
    print(f"training scenarios {len(train_data)} ({n_sets} closure sets, {len(train_nets)} network(s)) | epochs {epochs} | "
          f"validate every {val_every}" + (f" | zero-shot on {eval_nets}" if eval_nets else ""), flush=True)
    steps = epochs * (max_batches or len(loader))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.1,
                                                anneal_strategy="cos", div_factor=10.0, final_div_factor=1e3)
    history, best, start = [], np.inf, 1
    if resume and (run_dir / "last.pt").exists():
        state = torch.load(run_dir / "last.pt", weights_only=False)
        model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"]); sched.load_state_dict(state["sched"])
        history, best, start = state["history"], state["best"], state["epoch"] + 1
        torch.set_rng_state(state["rng"])
        print(f"resumed at epoch {start}")

    for epoch in range(start, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        losses = []
        for b, batch in enumerate(loader):
            if max_batches is not None and b >= max_batches:
                break
            opt.zero_grad()
            loss = model.loss(batch.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            sched.step()
            losses.append(loss.item())
        if epoch % val_every and epoch != epochs:
            continue
        vals = [score(model, dss[n], val_rows[n], cfg["batch_size"], device)[1].loc[("all", "real")] for n in train_nets]
        v = pd.DataFrame(vals).mean()
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_r2": v.r2, "val_mae": v.mae,
                        "val_wmape": v.wmape, "val_conservation": v.conservation, "lr": sched.get_last_lr()[0],
                        "seconds": time.perf_counter() - t0})
        if v.mae < best:
            best = v.mae
            torch.save(model.state_dict(), run_dir / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch,
                    "history": history, "best": best, "rng": torch.get_rng_state()}, run_dir / "last.pt")
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        h = history[-1]
        print(f"epoch {epoch}/{epochs} | loss {h['train_loss']:.4f} | val R² {h['val_r2']:.3f} MAE {h['val_mae']:.1f} "
              f"| {h['seconds']:.0f}s", flush=True)

    model.load_state_dict(torch.load(run_dir / "best.pt"))
    for net in train_nets + eval_nets:
        suffix = f"_{net}" if multi else ""
        preds, saved = {}, {}
        for name, rows in [("test", test_rows[net]), ("holdout", holdout_rows.get(net))]:
            if rows is None or len(rows) == 0:
                continue
            preds[name], table = score(model, dss[net], rows, cfg["batch_size"], device, gap=True)
            saved |= {f"{name}_rows": rows, name: preds[name]}
            table.to_csv(run_dir / f"{name}{suffix}.csv")
            label = f"{name} {net}" + (" (zero-shot: never trained on)" if net in eval_nets else "")
            print(f"== {label}\n{table.to_string(float_format=lambda x: f'{x:.4g}')}")
        np.savez_compressed(run_dir / f"predictions{suffix}.npz", **saved)
    return run_dir


# %% [markdown]
# ## Run folders


# %%
def new_run_dir(cfg, smoke=False):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{'smoke_' if smoke else ''}{stamp}_{cfg.get('name', cfg['dataset'])}"
    parent = ROOT / "runs" / cfg["model"]
    parent.mkdir(parents=True, exist_ok=True)
    for k in range(100):
        run_dir = parent / (base if k == 0 else f"{base}_{k}")
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"could not create a run folder for {base}")


# %% [markdown]
# ## Run

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", default="configs/train_ref_gatedgcn_main.yaml")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--device")
    parser.add_argument("--threads", type=int)
    args = parser.parse_known_args()[0]
    overrides = {k: v for k, v in [("device", args.device), ("threads", args.threads)] if v is not None}
    if args.resume:
        run_dir = ROOT / args.resume
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text()) | overrides
    else:
        cfg = yaml.safe_load((ROOT / args.config).read_text()) | overrides
        run_dir = new_run_dir(cfg, smoke=not args.full)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"run dir: {run_dir.relative_to(ROOT)} | pid {os.getpid()}")
    train(cfg, run_dir, smoke=not (args.full or args.resume), resume=bool(args.resume))
