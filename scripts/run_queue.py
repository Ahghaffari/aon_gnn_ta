# %% [markdown]
# # Run a queue of training configs

# %%
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from ta import ROOT

LOGS = ROOT / "runs" / "logs"

# %% [markdown]
# ## Plan


# %%
def read_queue(paths):
    configs = []
    for path in paths:
        for line in (ROOT / path).read_text().splitlines():
            line = line.split("#")[0].strip()
            if line:
                assert (ROOT / line).exists(), f"missing config {line}"
                configs.append(line)
    return configs


def command(cfg, device, smoke, threads=None):
    cmd = [sys.executable, "-u", str(ROOT / "scripts" / "train.py"), "--config", cfg, "--device", device]
    cmd += ["--threads", str(threads)] if threads else []
    return cmd if smoke else cmd + ["--full"]


# %% [markdown]
# ## Run


# %%
def run(configs, devices, parallel, smoke, threads=None):
    LOGS.mkdir(parents=True, exist_ok=True)
    pending, running, done = list(configs), {}, []
    free = list(devices)
    while pending or running:
        while pending and free and len(running) < parallel:
            cfg, device = pending.pop(0), free.pop(0)
            log = LOGS / f"{Path(cfg).stem}{'_smoke' if smoke else ''}.log"
            proc = subprocess.Popen(command(cfg, device, smoke, threads), cwd=ROOT, stdout=log.open("w"), stderr=subprocess.STDOUT,
                                    preexec_fn=lambda: os.nice(19 - os.nice(0)))
            running[proc] = (cfg, device, log, time.time())
            print(f"{datetime.now():%H:%M:%S} start {cfg} on {device} → {log.relative_to(ROOT)}", flush=True)
        time.sleep(5)
        for proc in [p for p in running if p.poll() is not None]:
            cfg, device, log, t0 = running.pop(proc)
            free.append(device)
            done.append((cfg, log, proc.returncode, time.time() - t0))
            print(f"{datetime.now():%H:%M:%S} done  {cfg} (exit {proc.returncode}, {(time.time() - t0) / 60:.1f} min)", flush=True)
    return done


# %% [markdown]
# ## Summary


# %%
def summarize(done):
    rows = []
    for cfg, log, code, sec in done:
        text = log.read_text()
        run_dir = next((ROOT / l.split("run dir: ")[1].split(" |")[0] for l in text.splitlines() if l.startswith("run dir: ")), None)
        row = {"config": Path(cfg).stem, "exit": code, "minutes": round(sec / 60, 1), "run_dir": run_dir.relative_to(ROOT) if run_dir else None}
        for name in ["test", "holdout"]:
            path = run_dir / f"{name}.csv" if run_dir else None
            if path and path.exists():
                t = pd.read_csv(path, index_col=[0, 1])
                row[f"{name}_r2"], row[f"{name}_mae"] = t.loc[("all", "real"), "r2"], t.loc[("all", "real"), "mae"]
        rows.append(row)
    table = pd.DataFrame(rows)
    path = LOGS / f"summary_{datetime.now():%Y%m%d_%H%M%S}.csv"
    table.to_csv(path, index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"summary → {path.relative_to(ROOT)}")
    return table


# %% [markdown]
# ## Main

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("queues", nargs="*", default=["runs/queues/weekend_A.txt", "runs/queues/weekend_B.txt"])
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--parallel", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_known_args()[0]
    configs = read_queue(args.queues)
    parallel = args.parallel or len(args.devices)
    print(f"{len(configs)} jobs, {parallel} at a time on {args.devices} ({'smoke' if args.smoke else 'full'} runs):")
    for i, cfg in enumerate(configs):
        print(f"  {i + 1:2d}. {cfg}")
    if args.run:
        summarize(run(configs, args.devices, parallel, args.smoke, args.threads))
    else:
        print("plan only — add --run to start")
