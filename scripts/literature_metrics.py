# %% [markdown]
# # Metrics in the units of the literature

# %%
import numpy as np
import pandas as pd

from ta import ROOT
from ta.features import aon_cache
from ta.metrics import evaluate
from ta.splits import load_set

NETWORK, SET = "Berlin-Friedrichshain", "main"
OUT = ROOT / "runs" / "results" / NETWORK
RUNS = {"HGNN structure": ROOT / "runs" / "ref_gatedgcn" / "20260929_193510", "AON-GNN": ROOT / "runs" / "aon_gnn" / "20260930_061736"}
COLUMNS = ["r2", "mae", "rmse", "mape", "wmape", "vc_mae", "vc_rmse", "conservation", "gap"]

# %% [markdown]
# ## Predictions of every method on the test scenarios

# %%
data, links, splits = load_set(NETWORK, SET)
test = np.flatnonzero(splits.split.to_numpy() == "test")
preds = {"true flows": data["flows"][test], "all-or-nothing": aon_cache(NETWORK, SET, data)["aon"][test],
         "boosted trees": np.load(OUT / "trees_test_predictions.npy")}
for name, run in RUNS.items():
    with np.load(run / "predictions.npz") as f:
        assert np.array_equal(f["test_rows"], test)
        preds[name] = f["test"]

# %% [markdown]
# ## Scores (open real links; conservation and gap are per scenario over all links)

# %%
conn = links.is_connector.to_numpy()
rows = []
for name, p in preds.items():
    t = evaluate(p, data, links, conn, test, splits.gridlock.to_numpy(), gap=True)
    rows.append({"method": name, **t.loc[("all", "real"), COLUMNS].to_dict()})
table = pd.DataFrame(rows).set_index("method")
OUT.mkdir(parents=True, exist_ok=True)
table.to_csv(OUT / "literature_metrics.csv")
print(table.to_string(float_format=lambda v: f"{v:.4g}"))
