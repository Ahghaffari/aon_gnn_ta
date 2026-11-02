# %% [markdown]
# # Splits and flags

# %%
import numpy as np
import pandas as pd

from ta import ROOT

SPLIT_COUNTS = {"train": 205, "val": 46, "test": 49}
GRIDLOCK_RATIO = 5.0
HOLDOUT_VARIANTS = 5

# %% [markdown]
# ## Loading a dataset


# %%
def dataset_dir(network):
    return ROOT / "data" / network


def load_set(network, name, with_splits=True):
    with np.load(dataset_dir(network) / f"{name}.npz") as f:
        data = {k: f[k] for k in f.files}
    links = pd.read_csv(dataset_dir(network) / f"{name}.links.csv", index_col="link")
    splits = pd.read_csv(dataset_dir(network) / f"{name}.splits.csv") if with_splits else None
    if splits is not None:
        assert (splits.closure_set.to_numpy() == data["closure_set"]).all() and (splits.variant.to_numpy() == data["variant"]).all()
    return data, links, splits


# %% [markdown]
# ## Building the table


# %%
def split_closure_sets(closure_sets, counts=SPLIT_COUNTS, seed=42):
    sets = np.unique(closure_sets)
    scale = len(sets) / sum(counts.values())
    n_val, n_test = round(counts["val"] * scale), round(counts["test"] * scale)
    order = np.random.default_rng(seed).permutation(sets)
    labels = {int(c): "test" for c in order[:n_test]}
    labels |= {int(c): "val" for c in order[n_test:n_test + n_val]}
    labels |= {int(c): "train" for c in order[n_test + n_val:]}
    return labels


def demand_totals(data):
    return data["demand"].sum(axis=(1, 2)) if "demand" in data else data["demand_prod"].sum(1)


def avg_trip_time(data):
    return np.nansum(data["flows"] * data["times"], axis=1) / demand_totals(data)


def make_split_table(data, seed=42, holdout_variants=HOLDOUT_VARIANTS, gridlock_ratio=GRIDLOCK_RATIO, counts=SPLIT_COUNTS):
    labels = split_closure_sets(data["closure_set"], counts=counts, seed=seed)
    table = pd.DataFrame({"closure_set": data["closure_set"], "variant": data["variant"]})
    table["split"] = table.closure_set.map(labels)
    n_variants = table.variant.max() + 1
    table["holdout_variant"] = (table.split == "train") & (table.variant >= n_variants - holdout_variants)
    att = avg_trip_time(data)
    table["att_ratio"] = att / np.median(att[(table.split == "train").to_numpy()])
    table["gridlock"] = table.att_ratio > gridlock_ratio
    return table


def save_splits(network, name, **kwargs):
    data, _, _ = load_set(network, name, with_splits=False)
    table = make_split_table(data, **kwargs)
    path = dataset_dir(network) / f"{name}.splits.csv"
    table.to_csv(path, index=False)
    return path, table


# %% [markdown]
# ## Build and save the Berlin splits

# %%
if __name__ == "__main__":
    network = "Berlin-Friedrichshain"
    tables = {}
    for name in ["main", "old_like"]:
        path, tables[name] = save_splits(network, name)
        print(f"saved {path.relative_to(ROOT)}")
    path, tables["main_large"] = save_splits(network, "main_large", holdout_variants=0, counts={"train": 1, "val": 0, "test": 0})
    print(f"saved {path.relative_to(ROOT)}")

# %% [markdown]
# ## Inspect: what each split contains

# %%
if __name__ == "__main__":
    rows = []
    for name, t in tables.items():
        data, _, _ = load_set(network, name)
        for split in ["train", "val", "test"]:
            m = (t.split == split).to_numpy()
            rows.append({"set": name, "split": split, "closure sets": t.closure_set[m].nunique(), "scenarios": int(m.sum()),
                         "holdout variants": int(t.holdout_variant[m].sum()),
                         "closed links (min-max)": f"{data['num_closed'][m].min()}-{data['num_closed'][m].max()}",
                         "closed links mean": data["num_closed"][m].mean(), "gridlock share": t.gridlock[m].mean()})
    print(pd.DataFrame(rows).set_index(["set", "split"]).to_string(float_format=lambda v: f"{v:.3g}"))

# %% [markdown]
# ## Inspect: the first rows of a table

# %%
if __name__ == "__main__":
    print(tables["main"].head(16).to_string(index=False))
