# %% [markdown]
# # Stage M tables over seeds

# %%
import re

import numpy as np
import pandas as pd

from ta import ROOT

OUT = ROOT / "runs" / "results" / "stage_m"
FEWSHOT = ROOT / "runs" / "results" / "Berlin-Friedrichshain" / "fewshot"
M2 = ROOT / "runs" / "results" / "m2"
SIZES = {"lc005": 5, "lc020": 20, "lc050": 50, "main": 205}
BASES = {"friedrichshain": "Berlin-Friedrichshain", "prenzlauerberg": "Berlin-Prenzlauerberg-Center",
         "tiergarten": "Berlin-Tiergarten", "mitte": "Berlin-Mitte-Center"}
MPFC = "Berlin-Mitte-Prenzlauerberg-Friedrichshain-Center"

# %% [markdown]
# ## Helpers


# %%
def seed_of(name):
    m = re.search(r"(?:_s|seed)(4[34])$", name)
    return int(m.group(1)) if m else 42


def mean_sd(values):
    v = np.asarray(values, dtype=float)
    return f"{v.mean():.3f}" if len(v) == 1 else f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"


def pivot(df, index, columns="k", value="r2"):
    cells = df.groupby(index + [columns])[value].agg(lambda v: mean_sd(v)).unstack(columns)
    seeds = df.groupby(index).seed.nunique().rename("seeds")
    return cells.join(seeds)


def save(table, name):
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / f"{name}.csv")
    (OUT / f"{name}.md").write_text(table.to_markdown() + "\n")
    print(f"== {name}\n{table.to_string()}\n")


# %% [markdown]
# ## M1: new closure sets of a known network


# %%
def m1_rows():
    rows = []
    for f in sorted(FEWSHOT.glob("*.csv")):
        t = pd.read_csv(f)
        for _, r in t.iterrows():
            model = str(r.model)
            maml = re.match(r"\d+_\d+_maml_(lc\d{3}|main)(_fo|_anil|_scratch_lr003)?(?:_s4[34])?$", model)
            if model == "scratch":
                arm, size = ("scratch (AON + k)" if r.method == "scratch" else "AON"), "-"
            elif maml:
                size, kind = maml.group(1), maml.group(2) or ""
                name = {"": "MAML 2nd order", "_fo": "MAML 1st order", "_anil": "MAML head only", "_scratch_lr003": "MAML random init"}[kind]
                wanted = "adapt_readout" if kind == "_anil" else "adapt_all"
                if r.method not in ("zero_shot", wanted):
                    continue
                arm = f"{name} (start)" if r.method == "zero_shot" else name
            else:
                size = next((s for s in SIZES if re.search(rf"_{s}(?:_s4[34]|_seed4[34])?$", model)), None)
                if size is None:
                    continue
                arm = {"zero_shot": "zero-shot", "adapt_all": "fine-tune all", "adapt_readout": "fine-tune head",
                       "shift": "shift", "support_mean": "support average"}[r.method]
            rows.append({"size": SIZES.get(size, 0), "arm": arm, "k": int(r.k), "seed": seed_of(model), "r2": r.r2,
                         "source": f.name, "model": model})
    df = pd.DataFrame(rows)
    return df.drop_duplicates(["size", "arm", "k", "seed", "model"], keep="last")


# %% [markdown]
# ## M2 / M2.3: new networks


# %%
def target_kind(target, held):
    if target == MPFC:
        return "MPFC"
    if target == held:
        return "held-out network"
    if target.startswith(held + "@v"):
        return "its virtual versions (mean)"
    return None


def network_rows(prefix):
    rows = []
    for f in sorted(M2.glob(f"fewshot_networks_*_{prefix}_*.csv")):
        t = pd.read_csv(f)
        t = t[t.main]
        run = t.run.iloc[0]
        held_key = re.search(rf"_{prefix}_([a-z]+)", run).group(1)
        held = BASES[held_key]
        t = t.assign(kind=[target_kind(x, held) for x in t.target]).dropna(subset=["kind"])
        g = t.groupby(["kind", "method", "k"]).r2.mean().reset_index()
        rows += [{"held_out": held_key, **r, "seed": seed_of(run)} for r in g.to_dict("records")]
    return pd.DataFrame(rows)


def maml_network_rows():
    rows = []
    for f in sorted((ROOT / "runs" / "aon_gnn").glob("*_m23_maml_*/fewshot_test.csv")):
        run = f.parent.name
        m = re.search(r"_m23_maml_([a-z]+)(_fo|_anil)?(?:_s4[34])?$", run)
        if not m or run.startswith("smoke"):
            continue
        held = BASES[m.group(1)]
        t = pd.read_csv(f)
        extra = f.with_name("fewshot_test_extra.csv")
        if extra.exists():
            t = pd.concat([t, pd.read_csv(extra)]).drop_duplicates(["network", "k"], keep="first")
        t = t.assign(kind=[target_kind(x, held) for x in t.network]).dropna(subset=["kind"])
        g = t.groupby(["kind", "k"]).r2.mean().reset_index()
        method = {None: "MAML 2nd order", "_fo": "MAML 1st order", "_anil": "MAML head only"}[m.group(2)]
        rows += [{"held_out": m.group(1), "method": method, **r, "seed": seed_of(run)} for r in g.to_dict("records")]
    return pd.DataFrame(rows)


# %% [markdown]
# ## Build every table

# %%
if __name__ == "__main__":
    m1 = m1_rows()
    save(pivot(m1, ["size", "arm"]), "m1")
    order = ["held_out", "kind", "method"]
    for prefix in ["m2_loo", "m23_erm"]:
        df = network_rows(prefix)
        save(pivot(df, order), prefix)
    maml = maml_network_rows()
    save(pivot(maml, order), "m23_maml")
