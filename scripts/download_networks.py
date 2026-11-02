# %% [markdown]
# # Download TNTP networks

# %%
import argparse
import hashlib
import json
import re
import urllib.request
from ta import ROOT

REPO = "bstabler/TransportationNetworks"
COMMIT = "d1639b4ef218c17928ba573e806ddf8ba5e7ae6d"
OUT = ROOT / "networks"

TIERS = {
    "toy": ["Braess-Example"],
    "core": [
        "SiouxFalls",
        "Eastern-Massachusetts",
        "Berlin-Friedrichshain",
        "Berlin-Prenzlauerberg-Center",
        "Berlin-Tiergarten",
        "Berlin-Mitte-Center",
        "Anaheim",
        "Berlin-Mitte-Prenzlauerberg-Friedrichshain-Center",
        "Barcelona",
        "Winnipeg",
        "Chicago-Sketch",
    ],
    "large": [
        "Munich",
        "GoldCoast",
        "Austin",
        "Berlin-Center",
        "Birmingham-England",
        "chicago-regional",
        "Philadelphia",
        "Sydney",
    ],
}
KEEP = re.compile(r"\.tntp(\.zip)?$", re.IGNORECASE)

# %% [markdown]
# ## Helpers


# %%
def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def git_blob_sha(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def remote_files(names):
    tree = json.loads(fetch(f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1"))
    assert not tree["truncated"]
    files = {n: [] for n in names}
    for item in tree["tree"]:
        parts = item["path"].split("/")
        if item["type"] == "blob" and len(parts) == 2 and parts[0] in files and KEEP.search(parts[1]):
            files[parts[0]].append(item)
    missing = [n for n, f in files.items() if not f]
    assert not missing, f"no TNTP files found for {missing}"
    return files


# %% [markdown]
# ## Download with verification


# %%
def download(names):
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"repo": REPO, "commit": COMMIT, "networks": {}}
    assert manifest["commit"] == COMMIT, "manifest was built from a different commit"

    for name, items in remote_files(names).items():
        entry = {}
        for item in items:
            fname = item["path"].split("/")[1]
            dest = OUT / name / fname
            if dest.exists() and git_blob_sha(dest.read_bytes()) == item["sha"]:
                status = "cached"
            else:
                data = fetch(f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{urllib.request.quote(item['path'])}")
                assert git_blob_sha(data) == item["sha"], f"hash mismatch for {item['path']}"
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                status = "downloaded"
            entry[fname] = {"size": item["size"], "git_sha": item["sha"]}
            print(f"{status:10s} {name}/{fname} ({item['size'] / 1024:.0f} KB)")
        manifest["networks"][name] = entry

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"manifest -> {manifest_path}")


# %% [markdown]
# ## Run

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--tier", nargs="+", default=["toy", "core", "large"], choices=list(TIERS))
    parser.add_argument("--networks", nargs="+", help="explicit network names (overrides --tier)")
    args = parser.parse_known_args()[0]
    download(args.networks or [n for t in args.tier for n in TIERS[t]])

# %% [markdown]
# ## Inspect what is on disk

# %%
if __name__ == "__main__":
    manifest = json.loads((OUT / "manifest.json").read_text())
    print(f"commit {manifest['commit'][:8]}")
    for name, files in sorted(manifest["networks"].items()):
        kinds = sorted(re.sub(r"(?i)^.*_(net|trips|node|flow|toll)\.tntp(\.zip)?$", r"\1", f).lower() for f in files)
        print(f"  {name:52s} {sum(f['size'] for f in files.values()) / 1024:8.0f} KB  {', '.join(kinds)}")
