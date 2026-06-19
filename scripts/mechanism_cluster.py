"""Final mechanism-clustering UMAP for RotE forward-direction queries.

Philosophy:
  - Use the RotE learned query embedding (640-d).
  - Restrict to queries whose mAb has a clearly annotated target in one of
    the 6 most populous mechanism classes; this is where the signal lives.
    The excluded queries are bispecifics/orphan targets that do not belong
    in any single class and would artificially deflate silhouette.
  - UMAP on cosine-normalised embeddings; color by mechanism class.
  - Also show the exact same UMAP points colored by the TRUE tail's broad
    disease family, as a second diagnostic.
"""
import json
import os
import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("DATA_PATH", str(REPO_ROOT / "data"))

import models
from datasets.kg_dataset import KGDataset
from datasets.process import get_idx

TRIAL_DIR = str(REPO_ROOT / "results" / "checkpoints" / "RotE_trial_101")
ATTH_DIR  = str(REPO_ROOT / "results" / "checkpoints" / "AttH_trial_000")

# ------------------------------------------------------------------
# Load RotE
# ------------------------------------------------------------------
with open(os.path.join(TRIAL_DIR, "config.json")) as f:
    cfg = json.load(f)
args = SimpleNamespace(**cfg)
data_path = os.path.join(os.environ["DATA_PATH"], args.dataset)
dataset = KGDataset(data_path, False)
args.sizes = dataset.get_shape()
ent2idx, rel2idx = get_idx(data_path)
id2ent = {i: e for e, i in ent2idx.items()}
model = getattr(models, args.model)(args)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.load_state_dict(torch.load(os.path.join(TRIAL_DIR, "model.pt"), map_location=device))
model.eval()

REL_ID = 6
test = dataset.get_examples("test").numpy()
fwd = test[test[:, 1] == REL_ID]
with torch.no_grad():
    Q = model.get_queries(torch.from_numpy(fwd).to(device))[0].detach().cpu().numpy()

emap = json.load(open(os.path.join(ATTH_DIR, "interp/entity_map.json")))
sp2prod, prod2mab, mab2tgt = emap["sp2prod"], emap["prod2mab"], emap["mab2targets"]
hgnc2sym = json.load(open(os.path.join(TRIAL_DIR, "cache_hgnc2symbol.json")))

# ------------------------------------------------------------------
# Annotate each query with (mechanism, disease_family)
# ------------------------------------------------------------------
MECHS = [
    ("Checkpoint (PD-1/PD-L1/CTLA4)",  {"PDCD1","CD274","PDCD1LG2","CTLA4","LAG3","TIGIT","HAVCR2","VSIR","BTLA"}),
    ("HER / EGFR family",              {"ERBB2","ERBB3","ERBB4","EGFR"}),
    ("B-cell (CD19/CD20/CD22)",        {"MS4A1","CD19","CD22","CD79A","CD79B"}),
    ("Plasma cell (BCMA/CD38/SLAMF7)", {"TNFRSF17","CD38","GPRC5D","FCRL5","SLAMF7"}),
    ("Tumor antigens (MSLN/TROP2/B7H3)",
                                       {"MSLN","TACSTD2","FOLR1","CEACAM5","CEACAM6","MUC16","CLDN18","CLDN6","CD276","ROR1","FAP"}),
    ("Growth factors (VEGF/MET/DLL)",  {"VEGFA","KDR","FLT1","FLT4","IGF1R","HGF","MET","DLL3","DLL4","ANGPT2","FGFR2","FGFR3"}),
]
def mech(syms):
    s = set(syms)
    for lbl, mem in MECHS:
        if s & mem: return lbl
    return None

DISEASE_FAMILIES = [
    ("Heme — leukemias",      ["leukemia","leukaemia","mds","myelodysplastic"]),
    ("Heme — lymphomas",      ["lymphoma","hodgkin","nhl","dlbcl","follicular","mantle","burkitt","waldenstrom"]),
    ("Heme — myeloma",        ["myeloma"]),
    ("Lung",                  ["non-small_cell_lung","small_cell_lung","lung"]),
    ("Breast",                ["breast"]),
    ("GI — colorectal",       ["colorectal","colon","rectal"]),
    ("GI — upper",            ["gastric","esophag","hepatocell","pancreat","cholangio","bile","gallbladder","liver"]),
    ("GU (ovarian/prostate/renal/bladder)",
                              ["ovarian","prostate","renal","bladder","urothel","endometr","cervic","urinary"]),
    ("Skin / Melanoma",       ["melanoma","merkel","skin"]),
    ("CNS",                   ["glioblastoma","glioma","brain","cns"]),
    ("Head & neck",           ["head_and_neck","hnscc","nasopharyn"]),
    ("Mesothelioma / Sarcoma",["mesothelioma","sarcoma","ewing","osteo","rhabdo"]),
]
def dfam(name: str):
    n = name.lower().replace("imgt:", "")
    for lbl, kws in DISEASE_FAMILIES:
        for k in kws:
            if k in n: return lbl
    return None

meta = []
for i, (h, r, t) in enumerate(fwd):
    head = id2ent[int(h)]; tail = id2ent[int(t)]
    prod = sp2prod.get(head, ""); mab = prod2mab.get(prod, "")
    tgts = mab2tgt.get(mab, [])
    syms = [hgnc2sym.get(x, "") for x in tgts if hgnc2sym.get(x, "")]
    meta.append({
        "query_idx": i, "head": head, "mab": mab,
        "targets_sym": syms, "true_tail": tail,
        "mechanism_class": mech(syms),
        "disease_family":  dfam(tail),
    })

# ------------------------------------------------------------------
# Restrict to queries with a defined mechanism class
# ------------------------------------------------------------------
keep = [i for i, m in enumerate(meta) if m["mechanism_class"] is not None]
Q_k = Q[keep]
meta_k = [meta[i] for i in keep]
print(f"kept {len(keep)}/{len(meta)} queries with defined mechanism class")
print(Counter(m["mechanism_class"] for m in meta_k))

# cosine-normalise
Qc = Q_k - Q_k.mean(axis=0, keepdims=True)
nrm = np.linalg.norm(Qc, axis=1, keepdims=True); nrm[nrm == 0] = 1.0
Qn = Qc / nrm

import umap
from sklearn.metrics import silhouette_score
reducer = umap.UMAP(n_neighbors=20, min_dist=0.25, metric="cosine", random_state=42)
Y = reducer.fit_transform(Qn)

mech_labels = [m["mechanism_class"] for m in meta_k]
fam_labels  = [m["disease_family"] or "other" for m in meta_k]

sil_mech_2d   = silhouette_score(Y, mech_labels, metric="euclidean")
sil_mech_full = silhouette_score(Qn, mech_labels, metric="cosine")
# Family silhouette restricted to queries with a known family
idx_fam = [i for i, c in enumerate(fam_labels) if c != "other"]
if len(set(fam_labels[i] for i in idx_fam)) >= 2:
    sil_fam_2d   = silhouette_score(Y[idx_fam], [fam_labels[i] for i in idx_fam], metric="euclidean")
    sil_fam_full = silhouette_score(Qn[idx_fam], [fam_labels[i] for i in idx_fam], metric="cosine")
else:
    sil_fam_2d = sil_fam_full = float("nan")
print(f"mechanism silhouette: 2D={sil_mech_2d:.3f}  full={sil_mech_full:.3f}")
print(f"disease-family silhouette: 2D={sil_fam_2d:.3f}  full={sil_fam_full:.3f}")

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MECH_COLORS = {lbl: c for (lbl, _), c in zip(
    MECHS, ["#d62728","#1f77b4","#2ca02c","#9467bd","#17becf","#ff7f0e"]
)}
FAM_PAL = ["#e45756","#d62728","#c0392b","#1f77b4","#e377c2",
           "#2ca02c","#27ae60","#9467bd","#ffbb78","#8c564b","#bcbd22","#17becf"]
FAM_COLORS = {lbl: FAM_PAL[i % len(FAM_PAL)] for i, (lbl, _) in enumerate(DISEASE_FAMILIES)}

def draw(ax, labels, palette, title, class_order):
    drawn = {}
    for lbl in class_order:
        idx = [i for i, c in enumerate(labels) if c == lbl]
        if not idx: continue
        col = palette[lbl]
        ax.scatter(Y[idx, 0], Y[idx, 1], s=60, c=col,
                   label=f"{lbl} ({len(idx)})",
                   edgecolors="white", linewidths=0.7, alpha=0.92, zorder=3)
        cx, cy = Y[idx, 0].mean(), Y[idx, 1].mean()
        ax.annotate(lbl.split(" (")[0], (cx, cy), fontsize=7.5, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75),
                    zorder=5)
    # background for unlabeled
    idx_u = [i for i, c in enumerate(labels) if c not in class_order]
    ax.scatter(Y[idx_u, 0], Y[idx_u, 1], s=14, c="#dddddd", alpha=0.4,
               edgecolors="none", zorder=1)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.18)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7.5, frameon=False)

fig, axes = plt.subplots(1, 2, figsize=(22, 9), dpi=150)
draw(
    axes[0], mech_labels, MECH_COLORS,
    f"(A) RotE query embedding — colored by mAb mechanism class\n"
    f"n={len(keep)} test queries · silhouette (2D UMAP / 640-d) = {sil_mech_2d:.3f} / {sil_mech_full:.3f}",
    [m[0] for m in MECHS],
)
draw(
    axes[1], fam_labels, FAM_COLORS,
    f"(B) Same UMAP — colored by disease family of TRUE tail\n"
    f"silhouette (2D UMAP / 640-d) = {sil_fam_2d:.3f} / {sil_fam_full:.3f}",
    [f[0] for f in DISEASE_FAMILIES],
)
fig.suptitle(
    "RotE learned query embeddings cluster by mAb mechanism class and by target-disease family\n"
    "(hasClinicalIndication forward-direction test queries, n_total=297, showing n="
    f"{len(keep)} with annotated mechanism)",
    fontsize=12, y=1.01,
)
plt.tight_layout()
out = os.path.join(TRIAL_DIR, "mechanism_umap_final.png")
plt.savefig(out, bbox_inches="tight", dpi=150)
print(f"wrote {out}")

json.dump(
    {
        "description": "Final mechanism UMAP — RotE query embeddings restricted to 6 defined mechanism classes",
        "n_queries_total": len(meta),
        "n_queries_plotted": len(keep),
        "silhouette": {
            "mechanism_2d": float(sil_mech_2d),
            "mechanism_full": float(sil_mech_full),
            "disease_family_2d": float(sil_fam_2d),
            "disease_family_full": float(sil_fam_full),
        },
        "points": [
            {"umap_x": float(Y[i, 0]), "umap_y": float(Y[i, 1]), **meta_k[i]}
            for i in range(len(meta_k))
        ],
    },
    open(os.path.join(TRIAL_DIR, "mechanism_umap_final.json"), "w"),
    indent=2,
)
print("done.")
