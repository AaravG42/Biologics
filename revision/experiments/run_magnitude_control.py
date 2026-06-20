"""Magnitude control: is the additive-ESM gain sequence information or just init scale?

The paper's additive-ESM init gives ESM-initialised mAb entities a row-norm of ~11,
vs ~0.146 for every other (random) entity -- a ~76x magnitude gap. This control keeps
that exact per-row magnitude but replaces each ESM mAb vector with a RANDOM direction
of the SAME norm (no sequence information). If 'scaled_random' matches additive-ESM,
the apparent ESM gain is a magnitude artifact, not sequence information.
"""
from __future__ import annotations
import argparse, csv, json, os, re, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("DATA_PATH", str(REPO / "data"))

import models  # noqa: F401
from datasets.kg_dataset import KGDataset
from datasets.process import get_idx
from utils.imgt_clinical import (enable_cpu_fallback, load_forward_clinical_head_entity_ids,
                                 load_forward_clinical_relation_id, resolve_device)
from scripts.hpo_imgt_clinical_directional import train_single_trial
from run_multiseed import DATASET, load_selected_config

MAB = re.compile(r"imgt:mAb_\d+$")
ADD_ESM = Path("/home/aarav/Biologics/esm_init/entity_init.npy")


def build_scaled_random(entity_to_id, init_seed=123):
    base = np.load(ADD_ESM).astype(np.float32)  # additive-ESM init
    out = base.copy()
    norms = np.linalg.norm(base, axis=1)
    rng = np.random.default_rng(init_seed)
    replaced = 0
    for name, i in entity_to_id.items():
        if MAB.fullmatch(name) and norms[i] > 0.3:  # ESM-initialised mAb row
            v = rng.standard_normal(base.shape[1]).astype(np.float32)
            out[i] = v / np.linalg.norm(v) * norms[i]
            replaced += 1
    return out, replaced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["TransE", "RefE"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "magnitude_control_runs")
    args = ap.parse_args()

    base_device = resolve_device(args.device)
    if base_device == "cpu":
        enable_cpu_fallback(base_device); device = "cpu"
    else:
        torch.cuda.set_device(args.gpu); device = f"cuda:{args.gpu}"

    dd = Path(os.environ["DATA_PATH"]) / DATASET
    ds = KGDataset(str(dd), debug=False)
    tr, va, te = ds.get_examples("train"), ds.get_examples("valid"), ds.get_examples("test")
    flt = ds.get_filters(); lhs, rhs = flt["lhs"], flt["rhs"]
    rel_off = ds.get_shape()[1] // 2
    fwd = load_forward_clinical_relation_id(dd); heads = load_forward_clinical_head_entity_ids(dd)
    e2i, _ = get_idx(str(dd))
    init, replaced = build_scaled_random(e2i)
    print(f"scaled_random: replaced {replaced} ESM mAb rows with same-norm random directions "
          f"(mean mAb norm {np.mean([np.linalg.norm(init[i]) for n,i in e2i.items() if MAB.fullmatch(n) and np.linalg.norm(init[i])>0.3]):.2f})", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in args.models:
        cfg = load_selected_config(model)
        for seed in args.seeds:
            td = args.out / model / "scaled_random" / f"seed_{seed}"
            td.mkdir(parents=True, exist_ok=True)
            done = td / "metrics.json"
            if done.exists():
                m = json.loads(done.read_text())
            else:
                metrics = train_single_trial(trial_config=cfg, trial_dir=td, dataset=ds,
                    train_examples=tr, valid_examples=va, test_examples=te, lhs_filters=lhs, rhs_filters=rhs,
                    forward_relation_id=fwd, relation_offset=rel_off, forward_clinical_head_entity_ids=heads,
                    entity_init=init, device=device, seed=seed, eval_batch_size=args.eval_batch_size)
                m = {"model": model, "init": "scaled_random", "seed": seed, "config": cfg, "metrics": metrics}
                done.write_text(json.dumps(m, indent=2))
            t = m["metrics"]["test"]
            rows.append([model, "scaled_random", seed, t["tail_mr"], t["tail_mrr"], t["tail_hits@1"], t["tail_hits@10"]])
            print(f"{model:7} scaled_random seed={seed}  MRR={t['tail_mrr']:.4f} MR={t['tail_mr']:.2f} "
                  f"H@10={t['tail_hits@10']*100:.1f}", flush=True)
    with (args.out / "magnitude_control_results.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "init", "seed", "test_mr", "test_mrr", "test_h1", "test_h10"])
        w.writerows(rows)
    print(f"\nWROTE {args.out}/magnitude_control_results.csv ({len(rows)} runs)")


if __name__ == "__main__":
    main()
