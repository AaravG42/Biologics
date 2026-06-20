"""Multi-seed clean-data benchmark + ESM-2-vs-random ablation.

Reuses the exact training/eval used in the paper (train_single_trial), running
each model's *selected* config across multiple seeds under two entity-init
conditions:
  - esm2   : mAb entities initialised by additive ESM-2 (paper condition);
             non-mAb entities random uniform(-0.01, 0.01).
  - random : ALL entities random uniform(-0.01, 0.01) (identical base; mAb rows
             NOT overwritten). Isolates the effect of the ESM-2 prior.

All runs use the leakage-free directional dataset
(IMGT_no_leakage_directional_clinical). Outputs one metrics.json per
(model, init, seed) plus an aggregated results.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("DATA_PATH", str(REPO / "data"))

import models  # noqa: E402,F401  (registers model classes used via getattr)
from datasets.kg_dataset import KGDataset  # noqa: E402
from utils.imgt_clinical import (  # noqa: E402
    build_additive_imgt_entity_init,
    enable_cpu_fallback,
    load_forward_clinical_head_entity_ids,
    load_forward_clinical_relation_id,
    resolve_device,
)
from scripts.hpo_imgt_clinical_directional import train_single_trial  # noqa: E402

# Per-model selected trial (matches docs/PROVENANCE.md mapping).
SELECTED_TRIAL = {
    "RotE": 101, "TransE": 79, "RefE": 65, "MurE": 163, "AttE": 154,
    "AttH": 0, "RefH": 59, "RotH": 31, "CP": 99,
}
HPO_DIR = REPO / "analysis_outputs" / "IMGT_no_leakage_directional_clinical" / "clinical_hpo"
DATASET = "IMGT_no_leakage_directional_clinical"
# ESM-2 source embeddings (artifacts/kg_vocab_onco_filtered_ESM_640 + data/mab_inn_sequences_split.csv).
# After the Biologics rebuild these live under Biologics/esm_init/; the backup keeps the old flat layout.
_ESM_CANDIDATES = [
    Path("/home/aarav/Biologics/esm_init"),
    Path("/home/aarav/Biologics_backup_2026-06-19"),
]
ESM_DIR = next(
    (p for p in _ESM_CANDIDATES if (p / "data" / "mab_inn_sequences_split.csv").exists()),
    _ESM_CANDIDATES[0],
)


def load_selected_config(model: str) -> dict:
    trial = SELECTED_TRIAL[model]
    cfg_path = HPO_DIR / model / f"trial_{trial:03d}" / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["model"] = model  # be explicit
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(SELECTED_TRIAL))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--inits", nargs="+", default=["esm2", "random"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "runs")
    ap.add_argument("--max-epochs-override", type=int, default=0, help="for smoke tests; 0 = use config")
    ap.add_argument("--rank", type=int, default=640)
    args = ap.parse_args()

    base_device = resolve_device(args.device)
    if base_device == "cpu":
        enable_cpu_fallback(base_device)
        device = "cpu"
    else:
        torch.cuda.set_device(args.gpu)
        device = f"cuda:{args.gpu}"

    dataset_dir = Path(os.environ["DATA_PATH"]) / DATASET
    dataset = KGDataset(str(dataset_dir), debug=False)
    train_ex = dataset.get_examples("train")
    valid_ex = dataset.get_examples("valid")
    test_ex = dataset.get_examples("test")
    filters = dataset.get_filters()
    lhs_filters, rhs_filters = filters["lhs"], filters["rhs"]
    relation_offset = dataset.get_shape()[1] // 2
    forward_relation_id = load_forward_clinical_relation_id(dataset_dir)
    forward_heads = load_forward_clinical_head_entity_ids(dataset_dir)
    num_entities = dataset.get_shape()[0]

    # Build the two init arrays once (identical random base; esm overwrites mAb rows).
    esm_artifact = ESM_DIR / "artifacts" / "kg_vocab_onco_filtered_ESM_640"
    esm_seq = ESM_DIR / "data" / "mab_inn_sequences_split.csv"
    esm_init, _ = build_additive_imgt_entity_init(
        dataset_dir=dataset_dir,
        biologics_entities_csv=esm_artifact / "entities.csv",
        biologics_embeddings_npy=esm_artifact / "entity_embeddings.npy",
        sequences_csv=esm_seq,
        output_dir=args.out / "_esm_additive_640_init",
        init_dim=args.rank,
        seed=42,
    )
    rng = np.random.default_rng(42)
    random_init = rng.uniform(-0.01, 0.01, size=(num_entities, args.rank)).astype(np.float32)
    init_arrays = {"esm2": esm_init.astype(np.float32), "random": random_init}

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in args.models:
        cfg = load_selected_config(model)
        if args.max_epochs_override > 0:
            cfg = {**cfg, "max_epochs": args.max_epochs_override, "patience": args.max_epochs_override}
        for init_name in args.inits:
            for seed in args.seeds:
                trial_dir = args.out / model / init_name / f"seed_{seed}"
                trial_dir.mkdir(parents=True, exist_ok=True)
                done = trial_dir / "metrics.json"
                if done.exists():
                    m = json.loads(done.read_text())
                else:
                    metrics = train_single_trial(
                        trial_config=cfg, trial_dir=trial_dir, dataset=dataset,
                        train_examples=train_ex, valid_examples=valid_ex, test_examples=test_ex,
                        lhs_filters=lhs_filters, rhs_filters=rhs_filters,
                        forward_relation_id=forward_relation_id, relation_offset=relation_offset,
                        forward_clinical_head_entity_ids=forward_heads,
                        entity_init=init_arrays[init_name], device=device, seed=seed,
                        eval_batch_size=args.eval_batch_size,
                    )
                    m = {"model": model, "init": init_name, "seed": seed,
                         "config": cfg, "metrics": metrics}
                    done.write_text(json.dumps(m, indent=2))
                t = m["metrics"]["test"]
                rows.append([model, init_name, seed, t["tail_mr"], t["tail_mrr"],
                             t["tail_hits@1"], t["tail_hits@3"], t["tail_hits@10"]])
                print(f"{model:7} {init_name:6} seed={seed}  test MRR={t['tail_mrr']:.4f} "
                      f"MR={t['tail_mr']:.2f} H@1={t['tail_hits@1']*100:.1f} "
                      f"H@10={t['tail_hits@10']*100:.1f}", flush=True)

    csv_path = args.out / f"results_gpu{args.gpu}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "init", "seed", "test_mr", "test_mrr", "test_h1", "test_h3", "test_h10"])
        w.writerows(rows)
    print(f"\nWROTE {csv_path} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
