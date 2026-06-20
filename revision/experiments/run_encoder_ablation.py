"""Antibody-encoder ablation: ESM-2 vs AbLang vs IgBert on clean data.

Compares sequence-encoder entity initialisation for the paper's RotE model on the
leakage-free directional dataset. All encoders go through the SAME pipeline (the
640-d vectors produced by encoder_inits/*.npz), and every encoder's mAb-init rows
are MAGNITUDE-MATCHED to the random-init row norm, so conditions differ only in
DIRECTION (information), not scale. The matched random baseline is the
RotE/random runs from run_multiseed.py.
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

import models  # noqa: E402,F401
from datasets.kg_dataset import KGDataset  # noqa: E402
from datasets.process import get_idx  # noqa: E402
from utils.imgt_clinical import (  # noqa: E402
    enable_cpu_fallback,
    load_forward_clinical_head_entity_ids,
    load_forward_clinical_relation_id,
    resolve_device,
)
from scripts.hpo_imgt_clinical_directional import train_single_trial  # noqa: E402
from run_multiseed import DATASET, load_selected_config  # noqa: E402

ENC_DIR = Path(__file__).parent / "encoder_inits"


def build_encoder_init(enc: str, entity_to_id: dict, num_entities: int, rank: int,
                       target_norm: float, base_seed: int = 42) -> np.ndarray:
    """Random base (uniform, matched to baseline) with mAb rows = magnitude-matched encoder direction."""
    rng = np.random.default_rng(base_seed)
    init = rng.uniform(-0.01, 0.01, size=(num_entities, rank)).astype(np.float32)
    data = np.load(ENC_DIR / f"{enc}_mab_vectors.npz", allow_pickle=True)
    names = list(data["names"])
    vecs = np.asarray(data["vecs"], dtype=np.float32)
    placed = 0
    for name, vec in zip(names, vecs):
        eid = entity_to_id.get(str(name))
        if eid is None:
            continue
        n = float(np.linalg.norm(vec))
        if n == 0.0:
            continue
        init[int(eid)] = (vec / n * target_norm).astype(np.float32)
        placed += 1
    return init, placed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=["esm2", "ablang", "igbert"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--model", default="RotE")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--rank", type=int, default=640)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "encoder_ablation_runs")
    args = ap.parse_args()

    base_device = resolve_device(args.device)
    if base_device == "cpu":
        enable_cpu_fallback(base_device); device = "cpu"
    else:
        torch.cuda.set_device(args.gpu); device = f"cuda:{args.gpu}"

    dataset_dir = Path(os.environ["DATA_PATH"]) / DATASET
    dataset = KGDataset(str(dataset_dir), debug=False)
    train_ex = dataset.get_examples("train")
    valid_ex = dataset.get_examples("valid")
    test_ex = dataset.get_examples("test")
    filters = dataset.get_filters()
    lhs_f, rhs_f = filters["lhs"], filters["rhs"]
    rel_off = dataset.get_shape()[1] // 2
    fwd_rel = load_forward_clinical_relation_id(dataset_dir)
    fwd_heads = load_forward_clinical_head_entity_ids(dataset_dir)
    num_entities = dataset.get_shape()[0]
    entity_to_id, _ = get_idx(str(dataset_dir))

    # target norm = mean L2 row norm of the matched random baseline (uniform(-0.01,0.01))
    ref = np.random.default_rng(42).uniform(-0.01, 0.01, size=(num_entities, args.rank))
    target_norm = float(np.mean(np.linalg.norm(ref, axis=1)))
    print(f"target_norm (matched to random baseline) = {target_norm:.4f}", flush=True)

    cfg = load_selected_config(args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for enc in args.encoders:
        init, placed = build_encoder_init(enc, entity_to_id, num_entities, args.rank, target_norm)
        print(f"[{enc}] placed {placed} mAb init vectors", flush=True)
        for seed in args.seeds:
            trial_dir = args.out / args.model / enc / f"seed_{seed}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            done = trial_dir / "metrics.json"
            if done.exists():
                m = json.loads(done.read_text())
            else:
                metrics = train_single_trial(
                    trial_config=cfg, trial_dir=trial_dir, dataset=dataset,
                    train_examples=train_ex, valid_examples=valid_ex, test_examples=test_ex,
                    lhs_filters=lhs_f, rhs_filters=rhs_f, forward_relation_id=fwd_rel,
                    relation_offset=rel_off, forward_clinical_head_entity_ids=fwd_heads,
                    entity_init=init, device=device, seed=seed, eval_batch_size=args.eval_batch_size,
                )
                m = {"model": args.model, "init": enc, "seed": seed, "config": cfg, "metrics": metrics}
                done.write_text(json.dumps(m, indent=2))
            t = m["metrics"]["test"]
            rows.append([args.model, enc, seed, t["tail_mr"], t["tail_mrr"],
                         t["tail_hits@1"], t["tail_hits@3"], t["tail_hits@10"]])
            print(f"{args.model:7} {enc:7} seed={seed}  test MRR={t['tail_mrr']:.4f} "
                  f"MR={t['tail_mr']:.2f} H@1={t['tail_hits@1']*100:.1f} H@10={t['tail_hits@10']*100:.1f}", flush=True)

    csv_path = args.out / "encoder_ablation_results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "init", "seed", "test_mr", "test_mrr", "test_h1", "test_h3", "test_h10"])
        w.writerows(rows)
    print(f"\nWROTE {csv_path} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
