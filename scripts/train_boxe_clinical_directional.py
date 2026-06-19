#!/usr/bin/env python3
"""Launch directional IMGT clinical HPO for BoxE using the RotE-style pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train BoxE on IMGT directional clinical indication prediction using "
            "the same HPO workflow used for RotE."
        )
    )
    parser.add_argument("--dataset", type=str, default="IMGT_no_leakage_directional_clinical")
    parser.add_argument("--rank", type=int, default=640)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--valid-every", type=int, default=5)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--trials-per-model", type=int, default=16)
    parser.add_argument("--search-seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse completed trials (default true).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/IMGT_no_leakage_directional_clinical/clinical_hpo"),
    )
    parser.add_argument(
        "--biologics-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "esm_init",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    base_script = Path(__file__).with_name("hpo_imgt_clinical_directional.py")

    cmd = [
        sys.executable,
        str(base_script),
        "--dataset",
        args.dataset,
        "--models",
        "BoxE",
        "--rank",
        str(args.rank),
        "--max-epochs",
        str(args.max_epochs),
        "--patience",
        str(args.patience),
        "--valid-every",
        str(args.valid_every),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--trials-per-model",
        str(args.trials_per_model),
        "--search-seed",
        str(args.search_seed),
        "--device",
        args.device,
        "--gpu-ids",
        args.gpu_ids,
        "--workers-per-gpu",
        str(args.workers_per_gpu),
        "--output-dir",
        str(args.output_dir),
        "--biologics-dir",
        str(args.biologics_dir),
    ]

    cmd.append("--resume" if args.resume else "--no-resume")
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{repo_root}:{existing_pythonpath}" if existing_pythonpath else str(repo_root)
    )
    subprocess.run(cmd, check=True, cwd=repo_root, env=env)


if __name__ == "__main__":
    main()
