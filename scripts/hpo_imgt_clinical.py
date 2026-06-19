#!/usr/bin/env python3
"""Hyperparameter search for IMGT clinical-indication tail ranking."""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import logging
import multiprocessing
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

import models
import optimizers.regularizers as regularizers
from datasets.kg_dataset import KGDataset
from models import all_models
from optimizers.kg_optimizer import KGOptimizer
from utils.imgt_clinical import (
    CLINICAL_INDICATION_INVERSE_RELATION,
    CLINICAL_INDICATION_RELATION,
    build_additive_imgt_entity_init,
    compute_rhs_metrics,
    enable_cpu_fallback,
    filter_triples_by_relation_ids,
    initialize_model_entities,
    load_clinical_relation_ids,
    resolve_device,
)
from utils.train import count_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run IMGT hyperparameter search across all repo models and select runs "
            "by validation clinical-indication tail mean rank."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="IMGT",
        choices=["IMGT", "IMGT_no_leakage"],
        help="Dataset name under DATA_PATH",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=all_models,
        choices=all_models,
        help="Subset of models to search. Defaults to all repo models.",
    )
    parser.add_argument("--rank", type=int, default=640, help="Fixed embedding dimension")
    parser.add_argument("--max-epochs", type=int, default=80, help="Max epochs per trial")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--valid-every", type=int, default=5, help="Validation interval in epochs")
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=256,
        help="Batch size for clinical tail evaluation",
    )
    parser.add_argument(
        "--trials-per-model",
        type=int,
        default=16,
        help="Number of sampled hyperparameter settings per model",
    )
    parser.add_argument(
        "--search-seed",
        type=int,
        default=42,
        help="Random seed for hyperparameter sampling",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device. 'auto' prefers CUDA when available.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0,1,2,3",
        help="Comma-separated CUDA device ids used for parallel trial execution.",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="How many concurrent trial processes to schedule per GPU.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/IMGT/clinical_hpo"),
        help="Directory for HPO summaries and per-trial checkpoints",
    )
    parser.add_argument(
        "--biologics-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "esm_init",
        help="Directory holding the prior ESM mAb embeddings (artifacts/kg_vocab_onco_filtered_ESM_640/ + data/mab_inn_sequences_split.csv). Defaults to this repo's esm_init/.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    """Set python, numpy, and torch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_search_space(model_name: str) -> Dict[str, List[object]]:
    """Return a compact model-aware search space."""
    space: Dict[str, List[object]] = {
        "regularizer": ["F2", "N3"],
        "reg": [0.0, 1e-6, 1e-5],
        "optimizer": ["Adam", "Adagrad"],
        "batch_size": [1024, 2048, 4096],
        "neg_sample_size": [32, 64, 128],
        "learning_rate": [3e-4, 1e-3, 3e-3],
        "dropout": [0.0],
        "init_size": [1e-3, 5e-3],
        "gamma": [0.0, 6.0, 12.0],
        "bias": ["constant", "none"],
        "dtype": ["single"],
        "double_neg": [False, True],
    }
    if model_name in {"ComplEx", "RotatE"}:
        space["regularizer"] = ["N3", "F2"]
        space["optimizer"] = ["Adagrad"]
    if model_name in {"AttE", "AttH"}:
        space["batch_size"] = [512, 1024, 2048]
    if model_name in {"RotH", "RefH", "AttH"}:
        space["multi_c"] = [False, True]
    return space


def sample_trial_configs(
    model_name: str,
    trials_per_model: int,
    base_args: argparse.Namespace,
    search_seed: int,
) -> List[Dict[str, object]]:
    """Sample a reproducible subset of hyperparameter combinations for a model."""
    space = build_search_space(model_name)
    keys = list(space.keys())
    all_combinations = [dict(zip(keys, values)) for values in itertools.product(*(space[key] for key in keys))]
    rng = random.Random(search_seed + sum(ord(ch) for ch in model_name))
    rng.shuffle(all_combinations)
    selected = all_combinations[: min(trials_per_model, len(all_combinations))]

    configs: List[Dict[str, object]] = []
    for index, hyperparams in enumerate(selected):
        config = {
            "dataset": base_args.dataset,
            "model": model_name,
            "rank": base_args.rank,
            "max_epochs": base_args.max_epochs,
            "patience": base_args.patience,
            "valid": base_args.valid_every,
            "debug": False,
        }
        config.update(hyperparams)
        config.setdefault("multi_c", False)
        configs.append(config)

        if index == 0:
            default_like = deepcopy(config)
            default_like.update(
                {
                    "regularizer": "N3" if model_name in {"ComplEx", "RotatE"} else "F2",
                    "reg": 0.0,
                    "optimizer": "Adam",
                    "batch_size": 2048,
                    "neg_sample_size": 64,
                    "learning_rate": 1e-3,
                    "init_size": 1e-3,
                    "gamma": 12.0 if model_name not in {"CP", "ComplEx"} else 0.0,
                    "bias": "constant",
                    "double_neg": False,
                    "multi_c": model_name in {"RotH", "RefH", "AttH"},
                }
            )
            configs.insert(0, default_like)

    deduped: List[Dict[str, object]] = []
    seen = set()
    for config in configs:
        key = tuple(sorted(config.items()))
        if key not in seen:
            seen.add(key)
            deduped.append(config)
    return deduped[:trials_per_model]


def build_trial_namespace(config: Dict[str, object], dataset: KGDataset) -> argparse.Namespace:
    """Convert a sampled config into the namespace expected by the repo models."""
    return argparse.Namespace(
        sizes=dataset.get_shape(),
        dropout=config["dropout"],
        gamma=config["gamma"],
        dtype=config["dtype"],
        bias=config["bias"],
        init_size=config["init_size"],
        rank=config["rank"],
        multi_c=config.get("multi_c", False),
    )


def setup_logging(output_dir: Path) -> None:
    """Log to stdout and a single hpo log file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(output_dir / "hpo.log"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def save_json(path: Path, payload: Dict[str, object]) -> None:
    """Write JSON with stable formatting."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_gpu_ids(gpu_ids: str) -> List[int]:
    """Parse comma-separated GPU ids."""
    values = [token.strip() for token in gpu_ids.split(",") if token.strip()]
    if not values:
        return []
    return [int(value) for value in values]


def select_device_for_trial(base_device: str, gpu_id: int | None) -> str:
    """Resolve the per-trial torch device string."""
    if base_device == "cpu":
        return "cpu"
    if gpu_id is None:
        return "cuda"
    return f"cuda:{gpu_id}"


def train_single_trial(
    trial_config: Dict[str, object],
    trial_dir: Path,
    dataset: KGDataset,
    train_examples: torch.Tensor,
    valid_examples: torch.Tensor,
    test_examples: torch.Tensor,
    rhs_filters,
    clinical_relation_ids: List[int],
    entity_init: np.ndarray,
    device: str,
    seed: int,
    eval_batch_size: int,
) -> Dict[str, object]:
    """Train one model/configuration pair and return its metrics."""
    seed_everything(seed)
    args = build_trial_namespace(trial_config, dataset)
    model = getattr(models, str(trial_config["model"]))(args)
    initialize_model_entities(model, entity_init)
    model.to(device)

    regularizer = getattr(regularizers, str(trial_config["regularizer"]))(
        float(trial_config["reg"])
    )
    optimizer = getattr(torch.optim, str(trial_config["optimizer"]))(
        model.parameters(), lr=float(trial_config["learning_rate"])
    )
    kg_optimizer = KGOptimizer(
        model=model,
        regularizer=regularizer,
        optimizer=optimizer,
        batch_size=int(trial_config["batch_size"]),
        neg_sample_size=int(trial_config["neg_sample_size"]),
        double_neg=bool(trial_config["double_neg"]),
        verbose=False,
    )

    if len(clinical_relation_ids) != 2:
        raise ValueError(
            "Expected two clinical relation ids (forward and inverse), "
            f"got {clinical_relation_ids}"
        )
    forward_relation_id, backward_relation_id = clinical_relation_ids

    def compute_directional_metrics(split_examples: torch.Tensor) -> Dict[str, Dict[str, float]]:
        forward_examples = filter_triples_by_relation_ids(split_examples, [forward_relation_id])
        backward_examples = filter_triples_by_relation_ids(split_examples, [backward_relation_id])
        combined_examples = filter_triples_by_relation_ids(split_examples, clinical_relation_ids)
        return {
            "forward": compute_rhs_metrics(
                model, forward_examples, rhs_filters, batch_size=eval_batch_size
            ),
            "backward": compute_rhs_metrics(
                model, backward_examples, rhs_filters, batch_size=eval_batch_size
            ),
            "combined": compute_rhs_metrics(
                model, combined_examples, rhs_filters, batch_size=eval_batch_size
            ),
        }

    best_metrics: Dict[str, object] | None = None
    best_epoch = -1
    stale_epochs = 0

    for epoch in range(int(trial_config["max_epochs"])):
        model.train()
        train_loss = float(kg_optimizer.epoch(train_examples).item())
        if (epoch + 1) % int(trial_config["valid"]) != 0:
            continue

        model.eval()
        valid_loss = float(kg_optimizer.calculate_valid_loss(valid_examples).item())
        valid_directional_metrics = compute_directional_metrics(valid_examples)
        test_directional_metrics = compute_directional_metrics(test_examples)
        valid_metrics = valid_directional_metrics["combined"]
        test_metrics = test_directional_metrics["combined"]
        current = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid": valid_metrics,
            "test": test_metrics,
            "valid_by_direction": valid_directional_metrics,
            "test_by_direction": test_directional_metrics,
        }

        is_better = best_metrics is None or valid_metrics["tail_mr"] < best_metrics["valid"]["tail_mr"]
        if is_better:
            best_metrics = current
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.cpu().state_dict(), trial_dir / "model.pt")
            model.to(device)
        else:
            stale_epochs += 1
            if stale_epochs >= int(trial_config["patience"]):
                break

    if best_metrics is None:
        raise RuntimeError("No validation checkpoint was produced; increase max_epochs or lower valid-every.")

    model.load_state_dict(torch.load(trial_dir / "model.pt", map_location=device))
    model.to(device)
    model.eval()
    best_metrics["best_epoch"] = best_epoch
    best_metrics["num_parameters"] = count_params(model)
    best_metrics["clinical_relation_ids"] = clinical_relation_ids
    best_metrics["clinical_relations"] = {
        "forward": {
            "name": CLINICAL_INDICATION_RELATION,
            "id": forward_relation_id,
        },
        "backward": {
            "name": CLINICAL_INDICATION_INVERSE_RELATION,
            "id": backward_relation_id,
        },
    }
    return best_metrics


def run_trial_job(job: Dict[str, object]) -> Dict[str, object]:
    """Worker entrypoint for one HPO trial."""
    dataset_dir = Path(str(job["dataset_dir"]))
    trial_dir = Path(str(job["trial_dir"]))
    trial_config = dict(job["trial_config"])
    gpu_id = job.get("gpu_id")
    base_device = str(job["base_device"])
    biologics_dir = Path(str(job["biologics_dir"]))
    output_dir = Path(str(job["output_dir"]))
    search_seed = int(job["search_seed"])
    eval_batch_size = int(job["eval_batch_size"])
    seed = int(job["seed"])
    rank = int(job["rank"])

    if base_device == "cuda":
        if gpu_id is None:
            raise ValueError("A GPU id is required for CUDA trial execution.")
        torch.cuda.set_device(int(gpu_id))

    device = enable_cpu_fallback(select_device_for_trial(base_device, gpu_id if base_device == "cuda" else None))

    biologics_artifact_dir = biologics_dir / "artifacts" / "kg_vocab_onco_filtered_ESM_640"
    biologics_sequences_csv = biologics_dir / "data" / "mab_inn_sequences_split.csv"
    entity_init, _ = build_additive_imgt_entity_init(
        dataset_dir=dataset_dir,
        biologics_entities_csv=biologics_artifact_dir / "entities.csv",
        biologics_embeddings_npy=biologics_artifact_dir / "entity_embeddings.npy",
        sequences_csv=biologics_sequences_csv,
        output_dir=output_dir / "esm_additive_640_init",
        init_dim=rank,
        seed=search_seed,
    )
    clinical_relation_ids = load_clinical_relation_ids(dataset_dir)
    dataset = KGDataset(str(dataset_dir), debug=False)
    train_examples = dataset.get_examples("train")
    valid_examples = dataset.get_examples("valid")
    test_examples = dataset.get_examples("test")
    rhs_filters = dataset.get_filters()["rhs"]

    metrics = train_single_trial(
        trial_config=trial_config,
        trial_dir=trial_dir,
        dataset=dataset,
        train_examples=train_examples,
        valid_examples=valid_examples,
        test_examples=test_examples,
        rhs_filters=rhs_filters,
        clinical_relation_ids=clinical_relation_ids,
        entity_init=entity_init,
        device=device,
        seed=seed,
        eval_batch_size=eval_batch_size,
    )
    return {
        "model": trial_config["model"],
        "trial_index": job["trial_index"],
        "seed": seed,
        "trial_dir": str(trial_dir),
        "gpu_id": gpu_id,
        "config": trial_config,
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    os.environ.setdefault("DATA_PATH", str(Path.cwd() / "data"))
    os.environ.setdefault("LOG_DIR", str(Path.cwd() / "logs"))
    setup_logging(args.output_dir)

    dataset_dir = Path(os.environ["DATA_PATH"]) / args.dataset
    base_device = resolve_device(args.device)
    if base_device == "cpu":
        enable_cpu_fallback(base_device)
    gpu_ids = parse_gpu_ids(args.gpu_ids) if base_device == "cuda" else []
    if base_device == "cuda" and not gpu_ids:
        raise ValueError("CUDA execution requires at least one GPU id in --gpu-ids.")
    max_workers = max(1, len(gpu_ids) * max(1, args.workers_per_gpu)) if base_device == "cuda" else 1
    logging.info("Using base device: %s", base_device)
    if gpu_ids:
        logging.info("Scheduling trials across GPUs: %s", gpu_ids)
    logging.info("Max parallel workers: %d", max_workers)

    biologics_artifact_dir = args.biologics_dir / "artifacts" / "kg_vocab_onco_filtered_ESM_640"
    biologics_sequences_csv = args.biologics_dir / "data" / "mab_inn_sequences_split.csv"
    entity_init, entity_init_metadata = build_additive_imgt_entity_init(
        dataset_dir=dataset_dir,
        biologics_entities_csv=biologics_artifact_dir / "entities.csv",
        biologics_embeddings_npy=biologics_artifact_dir / "entity_embeddings.npy",
        sequences_csv=biologics_sequences_csv,
        output_dir=args.output_dir / "esm_additive_640_init",
        init_dim=args.rank,
        seed=args.search_seed,
    )
    clinical_relation_ids = load_clinical_relation_ids(dataset_dir)

    overall_best: Dict[str, object] | None = None
    per_model_best: Dict[str, Dict[str, object]] = {}
    trial_summaries: List[Dict[str, object]] = []
    jobs: List[Dict[str, object]] = []

    for model_name in args.models:
        trial_configs = sample_trial_configs(
            model_name=model_name,
            trials_per_model=args.trials_per_model,
            base_args=args,
            search_seed=args.search_seed,
        )
        for trial_index, trial_config in enumerate(trial_configs):
            trial_seed = args.search_seed + (1000 * (all_models.index(model_name) + 1)) + trial_index
            trial_dir = args.output_dir / model_name / f"trial_{trial_index:03d}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            save_json(trial_dir / "config.json", trial_config)
            jobs.append(
                {
                    "dataset_dir": str(dataset_dir),
                    "trial_dir": str(trial_dir),
                    "trial_config": trial_config,
                    "trial_index": trial_index,
                    "seed": trial_seed,
                    "base_device": base_device,
                    "biologics_dir": str(args.biologics_dir),
                    "output_dir": str(args.output_dir),
                    "search_seed": args.search_seed,
                    "eval_batch_size": args.eval_batch_size,
                    "rank": args.rank,
                }
            )

    for job_index, job in enumerate(jobs):
        if base_device == "cuda":
            slot_count = max(1, len(gpu_ids) * max(1, args.workers_per_gpu))
            slot_index = job_index % slot_count
            job["gpu_id"] = gpu_ids[slot_index % len(gpu_ids)]
        else:
            job["gpu_id"] = None

    logging.info("Prepared %d total trials", len(jobs))

    if max_workers == 1:
        iterator = map(run_trial_job, jobs)
    else:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        futures: List[concurrent.futures.Future] = []
        try:
            futures = [executor.submit(run_trial_job, job) for job in jobs]
            iterator = concurrent.futures.as_completed(futures)
            for future in iterator:
                summary = future.result()
                save_json(Path(summary["trial_dir"]) / "metrics.json", summary)
                trial_summaries.append(summary)
                model_name = str(summary["model"])
                model_best = per_model_best.get(model_name)
                if model_best is None or summary["metrics"]["valid"]["tail_mr"] < model_best["metrics"]["valid"]["tail_mr"]:
                    per_model_best[model_name] = summary
                if overall_best is None or summary["metrics"]["valid"]["tail_mr"] < overall_best["metrics"]["valid"]["tail_mr"]:
                    overall_best = summary
                logging.info(
                    "[%s trial %03d gpu=%s] valid tail MR=%.4f | test tail MR=%.4f",
                    summary["model"],
                    summary["trial_index"],
                    summary["gpu_id"],
                    summary["metrics"]["valid"]["tail_mr"],
                    summary["metrics"]["test"]["tail_mr"],
                )
        except KeyboardInterrupt:
            logging.warning("Interrupted; cancelling pending trials and shutting down workers.")
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        except Exception:
            logging.exception("Trial execution failed; cancelling pending trials and shutting down workers.")
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True, cancel_futures=False)
        iterator = None

    if max_workers == 1:
        for summary in iterator:
            save_json(Path(summary["trial_dir"]) / "metrics.json", summary)
            trial_summaries.append(summary)
            model_name = str(summary["model"])
            model_best = per_model_best.get(model_name)
            if model_best is None or summary["metrics"]["valid"]["tail_mr"] < model_best["metrics"]["valid"]["tail_mr"]:
                per_model_best[model_name] = summary
            if overall_best is None or summary["metrics"]["valid"]["tail_mr"] < overall_best["metrics"]["valid"]["tail_mr"]:
                overall_best = summary
            logging.info(
                "[%s trial %03d gpu=%s] valid tail MR=%.4f | test tail MR=%.4f",
                summary["model"],
                summary["trial_index"],
                summary["gpu_id"],
                summary["metrics"]["valid"]["tail_mr"],
                summary["metrics"]["test"]["tail_mr"],
            )

    for model_name in args.models:
        model_best = per_model_best.get(model_name)
        if model_best is None:
            continue
        logging.info(
            "Best %s trial: valid tail MR=%.4f | test tail MR=%.4f | dir=%s",
            model_name,
            model_best["metrics"]["valid"]["tail_mr"],
            model_best["metrics"]["test"]["tail_mr"],
            model_best["trial_dir"],
        )

    if overall_best is None:
        raise RuntimeError("No trials completed successfully.")

    summary_payload = {
        "dataset": args.dataset,
        "device": base_device,
        "gpu_ids": gpu_ids,
        "workers_per_gpu": args.workers_per_gpu,
        "rank": args.rank,
        "clinical_relation_ids": clinical_relation_ids,
        "entity_init_metadata": entity_init_metadata,
        "search": {
            "models": args.models,
            "trials_per_model": args.trials_per_model,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "valid_every": args.valid_every,
            "eval_batch_size": args.eval_batch_size,
            "search_seed": args.search_seed,
        },
        "overall_best": overall_best,
        "per_model_best": per_model_best,
        "num_trials": len(trial_summaries),
        "trials": trial_summaries,
    }
    save_json(args.output_dir / "summary.json", summary_payload)

    print("Best overall model:", overall_best["model"])
    print("Best trial dir:", overall_best["trial_dir"])
    print("Validation clinical tail MR:", overall_best["metrics"]["valid"]["tail_mr"])
    print("Test clinical tail MR:", overall_best["metrics"]["test"]["tail_mr"])


if __name__ == "__main__":
    main()
