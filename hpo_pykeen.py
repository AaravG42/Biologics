#!/usr/bin/env python3
"""Run PyKEEN hyperparameter optimization on indexed triples."""

from __future__ import annotations

import argparse
import csv
import json
import os
from importlib.metadata import version
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# Keep PyKEEN/PyStow caches inside the repo so the script works without
# requiring write access to the user's home directory.
os.environ.setdefault("PYSTOW_HOME", str(Path("artifacts/pystow").resolve()))

from pykeen.hpo import hpo_pipeline
from pykeen.evaluation.rank_based_evaluator import RankBasedEvaluator, Ranks
from pykeen.triples import CoreTriplesFactory


IndexedTriple = Tuple[int, int, int]


def patch_rank_based_evaluator() -> None:
    """Work around PyKEEN 1.11.1 missing true_scores bug in rank evaluation."""

    original_process_scores = RankBasedEvaluator.process_scores_

    def patched_process_scores_(self, hrt_batch, target, scores, true_scores=None, dense_positive_mask=None):
        if true_scores is None:
            if target == "head":
                true_ids = hrt_batch[:, 0]
            elif target == "relation":
                true_ids = hrt_batch[:, 1]
            else:
                true_ids = hrt_batch[:, 2]
            true_scores = scores.gather(1, true_ids.view(-1, 1))

        batch_ranks = Ranks.from_scores(
            true_score=true_scores,
            all_scores=scores,
        )
        self.num_entities = scores.shape[1]
        for rank_type, values in batch_ranks.items():
            self.ranks[target, rank_type].append(values.detach().cpu().numpy())
        self.num_candidates[target].append(batch_ranks.number_of_options.detach().cpu().numpy())

    if RankBasedEvaluator.process_scores_ is not original_process_scores:
        return
    RankBasedEvaluator.process_scores_ = patched_process_scores_


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PyKEEN HPO on indexed triples. Defaults target BoxE on the "
            "literal-excluded graph and optimize validation tail mean rank."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="boxe",
        choices=["boxe", "transe"],
        help="PyKEEN model architecture to optimize",
    )
    parser.add_argument(
        "--triples",
        type=Path,
        default=Path("artifacts/kg_vocab_literal_excluded/triples_ids.csv"),
        help="CSV containing head_id,relation_id,tail_id",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("artifacts/kg_vocab_literal_excluded/metadata.json"),
        help="Metadata JSON for the indexed triples",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for HPO study artifacts. "
            "Defaults to artifacts/pykeen_<model>_hpo_cleaned."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs per HPO trial",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Training batch size per HPO trial",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=64,
        help="Evaluation batch size used by PyKEEN during HPO",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of Optuna trials",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional HPO timeout in seconds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split reproducibility",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of facts used for training",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Fraction of facts used for validation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="HPO/training device. 'auto' prefers CUDA when available.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optional Optuna study name",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optional Optuna storage URL",
    )
    return parser.parse_args()


def load_metadata(path: Path) -> Dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_triples(path: Path) -> torch.LongTensor:
    triples: List[IndexedTriple] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = {"head_id", "relation_id", "tail_id"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError(
                f"Expected columns {sorted(expected)}, got {reader.fieldnames}"
            )
        for row in reader:
            triples.append(
                (int(row["head_id"]), int(row["relation_id"]), int(row["tail_id"]))
            )
    return torch.as_tensor(triples, dtype=torch.long)


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def split_triples(
    triples: torch.LongTensor,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("train_fraction and validation_fraction must be positive")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be < 1.0")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(triples.shape[0], generator=generator)

    n_total = triples.shape[0]
    n_train = int(n_total * train_fraction)
    n_validation = int(n_total * validation_fraction)

    train_idx = permutation[:n_train]
    validation_idx = permutation[n_train : n_train + n_validation]
    test_idx = permutation[n_train + n_validation :]
    return triples[train_idx], triples[validation_idx], triples[test_idx]


def write_triples(path: Path, triples: torch.LongTensor) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["head_id", "relation_id", "tail_id"])
        writer.writerows(triples.tolist())


def build_factory(
    triples: torch.LongTensor,
    num_entities: int,
    num_relations: int,
) -> CoreTriplesFactory:
    return CoreTriplesFactory(
        mapped_triples=triples,
        num_entities=num_entities,
        num_relations=num_relations,
        create_inverse_triples=False,
    )


def default_model_kwargs_ranges(model: str) -> Dict[str, Dict[str, object]]:
    if model == "boxe":
        return {
            "embedding_dim": {"type": int, "low": 64, "high": 256, "step": 64},
            "p": {"type": int, "low": 1, "high": 2},
        }
    if model == "transe":
        return {
            "embedding_dim": {"type": int, "low": 64, "high": 256, "step": 64},
            "scoring_fct_norm": {"type": int, "low": 1, "high": 2},
        }
    raise ValueError(f"Unsupported model: {model}")


def default_optimizer_kwargs_ranges() -> Dict[str, Dict[str, object]]:
    return {
        "lr": {"type": float, "low": 1.0e-4, "high": 1.0e-2, "log": True},
    }


def default_training_kwargs_ranges() -> Dict[str, Dict[str, object]]:
    return {
        "batch_size": {"type": int, "low": 256, "high": 2048, "step": 256},
    }


def main() -> None:
    args = parse_args()
    patch_rank_based_evaluator()
    metadata = load_metadata(args.metadata)
    triples = load_triples(args.triples)
    num_entities = int(metadata["num_entities"])
    num_relations = int(metadata["num_relations"])
    device = resolve_device(args.device)

    output_dir = args.output_dir or Path(f"artifacts/pykeen_{args.model}_hpo_cleaned")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_triples, validation_triples, test_triples = split_triples(
        triples=triples,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )

    write_triples(output_dir / "train_triples.csv", train_triples)
    write_triples(output_dir / "validation_triples.csv", validation_triples)
    write_triples(output_dir / "test_triples.csv", test_triples)

    training_factory = build_factory(train_triples, num_entities, num_relations)
    validation_factory = build_factory(validation_triples, num_entities, num_relations)
    testing_factory = build_factory(test_triples, num_entities, num_relations)

    print(f"PyKEEN version: {version('pykeen')}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {device}")
    print(f"Model: {args.model}")
    print(f"Train triples: {train_triples.shape[0]}")
    print(f"Validation triples: {validation_triples.shape[0]}")
    print(f"Test triples: {test_triples.shape[0]}")
    print("Optimizing validation tail mean rank with direction=minimize")

    result = hpo_pipeline(
        training=training_factory,
        validation=validation_factory,
        testing=testing_factory,
        model=args.model,
        model_kwargs_ranges=default_model_kwargs_ranges(args.model),
        optimizer="Adam",
        optimizer_kwargs_ranges=default_optimizer_kwargs_ranges(),
        training_loop="slcwa",
        negative_sampler="basic",
        epochs=args.epochs,
        training_kwargs={
            "batch_size": args.batch_size,
        },
        training_kwargs_ranges=default_training_kwargs_ranges(),
        metric="tail.realistic.mean_rank",
        direction="minimize",
        evaluator_kwargs={
            "filtered": False,
        },
        evaluation_kwargs={
            "batch_size": args.eval_batch_size,
            "targets": ("tail",),
        },
        device=device,
        n_trials=args.n_trials,
        timeout=args.timeout,
        study_name=args.study_name,
        storage=args.storage,
    )

    result.save_to_directory(output_dir / "hpo_study")
    summary = {
        "library": "pykeen",
        "pykeen_version": version("pykeen"),
        "torch_version": torch.__version__,
        "model": args.model,
        "device": device,
        "num_entities": num_entities,
        "num_relations": num_relations,
        "num_triples": int(triples.shape[0]),
        "num_train_triples": int(train_triples.shape[0]),
        "num_validation_triples": int(validation_triples.shape[0]),
        "num_test_triples": int(test_triples.shape[0]),
        "metric": "tail.realistic.mean_rank",
        "direction": "minimize",
        "n_trials": args.n_trials,
        "timeout": args.timeout,
        "best_trial_number": result.study.best_trial.number,
        "best_trial_value": result.study.best_value,
        "best_params": result.study.best_params,
        "artifacts": {
            "study": "hpo_study",
            "best_pipeline_config": "hpo_study/best_pipeline/pipeline_config.json",
        },
        "notes": [
            "The HPO study used PyKEEN's hpo_pipeline on the validation split.",
            "replicate_best_pipeline is not used here because PyKEEN serializes in-memory factories as <user defined> and cannot reload them.",
        ],
    }
    (output_dir / "hpo_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"Best trial: {result.study.best_trial.number}")
    print(f"Best validation tail mean rank: {result.study.best_value}")
    print(f"Artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
