#!/usr/bin/env python3
"""Train a transductive PyKEEN model on indexed triples."""

from __future__ import annotations

import argparse
import csv
import json
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

# Keep PyKEEN/PyStow caches inside the repo so the script works without
# requiring write access to the user's home directory.
os.environ.setdefault("PYSTOW_HOME", str(Path("artifacts/pystow").resolve()))

from pykeen.models import BoxE, TransE
from pykeen.training import SLCWATrainingLoop
from pykeen.triples import CoreTriplesFactory


IndexedTriple = Tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a transductive PyKEEN model using an 80/20 split "
            "over the indexed knowledge-graph triples."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="boxe",
        choices=["transe", "boxe"],
        help="PyKEEN model architecture to train",
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
        help="Metadata JSON produced by build_kg_vocabulary.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for split files, trained model artifacts, and metrics. "
            "Defaults to artifacts/pykeen_<model>."
        ),
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Embedding dimension. Defaults to the vocabulary metadata value.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Training batch size",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=64,
        help="Evaluation batch size for tail ranking",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Optimizer learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split and training",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of facts used for training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device. 'auto' prefers CUDA when available.",
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


def split_triples(
    triples: torch.LongTensor, train_fraction: float, seed: int
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(triples.shape[0], generator=generator)
    cut = int(triples.shape[0] * train_fraction)
    train_idx = permutation[:cut]
    test_idx = permutation[cut:]
    return triples[train_idx], triples[test_idx]


def write_triples(path: Path, triples: torch.LongTensor) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["head_id", "relation_id", "tail_id"])
        writer.writerows(triples.tolist())


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_factory(
    triples: torch.LongTensor,
    num_entities: int,
    num_relations: int,
    create_inverse_triples: bool = False,
) -> CoreTriplesFactory:
    return CoreTriplesFactory(
        mapped_triples=triples,
        num_entities=num_entities,
        num_relations=num_relations,
        create_inverse_triples=create_inverse_triples,
    )


def compute_tail_ranks(
    model: Any,
    test_triples: torch.LongTensor,
    batch_size: int,
    device: str,
) -> torch.LongTensor:
    model.eval()
    ranks: List[torch.LongTensor] = []
    eval_device = torch.device(device)

    with torch.no_grad():
        for start in range(0, test_triples.shape[0], batch_size):
            batch = test_triples[start : start + batch_size].to(eval_device)
            hr_batch = batch[:, :2]
            true_tail_ids = batch[:, 2]

            # PyKEEN returns higher scores for more likely tails.
            scores = model.predict_t(hr_batch=hr_batch)
            true_tail_scores = scores.gather(1, true_tail_ids.unsqueeze(1))
            batch_ranks = torch.sum(scores > true_tail_scores, dim=1) + 1
            ranks.append(batch_ranks.cpu())

    return torch.cat(ranks, dim=0) if ranks else torch.empty(0, dtype=torch.long)


def extract_embeddings(model: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    entity_embeddings = model.entity_representations[0]().detach().cpu()
    relation_embeddings = model.relation_representations[0]().detach().cpu()
    return entity_embeddings, relation_embeddings


def build_model(
    model_name: str,
    training_factory: CoreTriplesFactory,
    embedding_dim: int,
    seed: int,
) -> Any:
    model_name = model_name.lower()
    if model_name == "transe":
        return TransE(
            triples_factory=training_factory,
            embedding_dim=embedding_dim,
            random_seed=seed,
        )
    if model_name == "boxe":
        return BoxE(
            triples_factory=training_factory,
            embedding_dim=embedding_dim,
            random_seed=seed,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata)
    triples = load_triples(args.triples)

    num_entities = int(metadata["num_entities"])
    num_relations = int(metadata["num_relations"])
    embedding_dim = (
        args.embedding_dim
        if args.embedding_dim is not None
        else int(metadata["embedding_dim"])
    )
    device = resolve_device(args.device)
    output_dir = args.output_dir or Path(f"artifacts/pykeen_{args.model}")

    train_triples, test_triples = split_triples(
        triples=triples,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_triples(output_dir / "train_triples.csv", train_triples)
    write_triples(output_dir / "test_triples.csv", test_triples)

    training_factory = build_factory(
        triples=train_triples,
        num_entities=num_entities,
        num_relations=num_relations,
    )

    print(f"PyKEEN version: {version('pykeen')}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {device}")
    print(f"Model: {args.model}")
    print(f"Train triples: {train_triples.shape[0]}")
    print(f"Test triples: {test_triples.shape[0]}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = build_model(
        model_name=args.model,
        training_factory=training_factory,
        embedding_dim=embedding_dim,
        seed=args.seed,
    )
    model = model.to(torch.device(device))

    training_loop = SLCWATrainingLoop(
        model=model,
        triples_factory=training_factory,
        optimizer="Adam",
        optimizer_kwargs={"lr": args.learning_rate},
    )
    losses = training_loop.train(
        triples_factory=training_factory,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        use_tqdm=True,
        use_tqdm_batch=True,
    )

    print("Evaluating held-out triples with tail prediction (h, r, ?)")
    tail_ranks = compute_tail_ranks(
        model=model,
        test_triples=test_triples,
        batch_size=args.eval_batch_size,
        device=device,
    )
    tail_mean_rank = float(tail_ranks.float().mean().item())

    entity_embeddings, relation_embeddings = extract_embeddings(model)
    torch.save(model.state_dict(), output_dir / "model_state_dict.pt")
    torch.save(entity_embeddings, output_dir / "entity_embeddings.pt")
    torch.save(relation_embeddings, output_dir / "relation_embeddings.pt")

    summary = {
        "library": "pykeen",
        "pykeen_version": version("pykeen"),
        "torch_version": torch.__version__,
        "model": model.__class__.__name__,
        "transductive": True,
        "train_fraction": args.train_fraction,
        "num_entities": num_entities,
        "num_relations": num_relations,
        "num_train_triples": int(train_triples.shape[0]),
        "num_test_triples": int(test_triples.shape[0]),
        "embedding_dim": embedding_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "training_losses": [float(loss) for loss in losses],
        "evaluation": {
            "task": "tail link prediction",
            "query_form": "(h, r, ?)",
            "ranking": "raw rank over all candidate tails using PyKEEN scores",
            "tail_mean_rank": tail_mean_rank,
        },
    }
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"Tail mean rank: {tail_mean_rank}")
    print(f"Artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
