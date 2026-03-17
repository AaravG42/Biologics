#!/usr/bin/env python3
"""Train a transductive TransE model from indexed knowledge-graph triples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


IndexedTriple = Tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a TransE model using an 80/20 transductive split over the "
            "indexed triples generated from the knowledge graph."
        )
    )
    parser.add_argument(
        "--triples",
        type=Path,
        default=Path("artifacts/kg_vocab/triples_ids.csv"),
        help="CSV containing head_id,relation_id,tail_id",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("artifacts/kg_vocab/metadata.json"),
        help="Metadata JSON produced by build_kg_vocabulary.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/transe"),
        help="Directory for trained embeddings, split files, and metrics",
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
        default=5,
        help="Number of SGD passes over the training triples",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for TransE updates",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.0,
        help="Margin used in the ranking loss",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split, initialization, and negative sampling",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=256,
        help="Number of test triples to score per evaluation batch",
    )
    return parser.parse_args()


def load_metadata(path: Path) -> Dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_triples(path: Path) -> np.ndarray:
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
    return np.asarray(triples, dtype=np.int64)


def write_triples(path: Path, triples: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["head_id", "relation_id", "tail_id"])
        writer.writerows(triples.tolist())


def split_triples(
    triples: np.ndarray, seed: int, train_fraction: float = 0.8
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(triples))
    cut = int(len(indices) * train_fraction)
    train_idx = indices[:cut]
    test_idx = indices[cut:]
    return triples[train_idx], triples[test_idx]


def initialize_embeddings(
    num_rows: int, embedding_dim: int, rng: np.random.Generator
) -> np.ndarray:
    limit = 0.01
    return rng.uniform(-limit, limit, size=(num_rows, embedding_dim)).astype(np.float32)


def normalize_rows(matrix: np.ndarray) -> None:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    matrix /= norms


def sample_alternative(
    rng: np.random.Generator, vocab_size: int, true_id: int
) -> int:
    candidate = int(rng.integers(vocab_size - 1))
    if candidate >= true_id:
        candidate += 1
    return candidate


def train_transe(
    train_triples: np.ndarray,
    num_entities: int,
    num_relations: int,
    embedding_dim: int,
    epochs: int,
    learning_rate: float,
    margin: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    rng = np.random.default_rng(seed)
    entity_embeddings = initialize_embeddings(num_entities, embedding_dim, rng)
    relation_embeddings = initialize_embeddings(num_relations, embedding_dim, rng)
    normalize_rows(entity_embeddings)

    history: List[Dict[str, float]] = []
    order = np.arange(len(train_triples))

    for epoch in range(epochs):
        rng.shuffle(order)
        total_loss = 0.0
        updates = 0

        for idx in order:
            head_id, relation_id, tail_id = train_triples[idx]
            negative_head_id = int(head_id)
            negative_relation_id = int(relation_id)
            negative_tail_id = int(tail_id)

            corruption_mode = int(rng.integers(3))
            if corruption_mode == 0:
                negative_head_id = sample_alternative(rng, num_entities, int(head_id))
            elif corruption_mode == 1:
                negative_relation_id = sample_alternative(
                    rng, num_relations, int(relation_id)
                )
            else:
                negative_tail_id = sample_alternative(rng, num_entities, int(tail_id))

            head = entity_embeddings[head_id]
            relation = relation_embeddings[relation_id]
            tail = entity_embeddings[tail_id]
            negative_head = entity_embeddings[negative_head_id]
            negative_relation = relation_embeddings[negative_relation_id]
            negative_tail = entity_embeddings[negative_tail_id]

            positive_delta = head + relation - tail
            negative_delta = negative_head + negative_relation - negative_tail
            positive_distance = float(np.linalg.norm(positive_delta))
            negative_distance = float(np.linalg.norm(negative_delta))

            loss = margin + positive_distance - negative_distance
            if loss <= 0.0:
                continue

            updates += 1
            total_loss += loss

            positive_grad = positive_delta / max(positive_distance, 1e-12)
            negative_grad = negative_delta / max(negative_distance, 1e-12)

            entity_embeddings[head_id] -= learning_rate * positive_grad
            relation_embeddings[relation_id] -= learning_rate * positive_grad
            entity_embeddings[tail_id] += learning_rate * positive_grad

            if corruption_mode == 0:
                entity_embeddings[negative_head_id] += learning_rate * negative_grad
                relation_embeddings[relation_id] += learning_rate * negative_grad
                entity_embeddings[tail_id] -= learning_rate * negative_grad
            elif corruption_mode == 1:
                entity_embeddings[head_id] += learning_rate * negative_grad
                relation_embeddings[negative_relation_id] += learning_rate * negative_grad
                entity_embeddings[tail_id] -= learning_rate * negative_grad
            else:
                entity_embeddings[head_id] += learning_rate * negative_grad
                relation_embeddings[relation_id] += learning_rate * negative_grad
                entity_embeddings[negative_tail_id] -= learning_rate * negative_grad

        normalize_rows(entity_embeddings)
        avg_loss = total_loss / max(updates, 1)
        history.append(
            {
                "epoch": epoch + 1,
                "avg_margin_loss": avg_loss,
                "updates": float(updates),
            }
        )
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"updates={updates} avg_margin_loss={avg_loss:.6f}"
        )

    return entity_embeddings, relation_embeddings, history


def compute_tail_ranks(
    entity_embeddings: np.ndarray,
    relation_embeddings: np.ndarray,
    test_triples: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    entity_sq_norms = np.sum(entity_embeddings * entity_embeddings, axis=1)
    ranks: List[np.ndarray] = []

    for start in range(0, len(test_triples), batch_size):
        batch = test_triples[start : start + batch_size]
        heads = entity_embeddings[batch[:, 0]]
        relations = relation_embeddings[batch[:, 1]]
        queries = heads + relations

        query_sq_norms = np.sum(queries * queries, axis=1, keepdims=True)
        distances = (
            query_sq_norms
            + entity_sq_norms[np.newaxis, :]
            - 2.0 * queries @ entity_embeddings.T
        )
        true_tail_scores = distances[np.arange(len(batch)), batch[:, 2]]
        batch_ranks = np.sum(distances < true_tail_scores[:, np.newaxis], axis=1) + 1
        ranks.append(batch_ranks.astype(np.int64))

    return np.concatenate(ranks, axis=0) if ranks else np.empty(0, dtype=np.int64)


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

    train_triples, test_triples = split_triples(triples, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_triples(args.output_dir / "train_triples.csv", train_triples)
    write_triples(args.output_dir / "test_triples.csv", test_triples)

    entity_embeddings, relation_embeddings, history = train_transe(
        train_triples=train_triples,
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=embedding_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        margin=args.margin,
        seed=args.seed,
    )

    print(
        f"Evaluating {len(test_triples)} held-out triples with tail prediction "
        f"(h, r, ?)"
    )
    tail_ranks = compute_tail_ranks(
        entity_embeddings=entity_embeddings,
        relation_embeddings=relation_embeddings,
        test_triples=test_triples,
        batch_size=args.eval_batch_size,
    )
    mean_rank = float(np.mean(tail_ranks)) if len(tail_ranks) else float("nan")

    np.save(args.output_dir / "entity_embeddings.npy", entity_embeddings)
    np.save(args.output_dir / "relation_embeddings.npy", relation_embeddings)

    metrics = {
        "model": "TransE",
        "transductive": True,
        "train_fraction": 0.8,
        "num_entities": num_entities,
        "num_relations": num_relations,
        "num_train_triples": int(len(train_triples)),
        "num_test_triples": int(len(test_triples)),
        "embedding_dim": embedding_dim,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "margin": args.margin,
        "seed": args.seed,
        "evaluation": {
            "task": "tail link prediction",
            "query_form": "(h, r, ?)",
            "ranking": "increasing TransE distance",
            "mean_rank": mean_rank,
        },
        "training_history": history,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(f"Train triples: {len(train_triples)}")
    print(f"Test triples: {len(test_triples)}")
    print(f"Mean rank: {mean_rank:.3f}")
    print(f"Artifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
