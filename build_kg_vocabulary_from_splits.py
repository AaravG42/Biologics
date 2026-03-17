#!/usr/bin/env python3
"""Build KG vocabularies and random embeddings from combined split TXT files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


Triple = Tuple[str, str, str]
IndexedTriple = Tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create integer vocabularies, indexed triples, and random embedding "
            "matrices from combined train/valid/test TXT triple files."
        )
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/train.txt"),
        help="Path to the train split TXT file",
    )
    parser.add_argument(
        "--valid",
        type=Path,
        default=Path("data/valid.txt"),
        help="Path to the validation split TXT file",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/test.txt"),
        help="Path to the test split TXT file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/kg_vocab_onco_filtered"),
        help="Directory where vocabularies, triples, and embeddings will be written",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=100,
        help="Embedding dimension for both entity and relation matrices",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible embedding initialization",
    )
    return parser.parse_args()


def load_triples(path: Path) -> List[Triple]:
    triples: List[Triple] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(
                    f"Expected 3 tab-separated columns in {path}:{line_number}, got {len(parts)}"
                )
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def build_entity_vocab(triples: Sequence[Triple]) -> Dict[str, int]:
    entity_to_id: Dict[str, int] = {}
    for head, _, tail in triples:
        if head not in entity_to_id:
            entity_to_id[head] = len(entity_to_id)
        if tail not in entity_to_id:
            entity_to_id[tail] = len(entity_to_id)
    return entity_to_id


def build_relation_vocab(triples: Sequence[Triple]) -> Dict[str, int]:
    relation_to_id: Dict[str, int] = {}
    for _, relation, _ in triples:
        if relation not in relation_to_id:
            relation_to_id[relation] = len(relation_to_id)
    return relation_to_id


def index_triples(
    triples: Iterable[Triple],
    entity_to_id: Dict[str, int],
    relation_to_id: Dict[str, int],
) -> List[IndexedTriple]:
    indexed: List[IndexedTriple] = []
    for head, relation, tail in triples:
        indexed.append(
            (entity_to_id[head], relation_to_id[relation], entity_to_id[tail])
        )
    return indexed


def initialize_embeddings(
    num_rows: int, embedding_dim: int, rng: np.random.Generator
) -> np.ndarray:
    limit = 0.01
    return rng.uniform(-limit, limit, size=(num_rows, embedding_dim)).astype(np.float32)


def write_vocab(path: Path, mapping: Dict[str, int], key_name: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([key_name, "id"])
        for value, idx in mapping.items():
            writer.writerow([value, idx])


def write_indexed_triples(path: Path, triples: Sequence[IndexedTriple]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["head_id", "relation_id", "tail_id"])
        writer.writerows(triples)


def write_metadata(
    path: Path,
    train_path: Path,
    valid_path: Path,
    test_path: Path,
    output_dir: Path,
    embedding_dim: int,
    train_triples: Sequence[Triple],
    valid_triples: Sequence[Triple],
    test_triples: Sequence[Triple],
    entity_to_id: Dict[str, int],
    relation_to_id: Dict[str, int],
    indexed_triples: Sequence[IndexedTriple],
) -> None:
    metadata = {
        "input_splits": {
            "train": str(train_path),
            "valid": str(valid_path),
            "test": str(test_path),
        },
        "output_dir": str(output_dir),
        "num_entities": len(entity_to_id),
        "num_relations": len(relation_to_id),
        "num_triples": len(indexed_triples),
        "split_counts": {
            "train": len(train_triples),
            "valid": len(valid_triples),
            "test": len(test_triples),
        },
        "embedding_dim": embedding_dim,
        "files": {
            "entities": "entities.csv",
            "relations": "relations.csv",
            "triples": "triples_ids.csv",
            "entity_embeddings": "entity_embeddings.npy",
            "relation_embeddings": "relation_embeddings.npy",
        },
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    train_triples = load_triples(args.train)
    valid_triples = load_triples(args.valid)
    test_triples = load_triples(args.test)
    combined_triples = [*train_triples, *valid_triples, *test_triples]

    entity_to_id = build_entity_vocab(combined_triples)
    relation_to_id = build_relation_vocab(combined_triples)
    indexed_triples = index_triples(combined_triples, entity_to_id, relation_to_id)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_vocab(args.output_dir / "entities.csv", entity_to_id, "entity")
    write_vocab(args.output_dir / "relations.csv", relation_to_id, "relation")
    write_indexed_triples(args.output_dir / "triples_ids.csv", indexed_triples)

    rng = np.random.default_rng(args.seed)
    entity_embeddings = initialize_embeddings(
        len(entity_to_id), args.embedding_dim, rng
    )
    relation_embeddings = initialize_embeddings(
        len(relation_to_id), args.embedding_dim, rng
    )
    np.save(args.output_dir / "entity_embeddings.npy", entity_embeddings)
    np.save(args.output_dir / "relation_embeddings.npy", relation_embeddings)

    write_metadata(
        args.output_dir / "metadata.json",
        args.train,
        args.valid,
        args.test,
        args.output_dir,
        args.embedding_dim,
        train_triples,
        valid_triples,
        test_triples,
        entity_to_id,
        relation_to_id,
        indexed_triples,
    )

    print(f"Loaded {len(train_triples)} train triples from {args.train}")
    print(f"Loaded {len(valid_triples)} valid triples from {args.valid}")
    print(f"Loaded {len(test_triples)} test triples from {args.test}")
    print(f"Combined triples: {len(combined_triples)}")
    print(f"Entities: {len(entity_to_id)}")
    print(f"Relations: {len(relation_to_id)}")
    print(f"Entity embeddings shape: {entity_embeddings.shape}")
    print(f"Relation embeddings shape: {relation_embeddings.shape}")
    print(f"Artifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
