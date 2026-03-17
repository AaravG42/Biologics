#!/usr/bin/env python3
"""Build entity/relation vocabularies and random embeddings from Query.csv."""

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
            "matrices from an RDF-style knowledge graph CSV."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/Query.csv"),
        help="Path to the source CSV with columns: sub,pred,obj",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/kg_vocab"),
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
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = {"sub", "pred", "obj"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError(
                f"Expected columns {sorted(expected)}, got {reader.fieldnames}"
            )
        for row in reader:
            triples.append((row["sub"], row["pred"], row["obj"]))
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
    input_path: Path,
    output_dir: Path,
    embedding_dim: int,
    entity_to_id: Dict[str, int],
    relation_to_id: Dict[str, int],
    indexed_triples: Sequence[IndexedTriple],
) -> None:
    metadata = {
        "input_csv": str(input_path),
        "output_dir": str(output_dir),
        "num_entities": len(entity_to_id),
        "num_relations": len(relation_to_id),
        "num_triples": len(indexed_triples),
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
    triples = load_triples(args.input)

    entity_to_id = build_entity_vocab(triples)
    relation_to_id = build_relation_vocab(triples)
    indexed_triples = index_triples(triples, entity_to_id, relation_to_id)

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
        args.input,
        args.output_dir,
        args.embedding_dim,
        entity_to_id,
        relation_to_id,
        indexed_triples,
    )

    print(f"Loaded {len(triples)} triples from {args.input}")
    print(f"Entities: {len(entity_to_id)}")
    print(f"Relations: {len(relation_to_id)}")
    print(f"Entity embeddings shape: {entity_embeddings.shape}")
    print(f"Relation embeddings shape: {relation_embeddings.shape}")
    print(f"Artifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
