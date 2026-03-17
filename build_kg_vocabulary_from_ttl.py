#!/usr/bin/env python3
"""Build KG vocabularies and random embeddings from a Turtle RDF file."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from rdflib import BNode, Graph, Literal, URIRef
from rdflib.term import Identifier


Triple = Tuple[str, str, str]
IndexedTriple = Tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create integer vocabularies, indexed triples, and random embedding "
            "matrices from an RDF Turtle knowledge graph."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/IMGT_ONTO_ABOX_MAB_ONCOLOGY_ONLY.ttl"),
        help="Path to the source Turtle file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/kg_vocab_oncology_ttl"),
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
    parser.add_argument(
        "--object-mode",
        type=str,
        choices=["all", "resources-only"],
        default="all",
        help=(
            "How to handle object terms: 'all' keeps URI, blank-node, and literal "
            "objects; 'resources-only' drops triples whose object is a literal."
        ),
    )
    return parser.parse_args()


def term_to_string(term: Identifier) -> str:
    if isinstance(term, URIRef):
        return str(term)
    if isinstance(term, BNode):
        return f"_:{term}"
    if isinstance(term, Literal):
        lexical = str(term)
        if term.language:
            return f"{lexical}@{term.language}"
        if term.datatype:
            return f"{lexical}^^{term.datatype}"
        return lexical
    raise TypeError(f"Unsupported RDF term type: {type(term)!r}")


def summarize_term_types(graph: Graph) -> Dict[str, Dict[str, int]]:
    return {
        "subjects": Counter(type(term).__name__ for term in graph.subjects()),
        "predicates": Counter(type(term).__name__ for term in graph.predicates()),
        "objects": Counter(type(term).__name__ for term in graph.objects()),
    }


def load_triples(
    path: Path, object_mode: str
) -> Tuple[List[Triple], int, Dict[str, Dict[str, int]]]:
    graph = Graph()
    graph.parse(path, format="turtle")
    term_summary = summarize_term_types(graph)

    triples: List[Triple] = []
    dropped = 0
    for subject, predicate, object_ in graph:
        if object_mode == "resources-only" and isinstance(object_, Literal):
            dropped += 1
            continue
        triples.append(
            (
                term_to_string(subject),
                term_to_string(predicate),
                term_to_string(object_),
            )
        )
    return triples, dropped, term_summary


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
    object_mode: str,
    num_input_triples: int,
    num_dropped_triples: int,
    entity_to_id: Dict[str, int],
    relation_to_id: Dict[str, int],
    indexed_triples: Sequence[IndexedTriple],
    term_summary: Dict[str, Dict[str, int]],
) -> None:
    metadata = {
        "input_ttl": str(input_path),
        "output_dir": str(output_dir),
        "object_mode": object_mode,
        "num_input_triples": num_input_triples,
        "num_dropped_triples": num_dropped_triples,
        "num_entities": len(entity_to_id),
        "num_relations": len(relation_to_id),
        "num_triples": len(indexed_triples),
        "embedding_dim": embedding_dim,
        "rdf_term_summary": term_summary,
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
    triples, dropped, term_summary = load_triples(args.input, args.object_mode)

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
        args.object_mode,
        len(triples) + dropped,
        dropped,
        entity_to_id,
        relation_to_id,
        indexed_triples,
        term_summary,
    )

    print(f"Loaded {len(triples) + dropped} triples from {args.input}")
    if args.object_mode == "resources-only":
        print(f"Dropped literal-object triples: {dropped}")
    print(f"Kept triples: {len(triples)}")
    print(f"Entities: {len(entity_to_id)}")
    print(f"Relations: {len(relation_to_id)}")
    print(f"Entity embeddings shape: {entity_embeddings.shape}")
    print(f"Relation embeddings shape: {relation_embeddings.shape}")
    print(f"Artifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
