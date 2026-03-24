#!/usr/bin/env python3
"""Build KG vocabularies with ESM-based mAb entity embeddings from split TXT files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


Triple = Tuple[str, str, str]
IndexedTriple = Tuple[int, int, int]

MAB_PATTERN = re.compile(r"imgt:mAb_(\d+)$")
INN_PATTERN = re.compile(r"imgt:(\d+)$")
STRUCTURE_LINK_RELATION = "imgt:isLinkedToStructureAccessNumb"
ENTITY_EMBEDDING_DIM = 1280
ESM_EMBEDDING_DIM = 640


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create integer vocabularies, indexed triples, and embedding matrices "
            "from combined train/valid/test TXT triple files. mAb entity embeddings "
            "are built by concatenating heavy and light chain ESM embeddings when "
            "a sequence mapping is available."
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
        "--sequences-csv",
        type=Path,
        default=Path("data/mab_inn_sequences_split.csv"),
        help="CSV containing inn_number, heavy_chain_sequence, and light_chain_sequence",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/kg_vocab_onco_filtered_ESM_640"),
        help="Directory where vocabularies, triples, and embeddings will be written",
    )
    parser.add_argument(
        "--relation-embedding-dim",
        type=int,
        default=ENTITY_EMBEDDING_DIM,
        help="Embedding dimension for relation embeddings",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible embedding initialization",
    )
    parser.add_argument(
        "--esm-model",
        type=str,
        default="esm2_t30_150M_UR50D",
        help="ESM model identifier. Uses facebook/<name> for transformers if needed.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for ESM sequence embedding",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Embedding device. 'auto' prefers CUDA when available.",
    )
    parser.add_argument(
        "--max-residue-length",
        type=int,
        default=1022,
        help="Maximum residues per chain before truncation for ESM2 inputs",
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


def normalize_transformers_model_name(model_name: str) -> str:
    return model_name if "/" in model_name else f"facebook/{model_name}"


def load_sequence_rows(path: Path) -> Dict[str, Tuple[str, str]]:
    sequences_by_inn: Dict[str, Tuple[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"inn_number", "heavy_chain_sequence", "light_chain_sequence"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError(
                f"Expected columns {sorted(expected)} in {path}, got {reader.fieldnames}"
            )
        for row in reader:
            inn_number = row["inn_number"].strip().strip('"')
            heavy = row["heavy_chain_sequence"].strip().upper()
            light = row["light_chain_sequence"].strip().upper()
            sequences_by_inn[inn_number] = (heavy, light)
    return sequences_by_inn


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def extract_mab_to_inn_mapping(triples: Sequence[Triple]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for head, relation, tail in triples:
        mab_match = MAB_PATTERN.fullmatch(head)
        inn_match = INN_PATTERN.fullmatch(tail)
        if mab_match and relation == STRUCTURE_LINK_RELATION and inn_match:
            mapping[head] = inn_match.group(1)
    return mapping


def get_mab_entities(entity_to_id: Dict[str, int]) -> List[str]:
    return [entity for entity in entity_to_id if MAB_PATTERN.fullmatch(entity)]


def truncate_sequence(sequence: str, max_residue_length: int) -> Tuple[str, bool]:
    if len(sequence) <= max_residue_length:
        return sequence, False
    return sequence[:max_residue_length], True


def load_embedding_backend(model_name: str):
    errors: List[str] = []

    try:
        from transformers import AutoTokenizer, EsmModel  # type: ignore

        hf_name = normalize_transformers_model_name(model_name)
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        model = EsmModel.from_pretrained(hf_name)
        if int(model.config.hidden_size) != ESM_EMBEDDING_DIM:
            raise ValueError(
                f"Expected ESM hidden size {ESM_EMBEDDING_DIM}, got {model.config.hidden_size}"
            )
        return ("transformers", tokenizer, model)
    except Exception as exc:
        errors.append(f"transformers backend unavailable: {exc}")

    try:
        import esm  # type: ignore

        loader = getattr(esm.pretrained, model_name, None)
        if loader is None:
            raise ValueError(
                f"esm.pretrained has no loader named {model_name}"
            )
        model, alphabet = loader()
        if int(model.embed_dim) != ESM_EMBEDDING_DIM:
            raise ValueError(
                f"Expected ESM hidden size {ESM_EMBEDDING_DIM}, got {model.embed_dim}"
            )
        batch_converter = alphabet.get_batch_converter()
        return ("esm", batch_converter, model)
    except Exception as exc:
        errors.append(f"esm backend unavailable: {exc}")

    raise RuntimeError(
        "Could not load an ESM backend. "
        "Tried transformers and fair-esm. "
        + " | ".join(errors)
    )


def embed_sequences_transformers(
    sequences: Sequence[str],
    tokenizer,
    model,
    device: str,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    model = model.to(torch.device(device))
    model.eval()
    embeddings: Dict[str, np.ndarray] = {}

    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch_sequences = list(sequences[start : start + batch_size])
            encoded = tokenizer(
                batch_sequences,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=True,
            )
            attention_mask = encoded["attention_mask"]
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state.detach().cpu()
            attention_mask = attention_mask.cpu()

            for index, sequence in enumerate(batch_sequences):
                token_count = len(sequence)
                pooled = hidden[index, 1 : token_count + 1].mean(dim=0)
                embeddings[sequence] = pooled.numpy().astype(np.float32, copy=False)

    return embeddings


def embed_sequences_esm(
    sequences: Sequence[str],
    batch_converter,
    model,
    device: str,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    model = model.to(torch.device(device))
    model.eval()
    embeddings: Dict[str, np.ndarray] = {}
    num_layers = len(model.layers)

    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch_sequences = list(sequences[start : start + batch_size])
            batch = [(str(i), sequence) for i, sequence in enumerate(batch_sequences)]
            _, _, tokens = batch_converter(batch)
            tokens = tokens.to(device)
            outputs = model(tokens, repr_layers=[num_layers], return_contacts=False)
            representations = outputs["representations"][num_layers].detach().cpu()

            for index, sequence in enumerate(batch_sequences):
                pooled = representations[index, 1 : len(sequence) + 1].mean(dim=0)
                embeddings[sequence] = pooled.numpy().astype(np.float32, copy=False)

    return embeddings


def embed_unique_sequences(
    sequences: Sequence[str],
    model_name: str,
    device: str,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    backend = load_embedding_backend(model_name)
    backend_name = backend[0]
    if backend_name == "transformers":
        _, tokenizer, model = backend
        return embed_sequences_transformers(
            sequences=sequences,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=batch_size,
        )
    _, batch_converter, model = backend
    return embed_sequences_esm(
        sequences=sequences,
        batch_converter=batch_converter,
        model=model,
        device=device,
        batch_size=batch_size,
    )


def build_mab_entity_embeddings(
    entity_to_id: Dict[str, int],
    triples: Sequence[Triple],
    sequences_by_inn: Dict[str, Tuple[str, str]],
    rng: np.random.Generator,
    model_name: str,
    device: str,
    batch_size: int,
    max_residue_length: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    entity_embeddings = initialize_embeddings(
        len(entity_to_id), ENTITY_EMBEDDING_DIM, rng
    )
    mab_entities = get_mab_entities(entity_to_id)
    mab_to_inn = extract_mab_to_inn_mapping(triples)

    valid_rows: Dict[str, Tuple[str, str]] = {}
    missing_inn_links: List[str] = []
    missing_sequences: List[str] = []
    empty_sequence_inns: List[str] = []
    truncated_inns: List[str] = []
    truncated_sequences: Dict[str, Tuple[str, str]] = {}

    for mab_entity in mab_entities:
        inn_number = mab_to_inn.get(mab_entity)
        if inn_number is None:
            missing_inn_links.append(mab_entity)
            continue
        sequences = sequences_by_inn.get(inn_number)
        if sequences is None:
            missing_sequences.append(mab_entity)
            continue
        heavy, light = sequences
        if not heavy or not light:
            empty_sequence_inns.append(inn_number)
            continue
        truncated_heavy, heavy_was_truncated = truncate_sequence(
            heavy, max_residue_length
        )
        truncated_light, light_was_truncated = truncate_sequence(
            light, max_residue_length
        )
        if heavy_was_truncated or light_was_truncated:
            truncated_inns.append(inn_number)
        valid_rows[inn_number] = (truncated_heavy, truncated_light)
        truncated_sequences[inn_number] = (truncated_heavy, truncated_light)

    unique_sequences = sorted(
        {
            sequence
            for heavy, light in truncated_sequences.values()
            for sequence in (heavy, light)
        }
    )

    if unique_sequences:
        sequence_to_embedding = embed_unique_sequences(
            sequences=unique_sequences,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
        )
    else:
        sequence_to_embedding = {}

    overwritten_entities: List[str] = []
    for mab_entity in mab_entities:
        inn_number = mab_to_inn.get(mab_entity)
        if inn_number is None or inn_number not in valid_rows:
            continue
        heavy, light = valid_rows[inn_number]
        heavy_embedding = sequence_to_embedding[heavy]
        light_embedding = sequence_to_embedding[light]
        entity_embeddings[entity_to_id[mab_entity]] = np.concatenate(
            [heavy_embedding, light_embedding]
        ).astype(np.float32, copy=False)
        overwritten_entities.append(mab_entity)

    summary = {
        "num_mab_entities_in_vocab": len(mab_entities),
        "num_mab_entities_with_inn_link": len(mab_to_inn),
        "num_mab_entities_with_sequence_embeddings": len(overwritten_entities),
        "num_mab_entities_random_fallback": len(mab_entities) - len(overwritten_entities),
        "num_mab_entities_missing_inn_link": len(missing_inn_links),
        "num_mab_entities_missing_sequence_row": len(missing_sequences),
        "num_inn_rows_with_empty_chain_sequence": len(set(empty_sequence_inns)),
        "num_inn_rows_truncated": len(set(truncated_inns)),
        "mab_entities_missing_inn_link_sample": sorted(missing_inn_links)[:10],
        "mab_entities_missing_sequence_row_sample": sorted(missing_sequences)[:10],
        "inn_rows_with_empty_chain_sequence_sample": sorted(set(empty_sequence_inns))[:10],
        "truncated_inn_rows_sample": sorted(set(truncated_inns))[:10],
    }
    return entity_embeddings, summary


def write_metadata(
    path: Path,
    train_path: Path,
    valid_path: Path,
    test_path: Path,
    sequences_csv_path: Path,
    output_dir: Path,
    relation_embedding_dim: int,
    train_triples: Sequence[Triple],
    valid_triples: Sequence[Triple],
    test_triples: Sequence[Triple],
    entity_to_id: Dict[str, int],
    relation_to_id: Dict[str, int],
    indexed_triples: Sequence[IndexedTriple],
    mapping_summary: Dict[str, object],
    args: argparse.Namespace,
) -> None:
    metadata = {
        "input_splits": {
            "train": str(train_path),
            "valid": str(valid_path),
            "test": str(test_path),
        },
        "sequence_source": str(sequences_csv_path),
        "output_dir": str(output_dir),
        "num_entities": len(entity_to_id),
        "num_relations": len(relation_to_id),
        "num_triples": len(indexed_triples),
        "split_counts": {
            "train": len(train_triples),
            "valid": len(valid_triples),
            "test": len(test_triples),
        },
        "embedding_dim": ENTITY_EMBEDDING_DIM,
        "entity_embedding_dim": ENTITY_EMBEDDING_DIM,
        "relation_embedding_dim": relation_embedding_dim,
        "mab_entity_embedding_source": {
            "relation_used_for_inn_mapping": STRUCTURE_LINK_RELATION,
            "esm_model": args.esm_model,
            "esm_hidden_size_per_chain": ESM_EMBEDDING_DIM,
            "entity_embedding_layout": "concat(heavy_chain, light_chain)",
            "sequence_pooling": "mean over residue token embeddings excluding special tokens",
            "max_residue_length": args.max_residue_length,
            "random_fallback_for_non_mab_or_unmapped": True,
        },
        "mapping_summary": mapping_summary,
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
    sequences_by_inn = load_sequence_rows(args.sequences_csv)

    entity_to_id = build_entity_vocab(combined_triples)
    relation_to_id = build_relation_vocab(combined_triples)
    indexed_triples = index_triples(combined_triples, entity_to_id, relation_to_id)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_vocab(args.output_dir / "entities.csv", entity_to_id, "entity")
    write_vocab(args.output_dir / "relations.csv", relation_to_id, "relation")
    write_indexed_triples(args.output_dir / "triples_ids.csv", indexed_triples)

    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    entity_embeddings, mapping_summary = build_mab_entity_embeddings(
        entity_to_id=entity_to_id,
        triples=combined_triples,
        sequences_by_inn=sequences_by_inn,
        rng=rng,
        model_name=args.esm_model,
        device=device,
        batch_size=args.batch_size,
        max_residue_length=args.max_residue_length,
    )
    relation_embeddings = initialize_embeddings(
        len(relation_to_id), args.relation_embedding_dim, rng
    )
    np.save(args.output_dir / "entity_embeddings.npy", entity_embeddings)
    np.save(args.output_dir / "relation_embeddings.npy", relation_embeddings)

    write_metadata(
        args.output_dir / "metadata.json",
        args.train,
        args.valid,
        args.test,
        args.sequences_csv,
        args.output_dir,
        args.relation_embedding_dim,
        train_triples,
        valid_triples,
        test_triples,
        entity_to_id,
        relation_to_id,
        indexed_triples,
        mapping_summary,
        args,
    )

    print(f"Loaded {len(train_triples)} train triples from {args.train}")
    print(f"Loaded {len(valid_triples)} valid triples from {args.valid}")
    print(f"Loaded {len(test_triples)} test triples from {args.test}")
    print(f"Combined triples: {len(combined_triples)}")
    print(f"Entities: {len(entity_to_id)}")
    print(f"Relations: {len(relation_to_id)}")
    print(f"Embedding device: {device}")
    print(f"Entity embeddings shape: {entity_embeddings.shape}")
    print(f"Relation embeddings shape: {relation_embeddings.shape}")
    print("mAb mapping summary:")
    print(json.dumps(mapping_summary, indent=2))
    print(f"Artifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
