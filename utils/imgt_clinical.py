"""Helpers for IMGT clinical-indication experiments."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from datasets.process import get_idx

Triple = Tuple[str, str, str]

MAB_PATTERN = re.compile(r"imgt:mAb_(\d+)$")
INN_PATTERN = re.compile(r"imgt:(\d+)$")
STRUCTURE_LINK_RELATION = "imgt:isLinkedToStructureAccessNumb"
CLINICAL_INDICATION_RELATION = "imgt:hasClinicalIndication"
CLINICAL_INDICATION_INVERSE_RELATION = "imgt:isClinicalIndicationOf"


def resolve_device(requested: str) -> str:
    """Resolve a training/eval device from a user request."""
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def enable_cpu_fallback(device: str) -> str:
    """Patch hardcoded .cuda() usage when running on CPU."""
    if device == "cuda":
        return device
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self
    return device


def load_raw_triples(dataset_dir: Path) -> List[Triple]:
    """Load raw train/valid/test triples from an IMGT dataset directory."""
    triples: List[Triple] = []
    for split in ("train", "valid", "test"):
        split_path = dataset_dir / split
        with split_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 3:
                    raise ValueError(
                        f"Expected 3 tab-separated values in {split_path}:{line_number}"
                    )
                triples.append((parts[0], parts[1], parts[2]))
    return triples


def load_sequence_rows(path: Path) -> Dict[str, Tuple[str, str]]:
    """Load heavy/light chain sequences keyed by INN number."""
    sequences_by_inn: Dict[str, Tuple[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
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


def extract_mab_to_inn_mapping(triples: Sequence[Triple]) -> Dict[str, str]:
    """Recover mAb entity to INN mappings from KG triples."""
    mapping: Dict[str, str] = {}
    for head, relation, tail in triples:
        mab_match = MAB_PATTERN.fullmatch(head)
        inn_match = INN_PATTERN.fullmatch(tail)
        if mab_match and relation == STRUCTURE_LINK_RELATION and inn_match:
            mapping[head] = inn_match.group(1)
    return mapping


def load_entity_name_to_embedding(
    entities_csv: Path,
    embeddings_npy: Path,
) -> Tuple[Dict[str, np.ndarray], int]:
    """Load Biologics entity embeddings keyed by raw entity string."""
    embeddings = np.load(embeddings_npy)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings in {embeddings_npy}, got {embeddings.shape}")

    name_to_id: Dict[str, int] = {}
    with entities_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"entity", "id"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError(
                f"Expected columns {sorted(expected)} in {entities_csv}, got {reader.fieldnames}"
            )
        for row in reader:
            name_to_id[row["entity"]] = int(row["id"])

    if len(name_to_id) != embeddings.shape[0]:
        raise ValueError(
            f"Entity count mismatch between {entities_csv} and {embeddings_npy}: "
            f"{len(name_to_id)} != {embeddings.shape[0]}"
        )

    return {name: embeddings[idx] for name, idx in name_to_id.items()}, int(embeddings.shape[1])


def build_additive_imgt_entity_init(
    dataset_dir: Path,
    biologics_entities_csv: Path,
    biologics_embeddings_npy: Path,
    sequences_csv: Path,
    output_dir: Path,
    init_dim: int = 640,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Build local IMGT entity initialization with additive heavy/light mAb priors."""
    output_dir.mkdir(parents=True, exist_ok=True)
    init_path = output_dir / "entity_init.npy"
    metadata_path = output_dir / "entity_init_metadata.json"
    if init_path.is_file() and metadata_path.is_file():
        entity_init = np.load(init_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if entity_init.shape[1] == init_dim:
            return entity_init, metadata

    entity_to_id, relation_to_id = get_idx(str(dataset_dir))
    triples = load_raw_triples(dataset_dir)
    sequences_by_inn = load_sequence_rows(sequences_csv)
    mab_to_inn = extract_mab_to_inn_mapping(triples)
    name_to_embedding, source_dim = load_entity_name_to_embedding(
        biologics_entities_csv, biologics_embeddings_npy
    )

    if source_dim % 2 != 0:
        raise ValueError(
            f"Expected even-dimension Biologics embeddings for additive split, got {source_dim}"
        )
    half_dim = source_dim // 2
    if half_dim != init_dim:
        raise ValueError(
            f"Expected additive init dim {init_dim} from Biologics half-dim, got {half_dim}"
        )

    rng = np.random.default_rng(seed)
    entity_init = rng.uniform(-0.01, 0.01, size=(len(entity_to_id), init_dim)).astype(np.float32)

    initialized_mabs: List[str] = []
    missing_entity_rows: List[str] = []
    missing_inn_links: List[str] = []
    empty_sequence_rows: List[str] = []

    for entity_name, entity_id in entity_to_id.items():
        if not MAB_PATTERN.fullmatch(entity_name):
            continue
        inn_number = mab_to_inn.get(entity_name)
        if inn_number is None:
            missing_inn_links.append(entity_name)
            continue
        heavy_light = sequences_by_inn.get(inn_number)
        if heavy_light is None or not heavy_light[0] or not heavy_light[1]:
            empty_sequence_rows.append(entity_name)
            continue
        source_embedding = name_to_embedding.get(entity_name)
        if source_embedding is None:
            missing_entity_rows.append(entity_name)
            continue
        entity_init[entity_id] = (
            source_embedding[:half_dim] + source_embedding[half_dim:]
        ).astype(np.float32, copy=False)
        initialized_mabs.append(entity_name)

    metadata = {
        "dataset_dir": str(dataset_dir),
        "biologics_entities_csv": str(biologics_entities_csv),
        "biologics_embeddings_npy": str(biologics_embeddings_npy),
        "sequences_csv": str(sequences_csv),
        "num_entities": len(entity_to_id),
        "num_relations": len(relation_to_id),
        "init_dim": init_dim,
        "source_embedding_dim": source_dim,
        "fusion": "add(first_half, second_half)",
        "clinical_relations": [
            CLINICAL_INDICATION_RELATION,
            CLINICAL_INDICATION_INVERSE_RELATION,
        ],
        "num_mab_entities_initialized": len(initialized_mabs),
        "num_mab_entities_missing_inn_link": len(missing_inn_links),
        "num_mab_entities_missing_or_empty_sequence": len(set(empty_sequence_rows)),
        "num_mab_entities_missing_source_embedding": len(missing_entity_rows),
        "initialized_mab_sample": sorted(initialized_mabs)[:10],
        "missing_inn_link_sample": sorted(missing_inn_links)[:10],
        "missing_or_empty_sequence_sample": sorted(set(empty_sequence_rows))[:10],
        "missing_source_embedding_sample": sorted(missing_entity_rows)[:10],
    }
    np.save(init_path, entity_init)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return entity_init, metadata


def load_clinical_relation_ids(dataset_dir: Path) -> List[int]:
    """Resolve both clinical-indication relation ids used for evaluation."""
    _, relation_to_id = get_idx(str(dataset_dir))
    relation_names = [
        CLINICAL_INDICATION_RELATION,
        CLINICAL_INDICATION_INVERSE_RELATION,
    ]
    missing = [name for name in relation_names if name not in relation_to_id]
    if missing:
        raise ValueError(f"Could not resolve clinical indication relations {missing} from {dataset_dir}")
    return [int(relation_to_id[name]) for name in relation_names]


def load_forward_clinical_relation_id(dataset_dir: Path) -> int:
    """Resolve the forward clinical-indication relation id."""
    _, relation_to_id = get_idx(str(dataset_dir))
    if CLINICAL_INDICATION_RELATION not in relation_to_id:
        raise ValueError(
            f"Could not resolve forward clinical indication relation "
            f"{CLINICAL_INDICATION_RELATION} from {dataset_dir}"
        )
    return int(relation_to_id[CLINICAL_INDICATION_RELATION])


def load_mab_entity_ids(dataset_dir: Path) -> List[int]:
    """Return sorted entity ids corresponding to IMGT mAb nodes."""
    entity_to_id, _ = get_idx(str(dataset_dir))
    mab_ids = [int(entity_id) for entity_name, entity_id in entity_to_id.items() if MAB_PATTERN.fullmatch(entity_name)]
    if not mab_ids:
        raise ValueError(f"Could not find any mAb entities in {dataset_dir}")
    return sorted(mab_ids)


def load_forward_clinical_head_entity_ids(dataset_dir: Path) -> List[int]:
    """Return sorted entity ids appearing as heads of forward clinical triples."""
    entity_to_id, _ = get_idx(str(dataset_dir))
    triples = load_raw_triples(dataset_dir)
    head_ids = sorted(
        {
            int(entity_to_id[head])
            for head, relation, _ in triples
            if relation == CLINICAL_INDICATION_RELATION
        }
    )
    if not head_ids:
        raise ValueError(f"Could not find forward clinical heads in {dataset_dir}")
    return head_ids


def filter_triples_by_relation_ids(triples: torch.Tensor, relation_ids: Sequence[int]) -> torch.Tensor:
    """Keep only triples whose relation id is in the requested set."""
    if not relation_ids:
        return triples[:0]
    relation_id_tensor = torch.as_tensor(relation_ids, device=triples.device, dtype=triples.dtype)
    return triples[torch.isin(triples[:, 1], relation_id_tensor)]


def compute_rhs_metrics(
    model,
    triples: torch.Tensor,
    rhs_filters: Dict[Tuple[int, int], List[int]],
    batch_size: int = 500,
) -> Dict[str, float]:
    """Compute tail-only ranking metrics for a batch of triples."""
    if triples.shape[0] == 0:
        return {
            "count": 0,
            "tail_mr": float("inf"),
            "tail_mrr": 0.0,
            "tail_hits@1": 0.0,
            "tail_hits@3": 0.0,
            "tail_hits@10": 0.0,
        }
    ranks = model.get_ranking(triples, rhs_filters, batch_size=batch_size)
    return {
        "count": int(triples.shape[0]),
        "tail_mr": float(torch.mean(ranks).item()),
        "tail_mrr": float(torch.mean(1.0 / ranks).item()),
        "tail_hits@1": float(torch.mean((ranks <= 1).float()).item()),
        "tail_hits@3": float(torch.mean((ranks <= 3).float()).item()),
        "tail_hits@10": float(torch.mean((ranks <= 10).float()).item()),
    }


def compute_lhs_metrics(
    model,
    triples: torch.Tensor,
    lhs_filters: Dict[Tuple[int, int], List[int]],
    relation_offset: int,
    batch_size: int = 500,
) -> Dict[str, float]:
    """Compute head-prediction metrics (?, r, t) using lhs filters."""
    if triples.shape[0] == 0:
        return {
            "count": 0,
            "head_mr": float("inf"),
            "head_mrr": 0.0,
            "head_hits@1": 0.0,
            "head_hits@3": 0.0,
            "head_hits@10": 0.0,
        }
    head_queries = triples.clone()
    tmp = head_queries[:, 0].clone()
    head_queries[:, 0] = head_queries[:, 2]
    head_queries[:, 2] = tmp
    head_queries[:, 1] += relation_offset
    ranks = model.get_ranking(head_queries, lhs_filters, batch_size=batch_size)
    return {
        "count": int(triples.shape[0]),
        "head_mr": float(torch.mean(ranks).item()),
        "head_mrr": float(torch.mean(1.0 / ranks).item()),
        "head_hits@1": float(torch.mean((ranks <= 1).float()).item()),
        "head_hits@3": float(torch.mean((ranks <= 3).float()).item()),
        "head_hits@10": float(torch.mean((ranks <= 10).float()).item()),
    }


def initialize_model_entities(model, entity_init: np.ndarray) -> None:
    """Copy entity initialization into a model, including complex models."""
    tensor = torch.from_numpy(entity_init)
    if hasattr(model, "embeddings"):
        entity_weight = model.embeddings[0].weight
    else:
        entity_weight = model.entity.weight
    if tuple(entity_weight.shape) != tuple(tensor.shape):
        raise ValueError(
            f"Entity init shape {tuple(tensor.shape)} does not match model shape "
            f"{tuple(entity_weight.shape)}"
        )
    entity_weight.data.copy_(tensor.to(device=entity_weight.device, dtype=entity_weight.dtype))
