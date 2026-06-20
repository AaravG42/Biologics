"""Shared utilities for the encoder-init ablation.

Loads antibody heavy/light sequences, resolves mAb-entity <-> INN mappings from
the IMGT KG triples, and applies the fair reduction pipeline that is identical
across every encoder (standardize -> PCA-640 -> L2-norm -> 0.01 scale).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple  # noqa: F401

import numpy as np

# --- Mirror utils/imgt_clinical.py mapping logic ---------------------------
MAB_PATTERN = re.compile(r"imgt:mAb_(\d+)$")
INN_PATTERN = re.compile(r"imgt:(\d+)$")
STRUCTURE_LINK_RELATION = "imgt:isLinkedToStructureAccessNumb"

SEQUENCES_CSV = Path("/home/aarav/Biologics/esm_init/data/mab_inn_sequences_split.csv")
DATASET_DIR = Path("/home/aarav/KGEmb/data/IMGT_no_leakage_directional_clinical")

PCA_DIM = 640
INIT_SCALE = 0.01
RANDOM_STATE = 42

Triple = Tuple[str, str, str]


def load_raw_triples(dataset_dir: Path = DATASET_DIR) -> List[Triple]:
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


def extract_mab_to_inn_mapping(triples: Sequence[Triple]) -> Dict[str, str]:
    """Recover mAb entity -> INN number mappings from KG triples."""
    mapping: Dict[str, str] = {}
    for head, relation, tail in triples:
        mab_match = MAB_PATTERN.fullmatch(head)
        inn_match = INN_PATTERN.fullmatch(tail)
        if mab_match and relation == STRUCTURE_LINK_RELATION and inn_match:
            mapping[head] = inn_match.group(1)
    return mapping


def load_sequence_rows(path: Path = SEQUENCES_CSV) -> Dict[str, Tuple[str, str]]:
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


def build_mab_records() -> List[Dict[str, str]]:
    """Return ordered list of mAb records with both chains and a resolvable INN.

    Each record: {name: 'imgt:mAb_<N>', inn: '<INN>', heavy: str, light: str}.
    Only mAbs with BOTH non-empty heavy & light AND a resolvable INN link are
    kept. Sorted by numeric mAb id for deterministic ordering.
    """
    triples = load_raw_triples()
    mab_to_inn = extract_mab_to_inn_mapping(triples)
    seqs_by_inn = load_sequence_rows()

    records: List[Dict[str, str]] = []
    skipped_no_seq_row: List[str] = []
    skipped_empty_chain: List[str] = []

    for mab_name, inn in mab_to_inn.items():
        chains = seqs_by_inn.get(inn)
        if chains is None:
            skipped_no_seq_row.append(mab_name)
            continue
        heavy, light = chains
        if not heavy or not light:
            skipped_empty_chain.append(mab_name)
            continue
        records.append({"name": mab_name, "inn": inn, "heavy": heavy, "light": light})

    records.sort(key=lambda r: int(MAB_PATTERN.fullmatch(r["name"]).group(1)))
    return records, {
        "num_mab_inn_links": len(mab_to_inn),
        "skipped_no_sequence_row": sorted(skipped_no_seq_row),
        "skipped_empty_chain": sorted(skipped_empty_chain),
    }


def fair_reduction(concat: np.ndarray) -> Tuple[np.ndarray, List[float], Dict[str, object]]:
    """Apply the identical reduction pipeline to a [N, D] concat matrix.

    standardize (zero-mean/unit-var) -> PCA(<=640, random_state=42)
    -> [zero-pad to 640 if PCA rank < 640] -> L2-normalize each row
    -> multiply by 0.01.

    PCA cannot emit more than min(n_samples, n_features) components. With
    n_samples=483 mAbs and a target of 640, the max attainable rank is bounded
    by the (centered) sample count. We therefore fit PCA with
    n_components=min(640, n_samples, n_features), then ZERO-PAD the trailing
    columns up to 640. The padded columns are exactly the directions PCA has no
    data to populate (the null space), so this introduces no fabricated signal;
    it only conforms the output to the required [N, 640] init shape. L2-norm and
    0.01 scaling are applied AFTER padding, identically for every encoder. This
    is a forced, documented deviation from a literal "PCA to 640" because
    483 < 640.

    Returns (vecs[N,640] float32, explained_variance_ratio[<=640], info).
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    concat = np.asarray(concat, dtype=np.float64)
    n, d = concat.shape
    if d < PCA_DIM:
        raise ValueError(
            f"Concat feature dim {d} < PCA target {PCA_DIM}; refusing to pad features."
        )

    scaled = StandardScaler().fit_transform(concat)
    max_components = min(PCA_DIM, n, d)
    pca = PCA(n_components=max_components, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(scaled)  # [N, max_components]

    padded_components = PCA_DIM - max_components
    if padded_components > 0:
        reduced = np.hstack(
            [reduced, np.zeros((n, padded_components), dtype=reduced.dtype)]
        )

    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    reduced = reduced / norms
    reduced = reduced * INIT_SCALE

    info = {
        "pca_components_fit": int(max_components),
        "pca_components_zero_padded": int(padded_components),
        "n_samples": int(n),
        "n_features": int(d),
    }
    return reduced.astype(np.float32), pca.explained_variance_ratio_.tolist(), info
