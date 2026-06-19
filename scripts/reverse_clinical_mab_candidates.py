#!/usr/bin/env python3
"""Generate disease -> mAb candidates from a trained clinical KG embedding model.

This script queries the reverse direction of:
    (?, imgt:hasClinicalIndication, disease)
by scoring study-product heads, then maps those study products to mAbs:
    StudyProduct --isStudyProductOf--> Product --isProductOf--> mAb

The output is a deduplicated mAb ranking with supporting study-product evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import models
from datasets.kg_dataset import KGDataset
from datasets.process import get_idx
from utils.imgt_clinical import CLINICAL_INDICATION_RELATION

Triple = Tuple[str, str, str]
SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reverse clinical query utility: rank candidate mAbs for a disease "
            "using a trained checkpoint."
        )
    )
    parser.add_argument(
        "--trial-dir",
        type=Path,
        required=True,
        help="Directory containing config.json and model.pt.",
    )
    parser.add_argument(
        "--disease",
        type=str,
        required=True,
        help="Disease entity (for example imgt:Chronic_lymphocytic_leukemia_(CLL)).",
    )
    parser.add_argument(
        "--top-k-mabs",
        type=int,
        default=15,
        help="Number of mAb candidates to keep in output.",
    )
    parser.add_argument(
        "--top-k-study-products",
        type=int,
        default=100,
        help="Number of top study-product heads to keep in output.",
    )
    parser.add_argument(
        "--known-scope",
        type=str,
        default="all",
        choices=["train", "all"],
        help=(
            "Scope used to tag whether a predicted mAb is already known for this disease. "
            "'train' uses train split only, 'all' uses train+valid+test."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for inference.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output json path. Defaults to <trial-dir>/reverse_<disease>.json.",
    )
    return parser.parse_args()


def resolve_device(choice: str) -> str:
    if choice == "cpu":
        return "cpu"
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_raw_split(dataset_dir: Path, split: str) -> List[Triple]:
    path = dataset_dir / split
    triples: List[Triple] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"Expected 3 tab-separated values in {path}:{line_number}")
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def load_raw_triples(dataset_dir: Path) -> Dict[str, List[Triple]]:
    return {split: load_raw_split(dataset_dir, split) for split in SPLITS}


def build_studyproduct_to_mab(triples: Iterable[Triple]) -> Dict[str, str]:
    studyproduct_to_product: Dict[str, str] = {}
    product_to_mab: Dict[str, str] = {}
    for head, relation, tail in triples:
        if relation == "imgt:isStudyProductOf":
            studyproduct_to_product[head] = tail
        elif relation == "imgt:isProductOf":
            product_to_mab[head] = tail

    studyproduct_to_mab: Dict[str, str] = {}
    for study_product, product in studyproduct_to_product.items():
        mab = product_to_mab.get(product)
        if mab is not None:
            studyproduct_to_mab[study_product] = mab
    return studyproduct_to_mab


def collect_known_mabs_for_disease(
    split_to_triples: Dict[str, Sequence[Triple]],
    studyproduct_to_mab: Dict[str, str],
    disease: str,
    scope: str,
) -> List[str]:
    if scope == "train":
        candidate_splits = ("train",)
    else:
        candidate_splits = SPLITS

    known = set()
    for split in candidate_splits:
        for head, relation, tail in split_to_triples[split]:
            if relation != CLINICAL_INDICATION_RELATION or tail != disease:
                continue
            mab = studyproduct_to_mab.get(head)
            if mab is not None:
                known.add(mab)
    return sorted(known)


def slugify_entity_name(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return slug[:120] if slug else "query"


def score_reverse_query(
    model,
    disease_id: int,
    relation_id: int,
    relation_offset: int,
) -> np.ndarray:
    query = torch.tensor(
        [[disease_id, relation_id + relation_offset, 0]],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    with torch.no_grad():
        query_embeddings, query_biases = model.get_queries(query)
        candidates, _candidate_bias = model.get_rhs(query, eval_mode=True)
        scores = model.score(
            (query_embeddings, query_biases),
            (candidates, model.bt.weight),
            eval_mode=True,
        )
    return scores.squeeze(0).detach().cpu().numpy()


def main() -> None:
    args = parse_args()
    trial_dir = args.trial_dir.resolve()
    config_path = trial_dir / "config.json"
    model_path = trial_dir / "model.pt"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json in {trial_dir}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model.pt in {trial_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_name = config.get("dataset", "IMGT")
    os.environ.setdefault("DATA_PATH", str(Path.cwd() / "data"))
    dataset_dir = Path(os.environ["DATA_PATH"]) / dataset_name
    split_to_triples = load_raw_triples(dataset_dir)
    all_triples: List[Triple] = [
        triple for split in SPLITS for triple in split_to_triples[split]
    ]

    dataset = KGDataset(str(dataset_dir), debug=False)
    config_with_sizes = dict(config)
    config_with_sizes["sizes"] = dataset.get_shape()
    model_args = SimpleNamespace(**config_with_sizes)
    model = getattr(models, model_args.model)(model_args)

    device = resolve_device(args.device)
    model.to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    entity_to_id, relation_to_id = get_idx(str(dataset_dir))
    id_to_entity = {index: name for name, index in entity_to_id.items()}
    if args.disease not in entity_to_id:
        lower_query = args.disease.lower()
        suggestions = [
            entity
            for entity in entity_to_id
            if lower_query in entity.lower() or entity.lower() in lower_query
        ]
        suggestion_text = ""
        if suggestions:
            suggestion_text = "\nClosest matches:\n  " + "\n  ".join(sorted(suggestions)[:15])
        raise ValueError(
            f"Disease entity not found in dataset: {args.disease}. "
            f"Provide the exact entity string.{suggestion_text}"
        )
    if CLINICAL_INDICATION_RELATION not in relation_to_id:
        raise ValueError(
            f"Missing relation in dataset: {CLINICAL_INDICATION_RELATION}"
        )

    studyproduct_to_mab = build_studyproduct_to_mab(all_triples)
    known_mabs = collect_known_mabs_for_disease(
        split_to_triples=split_to_triples,
        studyproduct_to_mab=studyproduct_to_mab,
        disease=args.disease,
        scope=args.known_scope,
    )
    known_mabs_set = set(known_mabs)

    clinical_heads = sorted(
        {
            head
            for head, relation, _tail in all_triples
            if relation == CLINICAL_INDICATION_RELATION and head in studyproduct_to_mab
        }
    )
    clinical_head_ids = np.array(
        [entity_to_id[head] for head in clinical_heads if head in entity_to_id],
        dtype=np.int64,
    )
    if clinical_head_ids.size == 0:
        raise RuntimeError("No clinical study-product heads found in dataset.")

    relation_offset = dataset.get_shape()[1] // 2
    scores = score_reverse_query(
        model=model,
        disease_id=entity_to_id[args.disease],
        relation_id=relation_to_id[CLINICAL_INDICATION_RELATION],
        relation_offset=relation_offset,
    )

    sorted_head_positions = np.argsort(-scores[clinical_head_ids])
    ranked_study_products: List[Dict[str, object]] = []
    for position, head_pos in enumerate(sorted_head_positions, start=1):
        entity_id = int(clinical_head_ids[head_pos])
        study_product = id_to_entity[entity_id]
        mab = studyproduct_to_mab.get(study_product)
        ranked_study_products.append(
            {
                "rank": position,
                "study_product": study_product,
                "study_product_id": entity_id,
                "score": float(scores[entity_id]),
                "mapped_mab": mab,
                "mapped_mab_id": entity_to_id.get(mab) if mab else None,
                "known_for_disease": bool(mab in known_mabs_set) if mab else False,
            }
        )

    best_by_mab: Dict[str, Dict[str, object]] = {}
    evidence_counts: Dict[str, int] = defaultdict(int)
    for row in ranked_study_products:
        mab = row["mapped_mab"]
        if mab is None:
            continue
        evidence_counts[mab] += 1
        if mab in best_by_mab:
            continue
        best_by_mab[mab] = {
            "rank": len(best_by_mab) + 1,
            "mab": mab,
            "mab_id": row["mapped_mab_id"],
            "best_score": row["score"],
            "best_supporting_study_product": row["study_product"],
            "best_supporting_study_product_rank": row["rank"],
            "num_supporting_study_products": evidence_counts[mab],
            "known_for_disease": bool(mab in known_mabs_set),
        }

    # Update support counts after dedup.
    for mab, candidate in best_by_mab.items():
        candidate["num_supporting_study_products"] = evidence_counts[mab]

    ranked_mabs = list(best_by_mab.values())
    top_mabs = ranked_mabs[: max(0, args.top_k_mabs)]
    top_study_products = ranked_study_products[: max(0, args.top_k_study_products)]

    known_hits = sum(1 for row in top_mabs if row["known_for_disease"])
    payload = {
        "trial_dir": str(trial_dir),
        "dataset": dataset_name,
        "model": model_args.model,
        "query": {
            "disease": args.disease,
            "relation": CLINICAL_INDICATION_RELATION,
            "reverse_query_form": "?, imgt:hasClinicalIndication, disease",
            "known_scope": args.known_scope,
        },
        "candidate_pool": {
            "num_entities_total": int(dataset.get_shape()[0]),
            "num_clinical_study_products": int(clinical_head_ids.size),
            "num_unique_mabs_mapped_from_clinical_heads": int(len(best_by_mab)),
        },
        "known_mabs_for_disease": {
            "count": len(known_mabs),
            "entities": known_mabs,
        },
        "top_mab_summary": {
            "top_k": args.top_k_mabs,
            "num_known_in_top_k": int(known_hits),
            "precision_at_k": float(known_hits / args.top_k_mabs) if args.top_k_mabs > 0 else 0.0,
            "recall_at_k": float(known_hits / len(known_mabs)) if known_mabs else 0.0,
        },
        "top_mabs": top_mabs,
        "top_study_products": top_study_products,
    }

    if args.output is not None:
        output_path = args.output.resolve()
    else:
        output_path = trial_dir / f"reverse_{slugify_entity_name(args.disease)}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote reverse candidate report: {output_path}")
    print(
        "Top-mAb summary: "
        f"known_in_top_{args.top_k_mabs}={known_hits}, "
        f"precision@{args.top_k_mabs}={payload['top_mab_summary']['precision_at_k']:.3f}, "
        f"recall@{args.top_k_mabs}={payload['top_mab_summary']['recall_at_k']:.3f}"
    )


if __name__ == "__main__":
    main()
