"""Evaluate saved checkpoints on IMGT clinical-indication relations."""

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

import models
from datasets.kg_dataset import KGDataset
from datasets.process import get_idx
from utils.imgt_clinical import CLINICAL_INDICATION_RELATION


def enable_cpu_fallback() -> str:
    """Run evaluation on CPU when CUDA is unavailable."""
    if torch.cuda.is_available():
        return "cuda"

    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self
    return "cpu"


def resolve_forward_clinical_relation_id(relation_to_id: Dict[str, int]) -> int:
    if CLINICAL_INDICATION_RELATION not in relation_to_id:
        raise ValueError(f"Missing relation: {CLINICAL_INDICATION_RELATION}")
    return relation_to_id[CLINICAL_INDICATION_RELATION]


def filter_triples_by_relation_ids(
    triples: torch.Tensor, relation_ids: Iterable[int]
) -> torch.Tensor:
    relation_ids = list(relation_ids)
    if not relation_ids:
        return triples[:0]
    relation_id_tensor = torch.as_tensor(relation_ids, dtype=triples.dtype, device=triples.device)
    return triples[torch.isin(triples[:, 1], relation_id_tensor)]


def load_model(model_dir: str, device: str):
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)

    dataset_name = config.get("dataset", "IMGT")
    dataset_path = os.path.join(os.environ["DATA_PATH"], dataset_name)
    dataset = KGDataset(dataset_path, False)
    model_args = dict(config)
    model_args.setdefault("sizes", dataset.get_shape())
    model_args.setdefault("dropout", 0.0)
    model_args.setdefault("gamma", 12.0)
    model_args.setdefault("dtype", "single")
    model_args.setdefault("bias", "constant")
    model_args.setdefault("init_size", 1e-3)
    model_args.setdefault("rank", 640)
    model_args.setdefault("multi_c", False)
    args = argparse.Namespace(**model_args)

    model = getattr(models, args.model)(args)
    model.to(device)
    state_dict = torch.load(os.path.join(model_dir, "model.pt"), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return args, dataset, model


def compute_tail_ranks(
    model,
    triples: torch.Tensor,
    rhs_filters: Dict[Tuple[int, int], Sequence[int]],
    batch_size: int,
) -> torch.Tensor:
    if triples.shape[0] == 0:
        return torch.empty(0, dtype=torch.float32)
    return model.get_ranking(triples, rhs_filters, batch_size=batch_size)


def compute_head_ranks(
    model,
    triples: torch.Tensor,
    lhs_filters: Dict[Tuple[int, int], Sequence[int]],
    relation_offset: int,
    batch_size: int,
) -> torch.Tensor:
    if triples.shape[0] == 0:
        return torch.empty(0, dtype=torch.float32)
    head_queries = triples.clone()
    tmp = head_queries[:, 0].clone()
    head_queries[:, 0] = head_queries[:, 2]
    head_queries[:, 2] = tmp
    head_queries[:, 1] += relation_offset
    return model.get_ranking(head_queries, lhs_filters, batch_size=batch_size)


def summarize_ranks(ranks: torch.Tensor) -> Dict[str, float]:
    if ranks.numel() == 0:
        return {
            "count": 0,
            "MR": float("inf"),
            "MRR": 0.0,
            "HITS@1": 0.0,
            "HITS@3": 0.0,
            "HITS@10": 0.0,
        }
    ranks = ranks.float()
    return {
        "count": int(ranks.shape[0]),
        "MR": float(torch.mean(ranks).item()),
        "MRR": float(torch.mean(1.0 / ranks).item()),
        "HITS@1": float(torch.mean((ranks <= 1).float()).item()),
        "HITS@3": float(torch.mean((ranks <= 3).float()).item()),
        "HITS@10": float(torch.mean((ranks <= 10).float()).item()),
    }


def evaluate_split_clinical(
    model,
    dataset: KGDataset,
    split: str,
    forward_relation_id: int,
    relation_offset: int,
    batch_size: int,
):
    examples = dataset.get_examples(split)
    filters = dataset.get_filters()
    rhs_filters = filters["rhs"]
    lhs_filters = filters["lhs"]

    forward_examples = filter_triples_by_relation_ids(examples, [forward_relation_id])
    tail_ranks = compute_tail_ranks(model, forward_examples, rhs_filters, batch_size)
    head_ranks = compute_head_ranks(
        model, forward_examples, lhs_filters, relation_offset, batch_size
    )
    combined_ranks = torch.cat([tail_ranks, head_ranks], dim=0)

    return {
        "forward_tail": summarize_ranks(tail_ranks),
        "forward_head": summarize_ranks(head_ranks),
        "combined": summarize_ranks(combined_ranks),
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    return (
        f"count: {metrics['count']}, "
        f"MR: {metrics['MR']:.6f}, "
        f"MRR: {metrics['MRR']:.6f}, "
        f"HITS@1: {metrics['HITS@1']:.6f}, "
        f"HITS@3: {metrics['HITS@3']:.6f}, "
        f"HITS@10: {metrics['HITS@10']:.6f}"
    )


def get_examples_for_head_csv(dataset: KGDataset, split_scope: str) -> torch.Tensor:
    if split_scope == "valid":
        return dataset.get_examples("valid")
    if split_scope == "test":
        return dataset.get_examples("test")
    if split_scope == "valid_test":
        return torch.cat([dataset.get_examples("valid"), dataset.get_examples("test")], dim=0)
    if split_scope == "all_clinical":
        # Use raw stored splits to avoid doubled train reciprocals.
        return torch.cat(
            [
                torch.from_numpy(dataset.data["train"]).long(),
                torch.from_numpy(dataset.data["valid"]).long(),
                torch.from_numpy(dataset.data["test"]).long(),
            ],
            dim=0,
        )
    raise ValueError(f"Unsupported --head-csv-split: {split_scope}")


def write_head_hits10_csv(
    output_path: str,
    model,
    dataset: KGDataset,
    forward_relation_id: int,
    relation_offset: int,
    idx_to_entity: Dict[int, str],
    batch_size: int,
    split_scope: str,
) -> None:
    lhs_filters = dataset.get_filters()["lhs"]
    all_raw_examples = torch.cat(
        [
            torch.from_numpy(dataset.data["train"]).long(),
            torch.from_numpy(dataset.data["valid"]).long(),
            torch.from_numpy(dataset.data["test"]).long(),
        ],
        dim=0,
    )
    all_forward = filter_triples_by_relation_ids(all_raw_examples, [forward_relation_id])
    fanout_by_tail: Dict[int, int] = defaultdict(int)
    for triple in all_forward:
        fanout_by_tail[int(triple[2].item())] += 1

    eval_examples = get_examples_for_head_csv(dataset, split_scope)
    eval_forward = filter_triples_by_relation_ids(eval_examples, [forward_relation_id])
    eval_ranks = compute_head_ranks(
        model,
        eval_forward,
        lhs_filters,
        relation_offset,
        batch_size,
    )

    hits10_sum_by_tail: Dict[int, float] = defaultdict(float)
    eval_count_by_tail: Dict[int, int] = defaultdict(int)
    for triple, rank in zip(eval_forward, eval_ranks):
        tail_id = int(triple[2].item())
        eval_count_by_tail[tail_id] += 1
        hits10_sum_by_tail[tail_id] += float(rank.item() <= 10)

    rows = []
    for tail_id in sorted(fanout_by_tail):
        eval_count = eval_count_by_tail.get(tail_id, 0)
        hits10 = (
            hits10_sum_by_tail[tail_id] / eval_count
            if eval_count > 0
            else float("nan")
        )
        rows.append(
            {
                "tail_name": idx_to_entity.get(tail_id, f"id:{tail_id}"),
                "fanout": fanout_by_tail[tail_id],
                "head_hits@10": hits10,
                "num_eval_triples": eval_count,
            }
        )

    rows.sort(
        key=lambda row: (
            row["head_hits@10"] != row["head_hits@10"],  # NaN last
            -row["head_hits@10"] if row["head_hits@10"] == row["head_hits@10"] else 0.0,
            -row["fanout"],
            row["tail_name"],
        )
    )

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["tail_name", "fanout", "head_hits@10", "num_eval_triples"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate clinical-indication ranking metrics")
    parser.add_argument("model_dirs", nargs="+", help="Checkpoint directories to evaluate")
    parser.add_argument("--batch-size", type=int, default=500, help="Evaluation batch size")
    parser.add_argument(
        "--head-csv-out",
        type=str,
        default=None,
        help=(
            "Optional CSV output path for forward-relation head prediction (?, r, t) "
            "grouped by tail entity. Only written when evaluating a single checkpoint."
        ),
    )
    parser.add_argument(
        "--head-csv-split",
        type=str,
        default="all_clinical",
        choices=["valid", "test", "valid_test", "all_clinical"],
        help=(
            "Triples used to compute per-tail head HITS@10 in the CSV. "
            "'all_clinical' spans raw train+valid+test clinical triples."
        ),
    )
    args = parser.parse_args()

    os.environ.setdefault("DATA_PATH", os.path.join(os.getcwd(), "data"))
    device = enable_cpu_fallback()

    for model_dir in args.model_dirs:
        model_args, dataset, model = load_model(model_dir, device)
        dataset_name = getattr(model_args, "dataset", "IMGT")
        dataset_path = os.path.join(os.environ["DATA_PATH"], dataset_name)
        entity_to_id, relation_to_id = get_idx(dataset_path)
        forward_relation_id = resolve_forward_clinical_relation_id(relation_to_id)
        relation_offset = dataset.get_shape()[1] // 2
        idx_to_entity = {idx: name for name, idx in entity_to_id.items()}

        print(
            "Clinical relation setup: "
            f"dataset={dataset_name}, "
            f"forward({CLINICAL_INDICATION_RELATION})={forward_relation_id}, "
            f"head_relation_offset={relation_offset}"
        )
        print(f"\n{model_args.model} [{model_dir}]")
        for split in ("valid", "test"):
            split_metrics = evaluate_split_clinical(
                model=model,
                dataset=dataset,
                split=split,
                forward_relation_id=forward_relation_id,
                relation_offset=relation_offset,
                batch_size=args.batch_size,
            )
            print(f"{split} | forward_tail : {format_metrics(split_metrics['forward_tail'])}")
            print(f"{split} | forward_head : {format_metrics(split_metrics['forward_head'])}")
            print(f"{split} | combined     : {format_metrics(split_metrics['combined'])}")

        if args.head_csv_out:
            if len(args.model_dirs) > 1:
                raise ValueError("--head-csv-out is supported only with exactly one model_dir")
            write_head_hits10_csv(
                output_path=args.head_csv_out,
                model=model,
                dataset=dataset,
                forward_relation_id=forward_relation_id,
                relation_offset=relation_offset,
                idx_to_entity=idx_to_entity,
                batch_size=args.batch_size,
                split_scope=args.head_csv_split,
            )
            print(
                f"Clinical head-prediction CSV written to {args.head_csv_out} "
                f"(split scope: {args.head_csv_split})"
            )


if __name__ == "__main__":
    main()
