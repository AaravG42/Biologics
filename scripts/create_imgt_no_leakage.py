#!/usr/bin/env python3
"""Create an IMGT split with reciprocal triples kept in the same partition."""

from __future__ import annotations

import argparse
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.process import process_dataset

Triple = Tuple[str, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine IMGT train/valid/test, then rebuild splits while forcing any "
            "reciprocal triples (h, r, t) and (t, r2, h) into the same split."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/IMGT"),
        help="Directory containing train, valid, and test raw split files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/IMGT_no_leakage"),
        help="Directory where the rebuilt dataset will be written.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=None,
        help="Train ratio. Defaults to the current source split ratio.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=None,
        help="Validation ratio. Defaults to the current source split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=None,
        help="Test ratio. Defaults to the current source split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when ordering reciprocal components.",
    )
    return parser.parse_args()


def load_split(path: Path) -> List[Triple]:
    triples: List[Triple] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = tuple(line.split("\t"))
            if len(parts) != 3:
                raise ValueError(f"Expected 3 tab-separated values in {path}:{line_number}")
            triples.append(parts)  # type: ignore[arg-type]
    return triples


def determine_target_ratios(
    split_sizes: Dict[str, int],
    train_ratio: float | None,
    valid_ratio: float | None,
    test_ratio: float | None,
) -> Dict[str, float]:
    if train_ratio is None and valid_ratio is None and test_ratio is None:
        total = sum(split_sizes.values())
        return {name: size / total for name, size in split_sizes.items()}

    if train_ratio is None or valid_ratio is None or test_ratio is None:
        raise ValueError("Either provide all ratios or leave all unset.")

    total = train_ratio + valid_ratio + test_ratio
    if total <= 0:
        raise ValueError("Ratios must sum to a positive value.")
    return {
        "train": train_ratio / total,
        "valid": valid_ratio / total,
        "test": test_ratio / total,
    }


def find_components(triples: Sequence[Triple]) -> List[List[int]]:
    parent = list(range(len(triples)))
    rank = [0] * len(triples)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        if rank[root_left] < rank[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        if rank[root_left] == rank[root_right]:
            rank[root_left] += 1

    reverse_index: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for index, (head, relation, tail) in enumerate(triples):
        reverse_index[(tail, head)].append(index)

    for index, (head, relation, tail) in enumerate(triples):
        for other_index in reverse_index[(head, tail)]:
            if other_index == index:
                continue
            other_head, other_relation, other_tail = triples[other_index]
            if other_head == tail and other_tail == head:
                union(index, other_index)

    components: Dict[int, List[int]] = defaultdict(list)
    for index in range(len(triples)):
        components[find(index)].append(index)
    return list(components.values())


def assign_components(
    triples: Sequence[Triple],
    components: Sequence[Sequence[int]],
    ratios: Dict[str, float],
    seed: int,
) -> Dict[str, List[Triple]]:
    total_triples = len(triples)
    targets = {name: ratios[name] * total_triples for name in ("train", "valid", "test")}
    assigned_counts = {name: 0 for name in ("train", "valid", "test")}
    assignments = {name: [] for name in ("train", "valid", "test")}

    component_infos = []
    for component in components:
        component_entities = set()
        component_relations = set()
        for index in component:
            head, relation, tail = triples[index]
            component_entities.add(head)
            component_entities.add(tail)
            component_relations.add(relation)
        component_infos.append(
            {
                "indices": list(component),
                "size": len(component),
                "entities": component_entities,
                "relations": component_relations,
            }
        )

    all_entities = set()
    all_relations = set()
    for info in component_infos:
        all_entities.update(info["entities"])
        all_relations.update(info["relations"])

    shuffled_component_infos = list(component_infos)
    random.Random(seed).shuffle(shuffled_component_infos)
    shuffled_component_infos.sort(key=lambda info: info["size"], reverse=True)

    remaining_component_ids = set(range(len(shuffled_component_infos)))
    uncovered_entities = set(all_entities)
    uncovered_relations = set(all_relations)

    # Ensure transductive train coverage: every entity/relation appears in train at least once.
    while uncovered_entities or uncovered_relations:
        best_id = None
        best_score = None
        for component_id in remaining_component_ids:
            info = shuffled_component_infos[component_id]
            new_entities = len(info["entities"] & uncovered_entities)
            new_relations = len(info["relations"] & uncovered_relations)
            coverage_gain = (new_entities * 1000) + (new_relations * 100)
            if coverage_gain == 0:
                continue
            train_remaining = targets["train"] - assigned_counts["train"]
            overshoot = max(0.0, info["size"] - train_remaining)
            score = (-coverage_gain, overshoot, info["size"])
            if best_score is None or score < best_score:
                best_score = score
                best_id = component_id

        if best_id is None:
            missing_entities = len(uncovered_entities)
            missing_relations = len(uncovered_relations)
            raise RuntimeError(
                "Unable to cover all train identities. "
                f"Missing entities: {missing_entities}, missing relations: {missing_relations}."
            )

        info = shuffled_component_infos[best_id]
        assignments["train"].extend(triples[index] for index in info["indices"])
        assigned_counts["train"] += info["size"]
        uncovered_entities -= info["entities"]
        uncovered_relations -= info["relations"]
        remaining_component_ids.remove(best_id)

    for component_id in sorted(remaining_component_ids, key=lambda item: shuffled_component_infos[item]["size"], reverse=True):
        info = shuffled_component_infos[component_id]
        size = info["size"]

        def score(split: str) -> Tuple[float, int]:
            remaining = targets[split] - assigned_counts[split]
            overshoot = max(0.0, size - remaining)
            unmet_after_assignment = max(0.0, remaining - size)
            return (overshoot, int(unmet_after_assignment))

        chosen_split = min(("train", "valid", "test"), key=score)
        assignments[chosen_split].extend(triples[index] for index in info["indices"])
        assigned_counts[chosen_split] += size

    return assignments


def count_unseen_train_identities(
    splits: Dict[str, Sequence[Triple]]
) -> Tuple[int, int]:
    train_entities = set()
    all_entities = set()
    train_relations = set()
    all_relations = set()

    for split_name, triples in splits.items():
        for head, relation, tail in triples:
            all_entities.add(head)
            all_entities.add(tail)
            all_relations.add(relation)
            if split_name == "train":
                train_entities.add(head)
                train_entities.add(tail)
                train_relations.add(relation)

    unseen_entities = all_entities - train_entities
    unseen_relations = all_relations - train_relations
    return len(unseen_entities), len(unseen_relations)


def write_split(path: Path, triples: Sequence[Triple]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for head, relation, tail in triples:
            handle.write(f"{head}\t{relation}\t{tail}\n")


def count_cross_split_reciprocals(splits: Dict[str, Sequence[Triple]]) -> Dict[Tuple[str, str], int]:
    pair_counts: Dict[Tuple[str, str], int] = {}
    split_sets = {name: set(values) for name, values in splits.items()}
    split_pairs = {
        name: {(head, tail) for head, relation, tail in values}
        for name, values in split_sets.items()
    }
    for source, target in (("train", "valid"), ("train", "test"), ("valid", "test")):
        count = 0
        for head, relation, tail in split_sets[target]:
            if (tail, head) in split_pairs[source]:
                count += 1
        pair_counts[(source, target)] = count
    return pair_counts


def save_pickles(output_dir: Path) -> None:
    examples, filters = process_dataset(str(output_dir))
    for split in ("train", "valid", "test"):
        with (output_dir / f"{split}.pickle").open("wb") as handle:
            pickle.dump(examples[split], handle)
    with (output_dir / "to_skip.pickle").open("wb") as handle:
        pickle.dump(filters, handle)


def main() -> None:
    args = parse_args()
    source_splits = {
        "train": load_split(args.source_dir / "train"),
        "valid": load_split(args.source_dir / "valid"),
        "test": load_split(args.source_dir / "test"),
    }
    split_sizes = {name: len(values) for name, values in source_splits.items()}
    combined = source_splits["train"] + source_splits["valid"] + source_splits["test"]

    ratios = determine_target_ratios(
        split_sizes=split_sizes,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
    )
    components = find_components(combined)
    rebuilt_splits = assign_components(
        triples=combined,
        components=components,
        ratios=ratios,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        write_split(args.output_dir / split, rebuilt_splits[split])

    save_pickles(args.output_dir)

    reciprocal_leakage = count_cross_split_reciprocals(rebuilt_splits)
    unseen_train_entities, unseen_train_relations = count_unseen_train_identities(rebuilt_splits)
    print("Created dataset:", args.output_dir)
    print("Target ratios:", {name: round(value, 6) for name, value in ratios.items()})
    for split in ("train", "valid", "test"):
        count = len(rebuilt_splits[split])
        print(f"{split}: {count} triples ({count / len(combined):.4%})")
    print(
        "Train coverage gaps:",
        {
            "unseen_entities": unseen_train_entities,
            "unseen_relations": unseen_train_relations,
        },
    )
    print("Reciprocal cross-split counts:")
    for (source, target), count in reciprocal_leakage.items():
        print(f"  {source} -> {target}: {count}")

    if unseen_train_entities > 0 or unseen_train_relations > 0:
        raise RuntimeError(
            "Transductive requirement not met: some entities/relations are not present in train."
        )


if __name__ == "__main__":
    main()
