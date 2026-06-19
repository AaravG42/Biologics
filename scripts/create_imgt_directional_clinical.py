#!/usr/bin/env python3
"""Create an IMGT dataset variant with directional clinical-indication edges only.

This script removes inverse clinical-indication triples (for example,
`imgt:isClinicalIndicationOf`) while keeping forward triples
(for example, `imgt:hasClinicalIndication`).
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.process import process_dataset

Triple = Tuple[str, str, str]
SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy an IMGT dataset while dropping inverse clinical-indication "
            "relations so only directional mAb -> hasClinicalIndication edges remain."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/IMGT_no_leakage"),
        help="Directory containing raw train/valid/test files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/IMGT_no_leakage_directional_clinical"),
        help="Directory where the filtered dataset will be written.",
    )
    parser.add_argument(
        "--drop-relation",
        action="append",
        default=["imgt:isClinicalIndicationOf"],
        help=(
            "Relation to remove. Can be provided multiple times. "
            "Default: imgt:isClinicalIndicationOf"
        ),
    )
    return parser.parse_args()


def load_split(path: Path) -> List[Triple]:
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


def write_split(path: Path, triples: Sequence[Triple]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for head, relation, tail in triples:
            handle.write(f"{head}\t{relation}\t{tail}\n")


def save_pickles(output_dir: Path) -> None:
    examples, filters = process_dataset(str(output_dir))
    for split in SPLITS:
        with (output_dir / f"{split}.pickle").open("wb") as handle:
            pickle.dump(examples[split], handle)
    with (output_dir / "to_skip.pickle").open("wb") as handle:
        pickle.dump(filters, handle)


def count_relation_occurrences(splits: Dict[str, Sequence[Triple]], relation_names: set[str]) -> int:
    total = 0
    for triples in splits.values():
        total += sum(1 for _h, relation, _t in triples if relation in relation_names)
    return total


def main() -> None:
    args = parse_args()

    source_splits = {
        split: load_split(args.source_dir / split)
        for split in SPLITS
    }

    drop_relations = set(args.drop_relation)
    filtered_splits: Dict[str, List[Triple]] = {}
    removed_by_split: Dict[str, int] = {}

    for split in SPLITS:
        before = source_splits[split]
        after = [triple for triple in before if triple[1] not in drop_relations]
        filtered_splits[split] = after
        removed_by_split[split] = len(before) - len(after)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_split(args.output_dir / split, filtered_splits[split])

    save_pickles(args.output_dir)

    removed_total = sum(removed_by_split.values())
    remaining_dropped_relation = count_relation_occurrences(filtered_splits, drop_relations)

    print("Created dataset:", args.output_dir)
    print("Source dataset:", args.source_dir)
    print("Dropped relations:", sorted(drop_relations))
    print("Removed triples:")
    for split in SPLITS:
        print(f"  {split}: {removed_by_split[split]}")
    print(f"Total removed: {removed_total}")
    print("Remaining dropped-relation triples:", remaining_dropped_relation)


if __name__ == "__main__":
    main()
