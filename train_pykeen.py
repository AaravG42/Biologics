#!/usr/bin/env python3
"""Train a transductive PyKEEN model on indexed triples."""

from __future__ import annotations

import argparse
import csv
import json
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

# Keep PyKEEN/PyStow caches inside the repo so the script works without
# requiring write access to the user's home directory.
os.environ.setdefault("PYSTOW_HOME", str(Path("artifacts/pystow").resolve()))

from pykeen.models import BoxE, TransE
from pykeen.training import SLCWATrainingLoop
from pykeen.triples import CoreTriplesFactory
from pykeen.sampling import BernoulliNegativeSampler

IndexedTriple = Tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a transductive PyKEEN model using an 80/20 split "
            "over the indexed knowledge-graph triples."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="boxe",
        choices=["transe", "boxe"],
        help="PyKEEN model architecture to train",
    )
    parser.add_argument(
        "--triples",
        type=Path,
        default=Path("artifacts/kg_vocab_onco_prior/triples_ids.csv"),
        help="CSV containing head_id,relation_id,tail_id",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("artifacts/kg_vocab_onco_prior/metadata.json"),
        help="Metadata JSON produced by build_kg_vocabulary.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for split files, trained model artifacts, and metrics. "
            "Defaults to artifacts/pykeen_<model>."
        ),
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Embedding dimension. Defaults to the vocabulary metadata value.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Training batch size",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=4,
        help="Evaluation batch size for tail ranking",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Optimizer learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split and training",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of facts used for training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device. 'auto' prefers CUDA when available.",
    )
    return parser.parse_args()


def load_metadata(path: Path) -> Dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_triples(path: Path) -> torch.LongTensor:
    triples: List[IndexedTriple] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = {"head_id", "relation_id", "tail_id"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError(
                f"Expected columns {sorted(expected)}, got {reader.fieldnames}"
            )
        for row in reader:
            triples.append(
                (int(row["head_id"]), int(row["relation_id"]), int(row["tail_id"]))
            )
    return torch.as_tensor(triples, dtype=torch.long)


def load_relation_id_map(path: Path) -> Dict[str, int]:
    relation_to_id: Dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = {"relation", "id"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError(
                f"Expected columns {sorted(expected)} in {path}, got {reader.fieldnames}"
            )
        for row in reader:
            relation_to_id[row["relation"]] = int(row["id"])
    return relation_to_id


def resolve_entity_embeddings_path(metadata: Dict[str, object], metadata_path: Path) -> Path:
    artifacts = metadata.get("artifacts")
    if isinstance(artifacts, dict):
        entity_embeddings_name = artifacts.get("entity_embeddings")
        if isinstance(entity_embeddings_name, str):
            candidate = metadata_path.parent / entity_embeddings_name
            if candidate.is_file():
                return candidate

    candidate = metadata_path.parent / "entity_embeddings.npy"
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        f"Could not find entity_embeddings.npy in {metadata_path.parent}"
    )


def load_entity_embeddings(path: Path, num_entities: int) -> np.ndarray:
    entity_embeddings = np.load(path)
    if entity_embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D entity embeddings in {path}, got shape {entity_embeddings.shape}"
        )
    if entity_embeddings.shape[0] != num_entities:
        raise ValueError(
            "Entity embedding row count does not match metadata: "
            f"{entity_embeddings.shape[0]} != {num_entities}"
        )
    return entity_embeddings.astype(np.float32, copy=False)


def split_triples(
    triples: torch.LongTensor, train_fraction: float, seed: int
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(triples.shape[0], generator=generator)
    cut = int(triples.shape[0] * train_fraction)
    train_idx = permutation[:cut]
    test_idx = permutation[cut:]
    return triples[train_idx], triples[test_idx]


def write_triples(path: Path, triples: torch.LongTensor) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["head_id", "relation_id", "tail_id"])
        writer.writerows(triples.tolist())


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_factory(
    triples: torch.LongTensor,
    num_entities: int,
    num_relations: int,
    create_inverse_triples: bool = False,
) -> CoreTriplesFactory:
    return CoreTriplesFactory(
        mapped_triples=triples,
        num_entities=num_entities,
        num_relations=num_relations,
        create_inverse_triples=create_inverse_triples,
    )


def compute_tail_ranks(
    model: Any,
    test_triples: torch.LongTensor,
    batch_size: int,
    device: str,
) -> torch.LongTensor:
    model.eval()
    ranks: List[torch.LongTensor] = []
    eval_device = torch.device(device)

    with torch.no_grad():
        for start in range(0, test_triples.shape[0], batch_size):
            batch = test_triples[start : start + batch_size].to(eval_device)
            hr_batch = batch[:, :2]
            true_tail_ids = batch[:, 2]

            # PyKEEN returns higher scores for more likely tails.
            scores = model.predict_t(hr_batch=hr_batch)
            true_tail_scores = scores.gather(1, true_tail_ids.unsqueeze(1))
            batch_ranks = torch.sum(scores > true_tail_scores, dim=1) + 1
            ranks.append(batch_ranks.cpu())

    return torch.cat(ranks, dim=0) if ranks else torch.empty(0, dtype=torch.long)


def summarize_tail_ranks(
    label: str,
    triples: torch.LongTensor,
    model: Any,
    batch_size: int,
    device: str,
) -> Dict[str, object]:
    if triples.shape[0] == 0:
        return {
            "label": label,
            "num_test_triples": 0,
            "task": "tail link prediction",
            "query_form": "(h, r, ?)",
            "ranking": "raw rank over all candidate tails using PyKEEN scores",
            "tail_mean_rank": None,
        }

    tail_ranks = compute_tail_ranks(
        model=model,
        test_triples=triples,
        batch_size=batch_size,
        device=device,
    )
    return {
        "label": label,
        "num_test_triples": int(triples.shape[0]),
        "task": "tail link prediction",
        "query_form": "(h, r, ?)",
        "ranking": "raw rank over all candidate tails using PyKEEN scores",
        "tail_mean_rank": float(tail_ranks.float().mean().item()),
    }


def resolve_clinical_indication_relation_ids(relation_to_id: Dict[str, int]) -> List[int]:
    # candidates = (
    #     "https://www.imgt.org/imgt-ontology#hasClinicalIndication",
    #     "https://www.imgt.org/imgt-ontology#isClinicalIndicationOf",
    #     "https://www.imgt.org/imgt-ontology#isClincicalIndicationOf",
    # )
    candidates = (
        "imgt:hasClinicalIndication",
        "imgt:isClinicalIndicationOf",
    )
    relation_ids = [
        relation_to_id[relation]
        for relation in candidates
        if relation in relation_to_id
    ]
    if len(relation_ids) < 2:
        raise ValueError(
            "Could not resolve both clinical-indication relation IDs from relations.csv"
        )
    return sorted(set(relation_ids))


def filter_triples_by_relation_ids(
    triples: torch.LongTensor, relation_ids: Sequence[int]
) -> torch.LongTensor:
    if triples.shape[0] == 0:
        return triples
    relation_id_tensor = torch.as_tensor(relation_ids, dtype=triples.dtype)
    mask = torch.isin(triples[:, 1], relation_id_tensor)
    return triples[mask]


def extract_embeddings(model: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    entity_embeddings = model.entity_representations[0]().detach().cpu()
    relation_embeddings = model.relation_representations[0]().detach().cpu()
    return entity_embeddings, relation_embeddings


def maybe_initialize_entity_representations(
    model: Any, entity_embeddings: np.ndarray, device: str
) -> bool:
    pretrained = torch.as_tensor(entity_embeddings, dtype=torch.float32, device=device)
    representation = model.entity_representations[0]

    candidate_parameters = []
    embedding_module = getattr(representation, "_embeddings", None)
    if embedding_module is not None and hasattr(embedding_module, "weight"):
        candidate_parameters.append(embedding_module.weight)
    if hasattr(representation, "weight"):
        candidate_parameters.append(representation.weight)

    for parameter in candidate_parameters:
        if tuple(parameter.shape) == tuple(pretrained.shape):
            with torch.no_grad():
                parameter.copy_(pretrained)
            return True

    return False


def build_model(
    model_name: str,
    training_factory: CoreTriplesFactory,
    embedding_dim: int,
    seed: int,
) -> Any:
    model_name = model_name.lower()
    if model_name == "transe":
        return TransE(
            triples_factory=training_factory,
            embedding_dim=embedding_dim,
            random_seed=seed,
        )
    if model_name == "boxe":
        return BoxE(
            triples_factory=training_factory,
            embedding_dim=embedding_dim,
            random_seed=seed,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata)
    triples = load_triples(args.triples)
    relation_to_id = load_relation_id_map(args.triples.parent / "relations.csv")
    clinical_indication_relation_ids = resolve_clinical_indication_relation_ids(
        relation_to_id
    )

    num_entities = int(metadata["num_entities"])
    num_relations = int(metadata["num_relations"])
    entity_embeddings_path = resolve_entity_embeddings_path(metadata, args.metadata)
    pretrained_entity_embeddings = load_entity_embeddings(
        entity_embeddings_path, num_entities
    )
    embedding_dim = (
        args.embedding_dim
        if args.embedding_dim is not None
        else int(pretrained_entity_embeddings.shape[1])
    )
    if pretrained_entity_embeddings.shape[1] != embedding_dim:
        raise ValueError(
            "Embedding dimension does not match entity_embeddings.npy: "
            f"{embedding_dim} != {pretrained_entity_embeddings.shape[1]}"
        )
    device = resolve_device(args.device)
    output_dir = args.output_dir or Path(f"artifacts/pykeen_{args.model}")

    train_triples, test_triples = split_triples(
        triples=triples,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_triples(output_dir / "train_triples.csv", train_triples)
    write_triples(output_dir / "test_triples.csv", test_triples)

    training_factory = build_factory(
        triples=train_triples,
        num_entities=num_entities,
        num_relations=num_relations,
    )

    print(f"PyKEEN version: {version('pykeen')}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {device}")
    print(f"Model: {args.model}")
    print(f"Entity embeddings source: {entity_embeddings_path}")
    print(f"Train triples: {train_triples.shape[0]}")
    print(f"Test triples: {test_triples.shape[0]}")
    print(
        "Clinical-indication relation IDs: "
        + ", ".join(str(relation_id) for relation_id in clinical_indication_relation_ids)
    )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = build_model(
        model_name=args.model,
        training_factory=training_factory,
        embedding_dim=embedding_dim,
        seed=args.seed,
    )
    model = model.to(torch.device(device))
    pretrained_loaded = maybe_initialize_entity_representations(
        model=model,
        entity_embeddings=pretrained_entity_embeddings,
        device=device,
    )
    if not pretrained_loaded:
        print(
            "Warning: could not preload entity_embeddings.npy into the PyKEEN "
            "entity representation; proceeding with the model's default initialization."
        )

    training_loop = SLCWATrainingLoop(
        model=model,
        triples_factory=training_factory,
        optimizer="Adam",
        optimizer_kwargs={"lr": args.learning_rate},
        negative_sampler=BernoulliNegativeSampler(triples_factory=training_factory),
        negative_sampler_kwargs={"num_negs_per_pos": 2},
    )
    losses = training_loop.train(
        triples_factory=training_factory,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        use_tqdm=True,
        use_tqdm_batch=True,
    )

    print("Evaluating held-out triples with tail prediction (h, r, ?)")
    full_evaluation = summarize_tail_ranks(
        label="all_test_triples",
        triples=test_triples,
        model=model,
        batch_size=args.eval_batch_size,
        device=device,
    )
    clinical_indication_test_triples = filter_triples_by_relation_ids(
        test_triples,
        clinical_indication_relation_ids,
    )
    clinical_indication_evaluation = summarize_tail_ranks(
        label="clinical_indication_relations_only",
        triples=clinical_indication_test_triples,
        model=model,
        batch_size=args.eval_batch_size,
        device=device,
    )

    entity_embeddings, relation_embeddings = extract_embeddings(model)
    torch.save(model.state_dict(), output_dir / "model_state_dict.pt")
    torch.save(entity_embeddings, output_dir / "entity_embeddings.pt")
    torch.save(relation_embeddings, output_dir / "relation_embeddings.pt")

    summary = {
        "library": "pykeen",
        "pykeen_version": version("pykeen"),
        "torch_version": torch.__version__,
        "model": model.__class__.__name__,
        "transductive": True,
        "train_fraction": args.train_fraction,
        "num_entities": num_entities,
        "num_relations": num_relations,
        "num_train_triples": int(train_triples.shape[0]),
        "num_test_triples": int(test_triples.shape[0]),
        "embedding_dim": embedding_dim,
        "entity_embeddings_source": str(entity_embeddings_path),
        "pretrained_entity_embeddings_loaded": pretrained_loaded,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "training_losses": [float(loss) for loss in losses],
        "evaluation": {
            "all_test_triples": full_evaluation,
            "clinical_indication_relations_only": clinical_indication_evaluation,
        },
    }
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"Tail mean rank (all test triples): {full_evaluation['tail_mean_rank']}")
    print(
        "Tail mean rank (clinical indication relations only): "
        f"{clinical_indication_evaluation['tail_mean_rank']}"
    )
    print(f"Artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
