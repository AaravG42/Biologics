#!/usr/bin/env python3
"""Train a transductive R-GCN encoder with a TransE or BoxE decoder."""

from __future__ import annotations

import argparse
import csv
import json
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

# Keep PyKEEN/PyStow caches inside the repo so the script works without
# requiring write access to the user's home directory.
os.environ.setdefault("PYSTOW_HOME", str(Path("artifacts/pystow").resolve()))

from pykeen.losses import NSSALoss, SoftplusLoss
from pykeen.models import ERModel
from pykeen.nn.message_passing import RGCNRepresentation
from pykeen.training import SLCWATrainingLoop
from pykeen.triples import CoreTriplesFactory


IndexedTriple = Tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a transductive PyKEEN R-GCN model over indexed knowledge-graph "
            "triples using a TransE or BoxE decoder."
        )
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default="boxe",
        choices=["transe", "boxe"],
        help="Decoder architecture applied on top of the R-GCN entity encoder",
    )
    parser.add_argument(
        "--triples",
        type=Path,
        default=Path("artifacts/kg_vocab_literal_excluded/triples_ids.csv"),
        help="CSV containing head_id,relation_id,tail_id",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("artifacts/kg_vocab_literal_excluded/metadata.json"),
        help="Metadata JSON produced by build_kg_vocabulary.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for split files, trained model artifacts, and metrics. "
            "Defaults to artifacts/rgcn_<decoder>."
        ),
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Embedding dimension. Defaults to the vocabulary metadata value.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of R-GCN message-passing layers",
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
        default=64,
        help="Evaluation batch size for tail ranking",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Optimizer learning rate",
    )
    parser.add_argument(
        "--edge-dropout",
        type=float,
        default=0.4,
        help="Dropout applied to non-self-loop graph edges during message passing",
    )
    parser.add_argument(
        "--self-loop-dropout",
        type=float,
        default=0.2,
        help="Dropout applied to self-loops in message passing",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "leaky_relu", "none"],
        help="Activation used between R-GCN layers",
    )
    parser.add_argument(
        "--decomposition",
        type=str,
        default="bases",
        choices=["bases", "block", "none"],
        help="R-GCN relation-weight decomposition",
    )
    parser.add_argument(
        "--num-bases",
        type=int,
        default=None,
        help="Optional number of bases when using bases decomposition",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=None,
        help="Optional number of blocks when using block decomposition",
    )
    parser.add_argument(
        "--p",
        type=int,
        default=2,
        choices=[1, 2],
        help="Norm used by the TransE or BoxE decoder",
    )
    parser.add_argument(
        "--boxe-tanh-map",
        action="store_true",
        default=False,
        help="Apply BoxE's tanh map. Disabled by default unless this flag is set.",
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


def resolve_activation(name: str) -> str | None:
    if name == "none":
        return None
    if name == "relu":
        return "relu"
    if name == "leaky_relu":
        return "leakyrelu"
    raise ValueError(f"Unsupported activation: {name}")


def resolve_decomposition(args: argparse.Namespace) -> Tuple[str | None, Dict[str, int]]:
    if args.decomposition == "none":
        return None, {}

    kwargs: Dict[str, int] = {}
    if args.decomposition == "bases" and args.num_bases is not None:
        kwargs["num_bases"] = args.num_bases
    if args.decomposition == "block" and args.num_blocks is not None:
        kwargs["num_blocks"] = args.num_blocks
    return args.decomposition, kwargs


def build_factory(
    triples: torch.LongTensor,
    num_entities: int,
    num_relations: int,
) -> CoreTriplesFactory:
    return CoreTriplesFactory(
        mapped_triples=triples,
        num_entities=num_entities,
        num_relations=num_relations,
        create_inverse_triples=False,
    )


def rgcn_representation_kwargs(
    training_factory: CoreTriplesFactory,
    embedding_dim: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    decomposition, decomposition_kwargs = resolve_decomposition(args)
    return {
        "triples_factory": training_factory,
        "entity_representations_kwargs": {
            "shape": embedding_dim,
        },
        "num_layers": args.num_layers,
        "activation": resolve_activation(args.activation),
        "edge_dropout": args.edge_dropout,
        "self_loop_dropout": args.self_loop_dropout,
        "decomposition": decomposition,
        "decomposition_kwargs": decomposition_kwargs or None,
        "cache": True,
    }


def build_model(
    decoder_name: str,
    training_factory: CoreTriplesFactory,
    embedding_dim: int,
    seed: int,
    args: argparse.Namespace,
) -> ERModel:
    decoder_name = decoder_name.lower()
    common_kwargs = dict(
        triples_factory=training_factory,
        random_seed=seed,
    )

    if decoder_name == "transe":
        return ERModel(
            interaction="TransE",
            interaction_kwargs={"p": args.p},
            entity_representations=RGCNRepresentation,
            entity_representations_kwargs=rgcn_representation_kwargs(
                training_factory=training_factory,
                embedding_dim=embedding_dim,
                args=args,
            ),
            relation_representations_kwargs={"shape": embedding_dim},
            loss=SoftplusLoss,
            **common_kwargs,
        )

    if decoder_name == "boxe":
        rgcn_kwargs = rgcn_representation_kwargs(
            training_factory=training_factory,
            embedding_dim=embedding_dim,
            args=args,
        )
        return ERModel(
            interaction="BoxE",
            interaction_kwargs={
                "p": args.p,
                "tanh_map": args.boxe_tanh_map,
            },
            entity_representations=[RGCNRepresentation, RGCNRepresentation],
            entity_representations_kwargs=[rgcn_kwargs, dict(rgcn_kwargs)],
            relation_representations_kwargs=[
                {"shape": embedding_dim},
                {"shape": embedding_dim},
                {"shape": (1,)},
                {"shape": embedding_dim},
                {"shape": embedding_dim},
                {"shape": (1,)},
            ],
            loss=NSSALoss,
            loss_kwargs={
                "margin": 3.0,
                "adversarial_temperature": 2.0,
                "reduction": "mean",
            },
            **common_kwargs,
        )

    raise ValueError(f"Unsupported decoder: {decoder_name}")


def compute_tail_ranks(
    model: ERModel,
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

            scores = model.score_t(hr_batch=hr_batch)
            true_tail_scores = scores.gather(1, true_tail_ids.unsqueeze(1))
            batch_ranks = torch.sum(scores > true_tail_scores, dim=1) + 1
            ranks.append(batch_ranks.cpu())

    return torch.cat(ranks, dim=0) if ranks else torch.empty(0, dtype=torch.long)


def representation_tensor_names(decoder_name: str) -> Tuple[Sequence[str], Sequence[str]]:
    if decoder_name == "transe":
        return ["entity_embeddings"], ["relation_embeddings"]
    if decoder_name == "boxe":
        return (
            ["entity_position_embeddings", "entity_bump_embeddings"],
            [
                "relation_head_base_embeddings",
                "relation_head_delta_embeddings",
                "relation_head_size_embeddings",
                "relation_tail_base_embeddings",
                "relation_tail_delta_embeddings",
                "relation_tail_size_embeddings",
            ],
        )
    raise ValueError(f"Unsupported decoder: {decoder_name}")


def save_representations(
    output_dir: Path,
    names: Sequence[str],
    representations: Sequence[Any],
) -> List[str]:
    artifact_paths: List[str] = []
    with torch.no_grad():
        for name, representation in zip(names, representations, strict=True):
            tensor = representation(indices=None).detach().cpu()
            artifact_name = f"{name}.pt"
            torch.save(tensor, output_dir / artifact_name)
            artifact_paths.append(artifact_name)
    return artifact_paths


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata)
    triples = load_triples(args.triples)

    num_entities = int(metadata["num_entities"])
    num_relations = int(metadata["num_relations"])
    embedding_dim = (
        args.embedding_dim
        if args.embedding_dim is not None
        else int(metadata["embedding_dim"])
    )
    device = resolve_device(args.device)
    output_dir = args.output_dir or Path(f"artifacts/rgcn_{args.decoder}")

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
    print("Recommended environment: conda run -n biologics python train_rgcn.py ...")
    print(f"Encoder: RGCN")
    print(f"Decoder: {args.decoder}")
    print(f"Train triples: {train_triples.shape[0]}")
    print(f"Test triples: {test_triples.shape[0]}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = build_model(
        decoder_name=args.decoder,
        training_factory=training_factory,
        embedding_dim=embedding_dim,
        seed=args.seed,
        args=args,
    )
    model = model.to(torch.device(device))

    training_loop = SLCWATrainingLoop(
        model=model,
        triples_factory=training_factory,
        optimizer="Adam",
        optimizer_kwargs={"lr": args.learning_rate},
    )
    losses = training_loop.train(
        triples_factory=training_factory,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        use_tqdm=True,
        use_tqdm_batch=True,
    )

    print("Evaluating held-out triples with tail prediction (h, r, ?)")
    tail_ranks = compute_tail_ranks(
        model=model,
        test_triples=test_triples,
        batch_size=args.eval_batch_size,
        device=device,
    )
    tail_mean_rank = float(tail_ranks.float().mean().item())

    entity_names, relation_names = representation_tensor_names(args.decoder)
    entity_artifacts = save_representations(
        output_dir=output_dir,
        names=entity_names,
        representations=model.entity_representations,
    )
    relation_artifacts = save_representations(
        output_dir=output_dir,
        names=relation_names,
        representations=model.relation_representations,
    )
    torch.save(model.state_dict(), output_dir / "model_state_dict.pt")

    summary = {
        "library": "pykeen",
        "pykeen_version": version("pykeen"),
        "torch_version": torch.__version__,
        "encoder": "RGCN",
        "decoder": args.decoder,
        "transductive": True,
        "train_fraction": args.train_fraction,
        "num_entities": num_entities,
        "num_relations": num_relations,
        "num_train_triples": int(train_triples.shape[0]),
        "num_test_triples": int(test_triples.shape[0]),
        "embedding_dim": embedding_dim,
        "num_layers": args.num_layers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "edge_dropout": args.edge_dropout,
        "self_loop_dropout": args.self_loop_dropout,
        "activation": args.activation,
        "decomposition": args.decomposition,
        "num_bases": args.num_bases,
        "num_blocks": args.num_blocks,
        "p": args.p,
        "boxe_tanh_map": args.boxe_tanh_map,
        "seed": args.seed,
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "training_losses": [float(loss) for loss in losses],
        "artifacts": {
            "model_state_dict": "model_state_dict.pt",
            "entity_representations": entity_artifacts,
            "relation_representations": relation_artifacts,
        },
        "evaluation": {
            "task": "tail link prediction",
            "query_form": "(h, r, ?)",
            "ranking": "raw rank over all candidate tails using PyKEEN scores",
            "tail_mean_rank": tail_mean_rank,
        },
    }
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"Tail mean rank: {tail_mean_rank}")
    print(f"Artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
