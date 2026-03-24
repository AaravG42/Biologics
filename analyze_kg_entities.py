#!/usr/bin/env python3
"""Analyze KG split files to summarize entity families, types, and relations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Set, Tuple


RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
SKOS_DEFINITION = "http://www.w3.org/2004/02/skos/core#definition"

URI_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("https://www.imgt.org/imgt-ontology#", "imgt:"),
    ("http://www.imgt.org/imgt-ontology#", "imgt:"),
    ("http://purl.obolibrary.org/obo/", "obo:"),
    ("https://purl.obolibrary.org/obo/", "obo:"),
    ("http://www.genenames.org/data/gene-symbol-report/#!/hgnc_id/", "HGNC:"),
    ("https://www.genenames.org/data/gene-symbol-report/#!/hgnc_id/", "HGNC:"),
    ("http://www.orpha.net/ORDO/", "ORDO:"),
    ("https://www.orpha.net/ORDO/", "ORDO:"),
    ("http://identifiers.org/drugbank/", "drugbank:"),
    ("https://identifiers.org/drugbank/", "drugbank:"),
    ("http://identifiers.org/uniprot/", "uniprot:"),
    ("https://identifiers.org/uniprot/", "uniprot:"),
)

FAMILY_EMBEDDING_GUIDANCE: Dict[str, Dict[str, str]] = {
    "mAb": {
        "strategy": "protein_sequence",
        "prior": "ESM or other antibody/protein LM on the amino-acid sequence",
        "alignment": "project sequence embeddings into KG space with a learned linear layer or small MLP",
    },
    "Construct": {
        "strategy": "composition",
        "prior": "compose from ordered Segment members and construct labels",
        "alignment": "encode segment order plus segment priors; use set/sequence pooling before KG projection",
    },
    "Segment": {
        "strategy": "structured_text",
        "prior": "embed IMGT segment labels, linker labels, payload labels, and chain labels as text",
        "alignment": "text encoder or ontology-label encoder projected into KG space",
    },
    "Product": {
        "strategy": "graph_composition",
        "prior": "compose from linked mAb, StudyProduct, company, construct, and conjugate metadata",
        "alignment": "aggregate neighboring typed priors with a relation-aware encoder",
    },
    "StudyProduct": {
        "strategy": "clinical_text",
        "prior": "compose from disease indication, phase, sponsor, designation, and mechanism nodes",
        "alignment": "clinical metadata encoder projected into KG space",
    },
    "Decision": {
        "strategy": "regulatory_metadata",
        "prior": "encode agency, status, designation, and linked study-product context",
        "alignment": "tabular/text metadata encoder into KG space",
    },
    "HGNC": {
        "strategy": "gene_text_or_sequence",
        "prior": "gene/protein embeddings from gene symbol, protein sequence, or biomedical text",
        "alignment": "map gene-side priors separately from antibody-side priors before fusion",
    },
    "MOA": {
        "strategy": "mechanism_text",
        "prior": "sentence embeddings over mechanism-of-action labels and linked references",
        "alignment": "text encoder with ontology-aware projection",
    },
    "Clone": {
        "strategy": "identifier_only",
        "prior": "usually weak without external metadata; fall back to learned KG embedding unless clone attributes exist",
        "alignment": "optionally tie clone and parent mAb embeddings with regularization",
    },
    "Disease": {
        "strategy": "biomedical_text_or_ontology",
        "prior": "disease ontology embeddings from MONDO/DOID/ORDO labels, definitions, and graph structure",
        "alignment": "separate ontology encoder aligned into KG space",
    },
    "Organisation": {
        "strategy": "name_text",
        "prior": "organization name embeddings or learned embeddings if names are sparse",
        "alignment": "small projection or plain learned KG vectors",
    },
    "Phase": {
        "strategy": "categorical",
        "prior": "small learned categorical embeddings",
        "alignment": "no heavy prior needed",
    },
    "NumericId": {
        "strategy": "categorical",
        "prior": "treat as identifier nodes unless linked metadata is available",
        "alignment": "plain learned embeddings are usually sufficient",
    },
}

GENERIC_CATEGORIES: Set[str] = {
    "mAb",
    "INN",
    "StudyProduct",
    "Product",
    "Decision",
    "Construct",
    "Segment",
    "Clone",
    "MOA",
    "Disease",
    "Organisation",
    "Phase",
    "NumericId",
    "Conjugate",
    "Fused",
    "ExpressionSystem",
    "ConstructLabels",
    "SegmentLabels",
    "ChainLabels",
    "ReceptorType",
    "ReceptorFormat",
}

DISEASE_HINTS: Tuple[str, ...] = (
    "cancer",
    "carcinoma",
    "lymphoma",
    "leukemia",
    "leukaemia",
    "myeloma",
    "melanoma",
    "tumor",
    "tumour",
    "sarcoma",
    "disease",
    "syndrome",
    "arthritis",
    "psoriasis",
    "sclerosis",
    "colitis",
    "hepatitis",
    "asthma",
    "fibrosis",
    "amyloidosis",
    "hemophilia",
    "enteropathy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze train/valid/test KG splits for entity and relation typing."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--query-csv", type=Path, default=Path("data/Query.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/kg_split_analysis"),
        help="Directory for JSON and text reports.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of top items to keep in compact summaries.",
    )
    return parser.parse_args()


def compact_term(value: str) -> str:
    value = value.strip()
    if not value:
        return value
    if not value.startswith("http://") and not value.startswith("https://"):
        return value
    for prefix, replacement in URI_PREFIXES:
        if value.startswith(prefix):
            return replacement + value[len(prefix) :]
    if "#" in value:
        return value.rsplit("#", 1)[1]
    return value.rstrip("/").rsplit("/", 1)[-1]


def local_name(value: str) -> str:
    compact = compact_term(value)
    if ":" in compact:
        return compact.split(":", 1)[1]
    return compact


def split_family(entity: str) -> str:
    token = local_name(entity)
    if token.isdigit():
        return "NumericId"
    if token.startswith("mAb_"):
        return "mAb"
    if token.startswith("StudyProduct_"):
        return "StudyProduct"
    if token.startswith("Product_"):
        return "Product"
    if token.startswith("Decision_"):
        return "Decision"
    if token.startswith("Construct_"):
        return "Construct"
    if token.startswith("Segment_"):
        return "Segment"
    if token.startswith("Clone_"):
        return "Clone"
    if token.startswith("MOA_"):
        return "MOA"
    if token.startswith("Phase_"):
        return "Phase"
    if token.startswith("HGNC:"):
        return "HGNC"
    if entity.startswith("HGNC:"):
        return "HGNC"
    if entity.startswith(("obo:MONDO_", "doid:", "ORDO:")):
        return "Disease"
    if token.startswith(("Cancers_", "Cancer_", "Carcinoma_", "Lymphoma_", "Leukemia_", "Melanoma_", "Myeloma_", "Solid_tumors")):
        return "Disease"
    if token in {"FDA", "EMA"} or token.endswith(("_Inc", "_Ltd", "_GmbH", "_Corp", "_Corporation", "_PLC", "_SA", "_SAS", "_LLC")):
        return "Organisation"
    if "_" in token:
        return token.split("_", 1)[0]
    return token


def load_split(path: Path) -> List[Tuple[str, str, str]]:
    triples: List[Tuple[str, str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"{path}:{line_number} does not have 3 tab-separated fields")
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def ordered_counter(counter: Counter[str], top_k: int) -> List[Dict[str, int]]:
    return [{"name": name, "count": count} for name, count in counter.most_common(top_k)]


def parse_query_metadata(
    query_csv: Path,
    focus_entities: Set[str],
    focus_relations: Set[str],
) -> Tuple[
    Dict[str, Set[str]],
    Dict[str, str],
    Dict[str, str],
    Dict[str, str],
    Dict[str, str],
]:
    entity_types: DefaultDict[str, Set[str]] = defaultdict(set)
    labels: Dict[str, str] = {}
    definitions: Dict[str, str] = {}
    relation_kind: Dict[str, str] = {}
    relation_labels: Dict[str, str] = {}

    with query_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"sub", "pred", "obj"}
        if set(reader.fieldnames or ()) != expected:
            raise ValueError(f"Expected columns {sorted(expected)} in {query_csv}, got {reader.fieldnames}")

        for row in reader:
            subject = compact_term(row["sub"])
            predicate = row["pred"]
            obj = row["obj"]

            if predicate == RDF_TYPE:
                type_term = compact_term(obj)
                if subject in focus_entities:
                    entity_types[subject].add(type_term)
                if subject in focus_relations:
                    relation_kind[subject] = local_name(type_term)
            elif predicate == RDFS_LABEL:
                label = compact_term(obj)
                if subject in focus_entities or subject in focus_relations or subject.startswith(("ncit:", "imgt:", "obo:", "ORDO:")):
                    labels[subject] = label
                    if subject in focus_relations:
                        relation_labels[subject] = label
            elif predicate == SKOS_DEFINITION:
                if subject in focus_entities or subject in focus_relations or subject.startswith(("ncit:", "imgt:", "obo:", "ORDO:")):
                    definitions[subject] = obj

    return entity_types, labels, definitions, relation_kind, relation_labels


def pick_primary_type(entity: str, types: Iterable[str], labels: Dict[str, str]) -> str:
    type_list = sorted(set(types))
    for type_term in type_list:
        label = labels.get(type_term)
        if label in GENERIC_CATEGORIES:
            return label
        local = local_name(type_term)
        if local in GENERIC_CATEGORIES:
            return local
    for type_term in type_list:
        label = labels.get(type_term)
        if label and label not in {"Class", "NamedIndividual"}:
            return label
    family = split_family(entity)
    if family == "Disease":
        return "Disease"
    return family


def summarize_relations(
    triples: Sequence[Tuple[str, str, str]],
    coarse_category: Dict[str, str],
    top_k: int,
) -> List[Dict[str, object]]:
    relation_counts: Counter[str] = Counter()
    relation_heads: DefaultDict[str, Counter[str]] = defaultdict(Counter)
    relation_tails: DefaultDict[str, Counter[str]] = defaultdict(Counter)

    for head, relation, tail in triples:
        relation_counts[relation] += 1
        relation_heads[relation][coarse_category[head]] += 1
        relation_tails[relation][coarse_category[tail]] += 1

    summary: List[Dict[str, object]] = []
    for relation, count in relation_counts.most_common():
        summary.append(
            {
                "relation": relation,
                "count": count,
                "head_categories": ordered_counter(relation_heads[relation], top_k),
                "tail_categories": ordered_counter(relation_tails[relation], top_k),
            }
        )
    return summary


def make_embedding_recommendation(primary_type: str, family: str) -> Dict[str, str]:
    if family in FAMILY_EMBEDDING_GUIDANCE:
        return FAMILY_EMBEDDING_GUIDANCE[family]
    if primary_type in FAMILY_EMBEDDING_GUIDANCE:
        return FAMILY_EMBEDDING_GUIDANCE[primary_type]
    return {
        "strategy": "text_or_learned",
        "prior": "use label/definition text if available; otherwise learn the embedding in the KG model",
        "alignment": "single projection into the KG latent space",
    }


def assign_coarse_category(
    entity: str,
    family: str,
    primary_type: str,
    explicit_types: Iterable[str],
) -> str:
    if family in GENERIC_CATEGORIES:
        return family
    if primary_type in GENERIC_CATEGORIES:
        return primary_type
    type_list = set(explicit_types)
    if any(type_term.startswith(("obo:MONDO_", "doid:", "ORDO:")) for type_term in type_list):
        return "Disease"
    token = local_name(entity).lower().replace("_", " ")
    if any(hint in token for hint in DISEASE_HINTS):
        return "Disease"
    if family == "":
        return primary_type
    return family


def analyze() -> None:
    args = parse_args()
    split_names = ("train", "valid", "test")
    split_paths = {name: args.data_dir / f"{name}.txt" for name in split_names}
    missing = [str(path) for path in split_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required split files: {missing}")

    split_triples = {name: load_split(path) for name, path in split_paths.items()}

    all_triples: List[Tuple[str, str, str]] = []
    entities_by_split: Dict[str, Set[str]] = {}
    relations_by_split: Dict[str, Set[str]] = {}
    relation_counts_by_split: Dict[str, Counter[str]] = {}
    entity_mentions_by_split: Dict[str, Counter[str]] = {}

    for split_name, triples in split_triples.items():
        all_triples.extend(triples)
        entities: Set[str] = set()
        relations: Set[str] = set()
        relation_counts: Counter[str] = Counter()
        entity_mentions: Counter[str] = Counter()
        for head, relation, tail in triples:
            entities.add(head)
            entities.add(tail)
            relations.add(relation)
            relation_counts[relation] += 1
            entity_mentions[head] += 1
            entity_mentions[tail] += 1
        entities_by_split[split_name] = entities
        relations_by_split[split_name] = relations
        relation_counts_by_split[split_name] = relation_counts
        entity_mentions_by_split[split_name] = entity_mentions

    all_entities = set().union(*entities_by_split.values())
    all_relations = set().union(*relations_by_split.values())

    entity_types, labels, definitions, relation_kind, relation_labels = parse_query_metadata(
        query_csv=args.query_csv,
        focus_entities=all_entities,
        focus_relations=all_relations,
    )

    primary_type: Dict[str, str] = {}
    family_by_entity: Dict[str, str] = {}
    coarse_category: Dict[str, str] = {}
    for entity in all_entities:
        family = split_family(entity)
        family_by_entity[entity] = family
        primary_type[entity] = pick_primary_type(entity, entity_types.get(entity, ()), labels)
        coarse_category[entity] = assign_coarse_category(
            entity,
            family,
            primary_type[entity],
            entity_types.get(entity, ()),
        )

    family_counter: Counter[str] = Counter()
    type_counter: Counter[str] = Counter()
    typed_entities = 0
    family_examples: DefaultDict[str, List[str]] = defaultdict(list)
    type_examples: DefaultDict[str, List[str]] = defaultdict(list)
    family_to_types: DefaultDict[str, Counter[str]] = defaultdict(Counter)

    for entity in sorted(all_entities):
        family = family_by_entity[entity]
        p_type = primary_type[entity]
        family_counter[family] += 1
        type_counter[p_type] += 1
        if entity_types.get(entity):
            typed_entities += 1
        if len(family_examples[family]) < 5:
            family_examples[family].append(entity)
        if len(type_examples[p_type]) < 5:
            type_examples[p_type].append(entity)
        family_to_types[family][p_type] += 1

    relation_summary = summarize_relations(all_triples, coarse_category, args.top_k)

    entity_family_summary: List[Dict[str, object]] = []
    for family, count in family_counter.most_common():
        dominant_type = family_to_types[family].most_common(1)[0][0]
        entity_family_summary.append(
            {
                "family": family,
                "count": count,
                "dominant_primary_type": dominant_type,
                "primary_types": ordered_counter(family_to_types[family], args.top_k),
                "examples": family_examples[family],
                "embedding_recommendation": make_embedding_recommendation(dominant_type, family),
            }
        )

    mab_entities = sorted(entity for entity in all_entities if family_by_entity[entity] == "mAb")
    mab_records: List[Dict[str, object]] = []
    for entity in mab_entities:
        mab_records.append(
            {
                "entity": entity,
                "primary_type": primary_type[entity],
                "types": sorted(labels.get(t, local_name(t)) for t in entity_types.get(entity, ())),
                "label": labels.get(entity, local_name(entity)),
            }
        )

    report = {
        "dataset": {
            "splits": {
                split_name: {
                    "triples": len(split_triples[split_name]),
                    "entities": len(entities_by_split[split_name]),
                    "relations": len(relations_by_split[split_name]),
                    "top_relations": ordered_counter(relation_counts_by_split[split_name], args.top_k),
                }
                for split_name in split_names
            },
            "global": {
                "triples": len(all_triples),
                "entities": len(all_entities),
                "relations": len(all_relations),
                "typed_entities": typed_entities,
                "typed_entity_fraction": round(typed_entities / len(all_entities), 4) if all_entities else 0.0,
            },
        },
        "entity_primary_types": {
            "top_primary_types": ordered_counter(type_counter, args.top_k * 2),
            "top_families": ordered_counter(family_counter, args.top_k * 2),
            "families": entity_family_summary,
        },
        "relations": {
            "count": len(all_relations),
            "items": relation_summary,
            "relation_kinds": {
                relation: {
                    "kind": relation_kind.get(relation, "Unknown"),
                    "label": relation_labels.get(relation, local_name(relation)),
                }
                for relation in sorted(all_relations)
            },
        },
        "mAb_entities": {
            "count": len(mab_records),
            "embedding_prior": FAMILY_EMBEDDING_GUIDANCE["mAb"],
            "items": mab_records,
        },
        "notes": {
            "type_source": str(args.query_csv),
            "fallback_typing": "Entity families are inferred from stable name patterns when explicit rdf:type is absent in Query.csv.",
            "alignment_hint": "Use family-specific encoders, then learn a shared projection into the KG latent space rather than forcing one raw embedding space for all entity kinds.",
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "kg_entity_relation_report.json"
    text_path = args.output_dir / "kg_entity_relation_report.txt"
    entity_catalog_path = args.output_dir / "entity_catalog.csv"
    relation_catalog_path = args.output_dir / "relation_catalog.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)

    with text_path.open("w", encoding="utf-8") as handle:
        handle.write("KG split analysis\n")
        handle.write(f"Triples: {len(all_triples)}\n")
        handle.write(f"Entities: {len(all_entities)}\n")
        handle.write(f"Relations: {len(all_relations)}\n")
        handle.write(f"Typed entities from Query.csv: {typed_entities}/{len(all_entities)}\n\n")

        handle.write("Top entity families\n")
        for item in entity_family_summary[: args.top_k]:
            handle.write(
                f"- {item['family']}: {item['count']} entities | dominant type: {item['dominant_primary_type']} | "
                f"prior: {item['embedding_recommendation']['strategy']}\n"
            )

        handle.write("\nTop relations\n")
        for item in relation_summary[: args.top_k]:
            head_types = ", ".join(f"{entry['name']} ({entry['count']})" for entry in item["head_categories"][:3])
            tail_types = ", ".join(f"{entry['name']} ({entry['count']})" for entry in item["tail_categories"][:3])
            handle.write(
                f"- {item['relation']}: {item['count']} | heads: {head_types} | tails: {tail_types}\n"
            )

        handle.write("\nmAb prior recommendation\n")
        mab_prior = FAMILY_EMBEDDING_GUIDANCE["mAb"]
        handle.write(f"- strategy: {mab_prior['strategy']}\n")
        handle.write(f"- prior: {mab_prior['prior']}\n")
        handle.write(f"- alignment: {mab_prior['alignment']}\n")

    with entity_catalog_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "entity",
                "label",
                "family",
                "primary_type",
                "coarse_category",
                "explicit_types",
                "embedding_strategy",
            ]
        )
        for entity in sorted(all_entities):
            recommendation = make_embedding_recommendation(primary_type[entity], family_by_entity[entity])
            writer.writerow(
                [
                    entity,
                    labels.get(entity, local_name(entity)),
                    family_by_entity[entity],
                    primary_type[entity],
                    coarse_category[entity],
                    "|".join(sorted(labels.get(t, local_name(t)) for t in entity_types.get(entity, ()))),
                    recommendation["strategy"],
                ]
            )

    with relation_catalog_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "relation",
                "label",
                "kind",
                "count",
                "head_categories",
                "tail_categories",
            ]
        )
        for item in relation_summary:
            writer.writerow(
                [
                    item["relation"],
                    relation_labels.get(item["relation"], local_name(item["relation"])),
                    relation_kind.get(item["relation"], "Unknown"),
                    item["count"],
                    "|".join(f"{entry['name']}:{entry['count']}" for entry in item["head_categories"]),
                    "|".join(f"{entry['name']}:{entry['count']}" for entry in item["tail_categories"]),
                ]
            )

    print(f"Wrote {json_path}")
    print(f"Wrote {text_path}")
    print(f"Wrote {entity_catalog_path}")
    print(f"Wrote {relation_catalog_path}")


if __name__ == "__main__":
    analyze()
