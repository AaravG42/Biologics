#!/usr/bin/env python3
"""
score_blinded.py
================

Systematic, BLINDED, reproducible biological-plausibility scorer for the RotE
forward clinical-indication queries.

Motivation
----------
A reviewer objected that the original paper assigned plausibility verdicts to
only 5 hand-selected cases, unblinded, with manual overrides.  This script
answers that objection: it pre-registers a fixed scoring rule (encoded below as
constants), then applies that rule IDENTICALLY and BLINDLY to ALL 297 forward
`(StudyProduct, imgt:hasClinicalIndication, ?)` queries.  At no point does the
tagging logic look at the ground-truth tail or the model's rank of the true
tail: only the model's TOP-20 predicted indications and the query mAb's target
gene(s) feed into the tag.

Pipeline (per query)
--------------------
  1. Resolve the query mAb's target gene(s) by traversing the KG:
         StudyProduct --isProductOf--> Product --isProductOf--> mAb
                       --(sio:SIO_000291 | isTargetOf)--> HGNC:id
     and HGNC -> approved symbol via MyGene.info (cached).
  2. Pull Open Targets associated diseases for each target gene (Ensembl id via
     Open Targets `search`, then `associatedDiseases` with `score`); cached.
  3. For each TOP-20 predicted indication, normalise the IMGT disease string to a
     comparable form (lowercase, strip qualifiers, expand abbreviations via a
     transparent synonym table), then classify:
        GENERIC   -> non-specific bucket node (excluded from scoring)
        PLAUSIBLE -> normalised disease matches an OT disease for the target gene
                     with association score >= THRESHOLD
        OFF-TARGET-> specific but no qualifying OT association
  4. model_fraction = plausible / (plausible + off-target)  over specific top-20.
  5. BASELINE: draw 20 indications from the empirical popularity prior, score by
     the SAME rule, 1000 Monte-Carlo draws, average -> prior_fraction.
     enrichment = model_fraction / prior_fraction.

All API responses are cached to disk so the run is repeatable and rate-limit
friendly; the script is fully resumable.

Pre-registered constants live in the PRE-REGISTRATION block below and were fixed
before any 297-query result was inspected.
"""

import os
import sys
import csv
import json
import time
import random
import argparse
import urllib.request
import urllib.error
from collections import defaultdict

# --------------------------------------------------------------------------- #
#  PRE-REGISTRATION  (fixed before looking at any 297-query result)           #
# --------------------------------------------------------------------------- #

# Primary plausibility threshold on the Open Targets overall association score.
PLAUSIBILITY_THRESHOLD = 0.01
# Sensitivity thresholds (reported, not used to define the primary verdict).
SENSITIVITY_THRESHOLDS = [0.01, 0.05, 0.1]

TOP_K = 20                 # rank window scored per query
N_PRIOR_DRAWS = 1000       # Monte-Carlo draws for the popularity baseline
PRIOR_DRAW_SIZE = 20       # indications drawn per Monte-Carlo replicate
RANDOM_SEED = 20240617     # fixed seed -> deterministic baseline
N_BOOTSTRAP = 5000         # bootstrap replicates for the enrichment CI

OT_DISEASES_PER_GENE = 50  # how many top OT diseases to retain per target gene
API_SLEEP = 0.34           # seconds between live API calls (rate-limit friendly)

# GENERIC buckets: non-specific umbrella nodes that are excluded from scoring.
# These are uninterpretable "catch-all" indications; a tag of PLAUSIBLE/OFF for
# them would be meaningless.  Matched on the normalised core term (see below).
GENERIC_TERMS = {
    "solid tumors", "solid tumours", "solid tumor", "solid tumour",
    "cancers", "cancer", "neoplasms", "neoplasm", "oncology",
    "tumors", "tumours", "tumor", "tumour", "malignancies", "malignancy",
    "hematologic malignancies", "hematologic blood cancer",
    "hematologic-blood cancer", "malignant neoplasm",
    "cancer (generic)",
}

# --------------------------------------------------------------------------- #
#  NORMALISATION MAP                                                           #
# --------------------------------------------------------------------------- #
#
# Transparent synonym / abbreviation table.  Keys are abbreviations or IMGT
# shorthand that appear (in parenthesis) inside the IMGT entity name; values are
# the canonical disease phrase used by Open Targets.  Expansion is applied to the
# normalised string so that e.g. "...(NSCLC)" becomes
# "non-small cell lung carcinoma".  This table is fixed in advance and recorded
# in METHODS.md.
SYNONYM_TABLE = {
    "nsclc": "non-small cell lung carcinoma",
    "sclc": "small cell lung carcinoma",
    "hnscc": "head and neck squamous cell carcinoma",
    "escc": "esophageal squamous cell carcinoma",
    "crc": "colorectal carcinoma",
    "hcc": "hepatocellular carcinoma",
    "rcc": "renal cell carcinoma",
    "mcc": "merkel cell carcinoma",
    "gbm": "glioblastoma",
    "nhl": "non-hodgkin lymphoma",
    "hl": "hodgkin lymphoma",
    "cll": "chronic lymphocytic leukemia",
    "sll": "small lymphocytic lymphoma",
    "all": "acute lymphoblastic leukemia",
    "aml": "acute myeloid leukemia",
    "cml": "chronic myeloid leukemia",
    "mds": "myelodysplastic syndrome",
    "mm": "multiple myeloma",
    "dlbcl": "diffuse large b-cell lymphoma",
    "fl": "follicular lymphoma",
    "mcl": "mantle cell lymphoma",
    "ctcl": "cutaneous t-cell lymphoma",
    "ptcl": "peripheral t-cell lymphoma",
    "alcl": "anaplastic large cell lymphoma",
    "pcalcl": "primary cutaneous anaplastic large cell lymphoma",
    "hcl": "hairy cell leukemia",
    "tnbc": "triple-negative breast cancer",
    "uc": "urothelial carcinoma",
    "ucec": "endometrial cancer",
    "ec": "endometrial cancer",
    "ncc": "nasopharyngeal carcinoma",
    "nc": "nasopharyngeal carcinoma",
    "bpdcn": "blastic plasmacytoid dendritic cell neoplasm",
    "sts": "soft tissue sarcoma",
    "ra": "rheumatoid arthritis",
    "ms": "multiple sclerosis",
    "sle": "systemic lupus erythematosus",
    "nmo": "neuromyelitis optica",
    "nmosd": "neuromyelitis optica spectrum disorder",
    "mg": "myasthenia gravis",
    "gvhd": "graft versus host disease",
    "mcd": "multicentric castleman disease",
    "msi-h": "microsatellite instability-high cancer",
    "pjia": "juvenile idiopathic arthritis",
    "sjia": "juvenile idiopathic arthritis",
    "amd": "macular degeneration",
    "covid-19": "covid-19",
    "sars-cov-2": "covid-19",
    "bcc": "basal cell carcinoma",
    "erbb2": "her2",
}

# Qualifier phrases that are stripped from the IMGT string before matching.
# These are stage / line-of-therapy / context qualifiers irrelevant to the
# target-disease association lookup.
QUALIFIER_PHRASES = [
    "advanced or metastatic", "advanced or unresectable",
    "recurrent or metastatic", "relapsed or refractory",
    "late-stage or metastatic", "metastatic or unresectable",
    "overexpressing erbb2", "overexpressing her2", "erbb2-positive",
    "her2-positive", "erb2 negative", "cd20 positive",
    "metastatic", "recurrent", "advanced", "refractory", "relapsed",
    "unresectable", "late-stage", "early-stage", "high-risk", "low grade",
    "low-grade", "high-grade", "pediatric", "paediatric", "adult",
    "malignant", "positive", "negative", "in combination with glucocorticoides",
    "with nodal involvement after surgical resection",
    "follicular cd20 positive relapsed or refractory low grade",
    "neovascular (wet)", "neovascular", "(wet)",
]

# IMGT-name prefixes that are organisational, not disease content.
PREFIXES = ["cancers ", "cancer ", "carcinoma ", "lymphoma ", "leukemia ",
            "mesothelioma ", "melanoma "]


# --------------------------------------------------------------------------- #
#  PATHS                                                                       #
# --------------------------------------------------------------------------- #

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

TRIAL_DIR = os.path.join(
    REPO, "analysis_outputs", "IMGT_no_leakage_directional_clinical",
    "clinical_hpo", "RotE", "trial_101")
INTERPRET_JSON = os.path.join(TRIAL_DIR, "interpret_topk_50.json")
PRIOR_JSON = os.path.join(TRIAL_DIR, "indication_prior.json")
REF_PLAUS_JSON = os.path.join(TRIAL_DIR, "interp_plausibility.json")
REF_HGNC_CACHE = os.path.join(TRIAL_DIR, "cache_hgnc2symbol.json")

KG_DIR = os.path.join(REPO, "data", "IMGT_no_leakage_directional_clinical")

CACHE_DIR = os.path.join(HERE, "cache")
HGNC_CACHE = os.path.join(CACHE_DIR, "hgnc2symbol.json")
ENSEMBL_CACHE = os.path.join(CACHE_DIR, "symbol2ensembl.json")
OT_CACHE = os.path.join(CACHE_DIR, "gene_ot_assoc.json")

OUT_CSV = os.path.join(HERE, "results_297.csv")
OUT_SUMMARY = os.path.join(HERE, "summary.json")

OT_GRAPHQL = "https://api.platform.opentargets.org/api/v4/graphql"
MYGENE_URL = "https://mygene.info/v3/query"


# --------------------------------------------------------------------------- #
#  small json cache helpers                                                    #
# --------------------------------------------------------------------------- #

def load_json(path, default=None):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {} if default is None else default


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def http_post_json(url, payload, retries=4):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json",
                 "Accept": "application/json",
                 "User-Agent": "blinded-plausibility/1.0"})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"POST {url} failed after {retries} tries: {last}")


def http_get_json(url, retries=4):
    req = urllib.request.Request(
        url, headers={"Accept": "application/json",
                      "User-Agent": "blinded-plausibility/1.0"})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} tries: {last}")


# --------------------------------------------------------------------------- #
#  KG loading & target resolution                                             #
# --------------------------------------------------------------------------- #

def load_kg():
    """Return (head_index, tail_index): dicts mapping entity -> list of (rel,other)."""
    head_index = defaultdict(list)   # h -> [(r, t), ...]
    tail_index = defaultdict(list)   # t -> [(r, h), ...]
    for split in ("train", "valid", "test"):
        path = os.path.join(KG_DIR, split)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue
                h, r, t = parts
                head_index[h].append((r, t))
                tail_index[t].append((r, h))
    return head_index, tail_index


def resolve_mab(study_product, head_index):
    """StudyProduct --isProductOf--> Product --isProductOf--> mAb."""
    product = None
    for r, t in head_index.get(study_product, []):
        if r == "imgt:isStudyProductOf":
            product = t
            break
    if product is None:
        return None, None
    mab = None
    for r, t in head_index.get(product, []):
        if r == "imgt:isProductOf":
            mab = t
            break
    return product, mab


def resolve_targets(mab, head_index, tail_index):
    """mAb --(sio:SIO_000291)--> HGNC ; also accept reverse isTargetOf."""
    hgnc = []
    if mab is None:
        return hgnc
    for r, t in head_index.get(mab, []):
        if r == "sio:SIO_000291" and t.startswith("HGNC:"):
            hgnc.append(t)
    for r, h in tail_index.get(mab, []):
        if r == "imgt:isTargetOf" and h.startswith("HGNC:"):
            hgnc.append(h)
    # de-dup, keep order
    seen, out = set(), []
    for x in hgnc:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# --------------------------------------------------------------------------- #
#  HGNC -> symbol (MyGene.info)                                               #
# --------------------------------------------------------------------------- #

def hgnc_to_symbol(hgnc_id, cache, live):
    if hgnc_id in cache:
        return cache[hgnc_id]
    if not live:
        return None
    num = hgnc_id.split(":", 1)[1]
    url = (MYGENE_URL + "?q=HGNC:" + urllib.parse.quote(num) +
           "&scopes=HGNC&fields=symbol&species=human")
    try:
        res = http_get_json(url)
        sym = None
        hits = res.get("hits", []) if isinstance(res, dict) else []
        if hits:
            sym = hits[0].get("symbol")
        cache[hgnc_id] = sym
    except Exception:
        cache[hgnc_id] = None
    time.sleep(API_SLEEP)
    return cache[hgnc_id]


# --------------------------------------------------------------------------- #
#  symbol -> Ensembl id, and Ensembl id -> associated diseases (Open Targets) #
# --------------------------------------------------------------------------- #

def symbol_to_ensembl(symbol, cache, live):
    if symbol in cache:
        return cache[symbol]
    if not live:
        return None
    query = """
    query S($q: String!) {
      search(queryString: $q, entityNames: ["target"], page: {index:0, size:5}) {
        hits { id name entity object { ... on Target { approvedSymbol } } }
      }
    }"""
    try:
        res = http_post_json(OT_GRAPHQL, {"query": query, "variables": {"q": symbol}})
        ens = None
        hits = (res.get("data", {}) or {}).get("search", {}).get("hits", []) or []
        # prefer an exact approvedSymbol match
        for h in hits:
            obj = h.get("object") or {}
            if obj.get("approvedSymbol", "").upper() == symbol.upper():
                ens = h.get("id")
                break
        if ens is None and hits:
            ens = hits[0].get("id")
        cache[symbol] = ens
    except Exception:
        cache[symbol] = None
    time.sleep(API_SLEEP)
    return cache[symbol]


def ensembl_assoc_diseases(ensembl_id, symbol, cache, live):
    """Return list of [disease_name_lower, score] for the gene, top OT_DISEASES_PER_GENE."""
    if symbol in cache:
        return cache[symbol]
    if not live or ensembl_id is None:
        return cache.get(symbol)
    query = """
    query A($ens: String!, $size: Int!) {
      target(ensemblId: $ens) {
        approvedSymbol
        associatedDiseases(page: {index:0, size:$size}) {
          rows { score disease { name } }
        }
      }
    }"""
    try:
        res = http_post_json(OT_GRAPHQL, {"query": query,
                                          "variables": {"ens": ensembl_id,
                                                        "size": OT_DISEASES_PER_GENE}})
        tgt = (res.get("data", {}) or {}).get("target") or {}
        rows = (tgt.get("associatedDiseases") or {}).get("rows", []) or []
        out = []
        for row in rows:
            name = ((row.get("disease") or {}).get("name") or "").strip().lower()
            score = row.get("score")
            if name and score is not None:
                out.append([name, round(float(score), 4)])
        cache[symbol] = out
    except Exception:
        cache[symbol] = cache.get(symbol)
    time.sleep(API_SLEEP)
    return cache[symbol]


# --------------------------------------------------------------------------- #
#  Disease-string normalisation                                               #
# --------------------------------------------------------------------------- #

import re
import urllib.parse


def _clean(s):
    s = s.replace("_", " ").strip().lower()
    s = s.replace("'s", "s").replace("’s", "s").replace("'", "").replace("’", "")
    s = re.sub(r"\s+", " ", s)
    return s


def normalise_disease(imgt_ent):
    """
    Turn an IMGT entity string (e.g. 'imgt:Cancers_non-small_cell_lung_(NSCLC)')
    into (core_phrase, keyword_set, is_generic).

    - core_phrase: a cleaned canonical disease phrase used for substring matching
    - keyword_set: a set of phrases (core + any synonym expansions) to match on
    - is_generic : True if the entity is a non-specific bucket node
    """
    ent = imgt_ent
    if ent.startswith("imgt:"):
        ent = ent[5:]
    raw = _clean(ent)

    # collect parenthetical abbreviations BEFORE we strip parens
    abbrevs = re.findall(r"\(([^)]*)\)", raw)
    abbrevs = [a.strip().lower() for a in abbrevs]

    # text with parentheticals removed
    no_paren = re.sub(r"\([^)]*\)", " ", raw)
    no_paren = re.sub(r"\s+", " ", no_paren).strip()

    keywords = set()
    expansions = set()

    # expand any recognised abbreviation
    for a in abbrevs:
        if a in SYNONYM_TABLE:
            expansions.add(SYNONYM_TABLE[a])
    # also handle the case where the whole no_paren *is* a known abbreviation
    if no_paren in SYNONYM_TABLE:
        expansions.add(SYNONYM_TABLE[no_paren])

    # strip qualifier phrases from no_paren to get a cleaner core
    core = no_paren
    for q in sorted(QUALIFIER_PHRASES, key=len, reverse=True):
        core = core.replace(q, " ")
    core = re.sub(r"\s+", " ", core).strip()

    # Is this a generic bucket?  Tested on the FULL (un-qualifier-stripped)
    # phrase so that qualified variants such as "solid tumors advanced or
    # metastatic" remain SPECIFIC (they get an OFF-TARGET tag, matching the
    # original hand-tagging which counted them in the denominator).
    is_generic = no_paren.strip() in GENERIC_TERMS

    # Build a "cancers_X" -> "X cancer" reordering so 'cancers breast'->'breast cancer'
    reordered = None
    for p in PREFIXES:
        if core.startswith(p):
            rest = core[len(p):].strip()
            if rest:
                head = p.strip()
                reordered = f"{rest} {head}"
            break

    if core:
        keywords.add(core)
    if reordered:
        keywords.add(reordered)
    keywords |= expansions
    # also keep the full no_paren phrase
    if no_paren:
        keywords.add(no_paren)

    # drop empty / pure-generic tokens from keyword set used for matching
    keywords = {k for k in keywords if k and k not in GENERIC_TERMS}

    return core, keywords, is_generic, expansions


# Tokens that are NOT disease-discriminative: shared occurrence of one of these
# between an IMGT phrase and an OT disease name does NOT by itself constitute a
# match.  This prevents every specific tumour from spuriously matching Open
# Targets' umbrella entries "cancer" / "neoplasm".  Anatomical-site and
# disease-class tokens (gastric, breast, lymphoma, melanoma, ...) ARE
# discriminative and drive the match.
STOP_TOKENS = {
    "cancer", "cancers", "neoplasm", "neoplasms", "neoplasia", "tumor", "tumour",
    "tumors", "tumours", "malignant", "malignancy", "malignancies", "disease",
    "diseases", "disorder", "disorders", "of", "the", "and", "or", "in", "with",
    "a", "an", "to", "cell", "cells", "syndrome", "advanced", "metastatic",
    "recurrent", "refractory", "relapsed", "primary", "secondary", "stage",
    "positive", "negative", "overexpressing", "type", "patients", "from",
    "solid", "blood", "system", "tissue", "tissues", "high", "low", "grade",
    # Broad HISTOLOGY-suffix words.  On their own they are NOT discriminative
    # (every carcinoma shares "carcinoma"), so a shared "carcinoma"/"adeno-
    # carcinoma"/"squamous" alone does not create a match.  Anatomical site
    # tokens (lung, breast, gastric, ...) and lineage tokens (lymphoma,
    # leukemia, melanoma, ...) remain discriminative.  This stops e.g. HNSCC
    # matching "... lung carcinoma" purely on the suffix "carcinoma".
    "carcinoma", "adenocarcinoma", "squamous", "large",
}

# Multiword discriminative phrases that must match as a whole (single tokens
# would be ambiguous, e.g. "small cell").
DISCRIMINATIVE_PHRASES = [
    "non-small cell lung", "small cell lung", "head and neck",
    "diffuse large b-cell", "triple-negative breast",
    "b-cell", "t-cell", "nk/t", "renal cell", "merkel cell", "basal cell",
    "squamous cell", "hairy cell", "mantle cell", "follicular",
    "hepatocellular", "cholangiocarcinoma", "glioblastoma", "neuroblastoma",
]


def _tokens(phrase):
    """Discriminative token / phrase set for matching."""
    p = phrase.lower()
    toks = set()
    for ph in DISCRIMINATIVE_PHRASES:
        if ph in p:
            toks.add(ph)
    for w in re.split(r"[^a-z0-9]+", p):
        if w and w not in STOP_TOKENS and len(w) >= 4:
            toks.add(w)
    return toks


def match_against_ot(keywords, ot_diseases, threshold):
    """
    ot_diseases: list of [name_lower, score].

    Match rule (token overlap, generic-aware): the IMGT prediction matches an OT
    disease iff they share at least one DISCRIMINATIVE token/phrase (anatomical
    site or disease class) AND that OT disease's overall association score
    >= threshold.  Sharing only a generic token ("cancer"/"neoplasm"/...) does
    NOT count.  Returns (matched_bool, best_score, best_ot_name).
    """
    if not ot_diseases:
        return False, None, None
    kw_tokens = set()
    for kw in keywords:
        kw_tokens |= _tokens(kw)
    if not kw_tokens:
        return False, None, None
    best = (False, -1.0, None)
    for name, score in ot_diseases:
        if score < threshold:
            continue
        ot_tokens = _tokens(name)
        if kw_tokens & ot_tokens:
            if score > best[1]:
                best = (True, score, name)
    if best[0]:
        return True, best[1], best[2]
    return False, None, None


# --------------------------------------------------------------------------- #
#  Scoring                                                                     #
# --------------------------------------------------------------------------- #

def classify_pred(imgt_ent, ot_diseases, threshold):
    """Return one of 'GENERIC','PLAUSIBLE','OFF-TARGET', plus (matched_ot, score)."""
    core, keywords, is_generic, _ = normalise_disease(imgt_ent)
    if is_generic:
        return "GENERIC", None, None, True
    matched, score, ot_name = match_against_ot(keywords, ot_diseases, threshold)
    if matched:
        return "PLAUSIBLE", ot_name, score, False
    return "OFF-TARGET", None, None, False


def score_query_topk(top_ents, ot_diseases, threshold):
    """top_ents: list of imgt entity strings (already top-20).
    Returns dict with counts and per-prediction tags."""
    n_specific = n_plausible = n_offtarget = n_generic = 0
    tags = []
    for ent in top_ents:
        tag, ot_name, score, is_generic = classify_pred(ent, ot_diseases, threshold)
        tags.append({"pred": ent, "tag": tag, "ot_match": ot_name, "ot_score": score})
        if tag == "GENERIC":
            n_generic += 1
        elif tag == "PLAUSIBLE":
            n_specific += 1
            n_plausible += 1
        else:
            n_specific += 1
            n_offtarget += 1
    frac = (n_plausible / n_specific) if n_specific else None
    return {"n_generic": n_generic, "n_specific": n_specific,
            "n_plausible": n_plausible, "n_offtarget": n_offtarget,
            "fraction": frac, "tags": tags}


def baseline_fraction(prior_names, prior_weights, ot_diseases, threshold, rng,
                      cache_tag):
    """Monte-Carlo: draw PRIOR_DRAW_SIZE indications (weighted, with replacement)
    from the empirical prior, score by the SAME rule, average plausible-fraction.

    `cache_tag` is a per-ent dict mapping ent->tag (specific/plausible) computed
    once per gene-set to avoid re-normalising the same ent thousands of times.
    """
    fracs = []
    n = len(prior_names)
    for _ in range(N_PRIOR_DRAWS):
        idxs = rng.choices(range(n), weights=prior_weights, k=PRIOR_DRAW_SIZE)
        n_spec = n_plaus = 0
        for i in idxs:
            tag = cache_tag[i]
            if tag == "GENERIC":
                continue
            n_spec += 1
            if tag == "PLAUSIBLE":
                n_plaus += 1
        if n_spec:
            fracs.append(n_plaus / n_spec)
        else:
            fracs.append(0.0)
    return sum(fracs) / len(fracs) if fracs else 0.0


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def bootstrap_ci(values, statistic, n_boot, seed, alpha=0.05):
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return (None, None)
    boots = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(statistic(sample))
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot)]
    return (lo, hi)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline", action="store_true",
                    help="Do not hit any API; use only on-disk caches.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N queries (debug).")
    ap.add_argument("--validate-only", action="store_true",
                    help="Only run the 5-case validation and exit.")
    args = ap.parse_args()
    live = not args.offline

    os.makedirs(CACHE_DIR, exist_ok=True)

    # --- load inputs -----------------------------------------------------
    interp = load_json(INTERPRET_JSON)
    results = interp["results"]
    prior = load_json(PRIOR_JSON)
    prior_counts = prior["counts"]

    # --- caches (seed HGNC cache from the existing reference cache) -------
    hgnc_cache = load_json(HGNC_CACHE, {})
    if not hgnc_cache and os.path.exists(REF_HGNC_CACHE):
        hgnc_cache = dict(load_json(REF_HGNC_CACHE, {}))
    ensembl_cache = load_json(ENSEMBL_CACHE, {})
    ot_cache = load_json(OT_CACHE, {})

    head_index, tail_index = load_kg()

    # --- resolve genes & OT associations for every gene we will need ------
    def get_gene_ot(symbol):
        """symbol -> list[[disease, score]] (cached, possibly via API)."""
        if symbol in ot_cache:
            return ot_cache[symbol]
        ens = symbol_to_ensembl(symbol, ensembl_cache, live)
        save_json(ENSEMBL_CACHE, ensembl_cache)
        assoc = ensembl_assoc_diseases(ens, symbol, ot_cache, live)
        save_json(OT_CACHE, ot_cache)
        return assoc

    # =====================================================================
    #  5-CASE VALIDATION
    # =====================================================================
    ref = load_json(REF_PLAUS_JSON, [])
    val_lines = []
    val_targets = {"PDCD1": (9, 17), "MSLN": (10, 18), "ERBB2": (9, 17),
                   "EGFR": (9, 18), "TNFRSF17": (14, 18)}
    print("=== 5-case validation (blinded auto-tagger vs original tags) ===")
    for case in ref:
        gene = case["gene"]
        top_ents = [a["pred"] for a in case["analysis"]]  # already top-20
        ot = get_gene_ot(gene)
        if ot is None:
            print(f"  {gene}: OT assoc unavailable (offline & uncached) -> skip")
            continue
        sc = score_query_topk(top_ents, ot, PLAUSIBILITY_THRESHOLD)
        tgt = val_targets.get(gene, (None, None))
        line = (f"  {gene:9s} blinded={sc['n_plausible']}/{sc['n_specific']}"
                f"   original={tgt[0]}/{tgt[1]}")
        print(line)
        val_lines.append({"gene": gene,
                          "blinded_plausible": sc["n_plausible"],
                          "blinded_specific": sc["n_specific"],
                          "original_plausible": tgt[0],
                          "original_specific": tgt[1]})
    save_json(HGNC_CACHE, hgnc_cache)
    if args.validate_only:
        save_json(os.path.join(HERE, "validation.json"), val_lines)
        return

    # =====================================================================
    #  RESOLVE TARGETS FOR ALL 297 QUERIES (blinded - no rank/true-tail use)
    # =====================================================================
    rows = []
    queries = results if args.limit is None else results[:args.limit]
    n_total = len(queries)
    print(f"\n=== scoring {n_total} forward queries ===")

    # First pass: resolve all genes & fetch OT (cache-building, resumable)
    for qi, q in enumerate(queries):
        sp = q["head"]
        _, mab = resolve_mab(sp, head_index)
        hgncs = resolve_targets(mab, head_index, tail_index)
        symbols = []
        for h in hgncs:
            sym = hgnc_to_symbol(h, hgnc_cache, live)
            if sym:
                symbols.append(sym)
        save_json(HGNC_CACHE, hgnc_cache)
        for sym in symbols:
            get_gene_ot(sym)
        q["_mab"] = mab
        q["_symbols"] = symbols
        if (qi + 1) % 25 == 0:
            print(f"  resolved {qi+1}/{n_total} targets")

    # =====================================================================
    #  PRIOR baseline machinery (shared structures)
    # =====================================================================
    # Build prior arrays once.
    prior_names = list(prior_counts.keys())
    prior_weights = [prior_counts[k] for k in prior_names]

    # =====================================================================
    #  SCORE each query (model fraction + prior baseline + enrichment)
    # =====================================================================
    rng = random.Random(RANDOM_SEED)
    n_scored = 0
    coverage_total = 0          # specific predictions across all queries
    coverage_matched = 0        # specific preds that matched an OT disease (any gene, thr=0.01)
    coverage_specific_seen = 0

    # caches for baseline per gene-set (so we don't re-normalise prior 1000x/query)
    baseline_cache = {}

    for q in queries:
        symbols = q.get("_symbols", [])
        # union OT diseases across the (possibly multiple) target genes
        ot_union = []
        ok = True
        for sym in symbols:
            a = ot_cache.get(sym)
            if a is None:
                ok = False
            else:
                ot_union.extend(a)
        top_ents = [item["ent"] for item in q["top_k"][:TOP_K]]

        if not symbols or not ok or not ot_union:
            # cannot score - no resolvable target / OT data
            rows.append({
                "query_idx": q["query_idx"], "mab": q.get("_mab"),
                "study_product": q["head"], "genes": ";".join(symbols),
                "n_specific": "", "n_plausible": "", "model_fraction": "",
                "prior_fraction": "", "enrichment": "", "n_unmatched": "",
                "status": "no_target_or_ot",
            })
            continue

        sc = score_query_topk(top_ents, ot_union, PLAUSIBILITY_THRESHOLD)

        # --- coverage accounting (how many specific preds matched anything) ---
        n_unmatched = 0
        for t in sc["tags"]:
            if t["tag"] == "GENERIC":
                continue
            coverage_specific_seen += 1
            if t["tag"] == "PLAUSIBLE":
                coverage_matched += 1
            else:
                # specific but unmatched at thr 0.01
                n_unmatched += 1
        coverage_total += sc["n_specific"]

        # --- baseline (prior) on the SAME gene-set --------------------------
        key = tuple(sorted(symbols))
        if key not in baseline_cache:
            # precompute the tag of every prior ent against this ot_union
            tag_by_idx = []
            for nm in prior_names:
                tag, _, _, is_gen = classify_pred(nm, ot_union, PLAUSIBILITY_THRESHOLD)
                tag_by_idx.append(tag)
            baseline_cache[key] = tag_by_idx
        tag_by_idx = baseline_cache[key]
        prior_frac = baseline_fraction(prior_names, prior_weights, ot_union,
                                        PLAUSIBILITY_THRESHOLD, rng, tag_by_idx)

        model_frac = sc["fraction"]
        if model_frac is None:
            enrichment = None
        elif prior_frac and prior_frac > 0:
            enrichment = model_frac / prior_frac
        else:
            enrichment = None

        rows.append({
            "query_idx": q["query_idx"], "mab": q.get("_mab"),
            "study_product": q["head"], "genes": ";".join(symbols),
            "n_specific": sc["n_specific"], "n_plausible": sc["n_plausible"],
            "model_fraction": round(model_frac, 4) if model_frac is not None else "",
            "prior_fraction": round(prior_frac, 4),
            "enrichment": round(enrichment, 4) if enrichment is not None else "",
            "n_unmatched": n_unmatched,
            "status": "scored",
        })
        n_scored += 1

    # =====================================================================
    #  WRITE results_297.csv
    # =====================================================================
    fieldnames = ["query_idx", "mab", "study_product", "genes",
                  "n_specific", "n_plausible", "model_fraction",
                  "prior_fraction", "enrichment", "n_unmatched", "status"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # =====================================================================
    #  THRESHOLD SENSITIVITY  (re-score model + baseline at 0.05 / 0.1)
    # =====================================================================
    sens = {}
    for thr in SENSITIVITY_THRESHOLDS:
        m_fracs, p_fracs, enr = [], [], []
        for q in queries:
            symbols = q.get("_symbols", [])
            ot_union = []
            ok = True
            for sym in symbols:
                a = ot_cache.get(sym)
                if a is None:
                    ok = False
                else:
                    ot_union.extend(a)
            if not symbols or not ok or not ot_union:
                continue
            top_ents = [item["ent"] for item in q["top_k"][:TOP_K]]
            sc = score_query_topk(top_ents, ot_union, thr)
            if sc["fraction"] is None:
                continue
            tag_by_idx = [classify_pred(nm, ot_union, thr)[0] for nm in prior_names]
            pf = baseline_fraction(prior_names, prior_weights, ot_union, thr,
                                   random.Random(RANDOM_SEED), tag_by_idx)
            m_fracs.append(sc["fraction"])
            p_fracs.append(pf)
            if pf > 0:
                enr.append(sc["fraction"] / pf)
        sens[str(thr)] = {
            "mean_model_fraction": mean(m_fracs),
            "mean_prior_fraction": mean(p_fracs),
            "mean_enrichment": mean(enr),
            "frac_enrichment_gt1": (sum(1 for e in enr if e > 1) / len(enr)) if enr else None,
            "n_scored": len(m_fracs),
        }

    # =====================================================================
    #  AGGREGATE summary.json
    # =====================================================================
    scored_rows = [r for r in rows if r["status"] == "scored"]
    model_fracs = [r["model_fraction"] for r in scored_rows if r["model_fraction"] != ""]
    prior_fracs = [r["prior_fraction"] for r in scored_rows if r["prior_fraction"] != ""]
    enrichments = [r["enrichment"] for r in scored_rows if r["enrichment"] != ""]

    ci = bootstrap_ci(enrichments, lambda s: sum(s) / len(s) if s else 0.0,
                      N_BOOTSTRAP, RANDOM_SEED)

    summary = {
        "pre_registration": {
            "threshold": PLAUSIBILITY_THRESHOLD,
            "sensitivity_thresholds": SENSITIVITY_THRESHOLDS,
            "top_k": TOP_K,
            "n_prior_draws": N_PRIOR_DRAWS,
            "prior_draw_size": PRIOR_DRAW_SIZE,
            "random_seed": RANDOM_SEED,
            "ot_diseases_per_gene": OT_DISEASES_PER_GENE,
            "blinded": True,
            "note": ("Tags computed only from model top-20 predictions and the "
                     "query mAb target gene(s); ground-truth tail/rank never used."),
        },
        "n_queries_total": n_total,
        "n_queries_scored": len(scored_rows),
        "n_queries_unscorable": n_total - len(scored_rows),
        "model_fraction_mean": mean(model_fracs),
        "model_fraction_median": median(model_fracs),
        "baseline_fraction_mean": mean(prior_fracs),
        "baseline_fraction_median": median(prior_fracs),
        "enrichment_mean": mean(enrichments),
        "enrichment_median": median(enrichments),
        "enrichment_bootstrap_ci95": {"low": ci[0], "high": ci[1]},
        "frac_queries_enrichment_gt1": (
            sum(1 for e in enrichments if e > 1) / len(enrichments)
            if enrichments else None),
        "normalisation_coverage": {
            "specific_predictions_scored": coverage_specific_seen,
            "specific_predictions_matched_OT": coverage_matched,
            "coverage_fraction": (coverage_matched / coverage_specific_seen
                                  if coverage_specific_seen else None),
            "note": ("coverage_fraction = fraction of SPECIFIC (non-generic) top-20 "
                     "predictions that matched an OT disease for the target at thr=0.01; "
                     "the complement are OFF-TARGET (no qualifying association)."),
        },
        "threshold_sensitivity": sens,
        "validation_5case": val_lines,
    }
    save_json(OUT_SUMMARY, summary)

    print(f"\n=== DONE ===")
    print(f"  scored {len(scored_rows)}/{n_total} queries")
    print(f"  mean model fraction   = {summary['model_fraction_mean']}")
    print(f"  mean baseline fraction= {summary['baseline_fraction_mean']}")
    print(f"  mean enrichment       = {summary['enrichment_mean']}  "
          f"CI95=[{ci[0]}, {ci[1]}]")
    print(f"  frac enrichment>1     = {summary['frac_queries_enrichment_gt1']}")
    print(f"  norm coverage         = {summary['normalisation_coverage']['coverage_fraction']}")
    print(f"  wrote: {OUT_CSV}")
    print(f"  wrote: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
