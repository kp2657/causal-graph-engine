"""
pipelines/program_annotator.py — Phase 1+2: per-program biological + genetic identity.

Phase 1 — biological identity (per program):
  - functional_terms : top MSigDB Hallmark overlaps (hypergeometric ORA)
  - cell_state       : Resting / Activated / Mixed (from activation bias; Schnitzler=Mixed)
  - direction        : atherogenic (+1) / protective (-1) / depleted (0)
  - top_genes        : top-N genes by |beta| in this program
  - gwas_t           : GWAS-alignment t-score (from gwas_aligned_programs.json)
  - gamma            : S-LDSC / eQTL-direction γ

Phase 2 — human genetic interpretation (per program):
  - gwas_anchor_enrichment : Fisher's exact p for GWAS-proximal gene over-representation
  - gwas_anchor_overlap    : genes at the intersection of top-N and GWAS-proximal set
  - genetic_direction_score: Σ sign(eQTL_β) × sign(program_β) / n_eqtl_genes
                             Positive → risk alleles coordinately activate this program
                             Negative → risk alleles suppress this program
  - wes_direction_score    : fraction of top genes with WES burden direction concordant
                             with program direction (concordant = LoF effect opposes γ sign)
  - n_wes_hits             : number of top genes with nominal WES association (p<0.05)

Output is written to  data/perturbseq/{dataset_id}/program_annotations.json
and returned as a dict  {program_name → annotation_dict}.

Public API
----------
annotate_programs(disease_key: str) -> dict[str, dict]
    Top-level call. Loads all required files and returns the annotation dict.
    Also writes the JSON file.
"""
from __future__ import annotations

import json
import logging
import math
import pathlib
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_ROOT         = pathlib.Path(__file__).parent.parent
_PERTURBSEQ   = _ROOT / "data" / "perturbseq"
_MSIGDB_PATH  = _ROOT / "data" / "msigdb_hallmark_2024.json"
_EQTL_IDX_DIR = _ROOT / "data" / "eqtl" / "indices"
_WES_DIR      = _ROOT / "data" / "wes"

# Number of top-|beta| genes to use for ORA
_TOP_N_GENES  = 50
# Minimum overlap for a Hallmark term to be reported
_MIN_OVERLAP  = 3
# Maximum BH-corrected p-value to include a term
_MAX_FDR      = 0.20
# WES nominal significance threshold (not corrected — used for n_wes_hits count)
_WES_NOMINAL_P = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_msigdb() -> dict[str, frozenset[str]]:
    with open(_MSIGDB_PATH) as f:
        raw = json.load(f)
    return {k: frozenset(v) for k, v in raw.items()}


def _hypergeometric_p(k: int, M: int, n: int, N: int) -> float:
    """P(X >= k) for hypergeometric(M, n, N).
    M = background size, n = gene set size, N = query size, k = overlap.
    """
    from math import lgamma, exp, log as mlog

    def log_comb(a: int, b: int) -> float:
        if b < 0 or b > a:
            return -math.inf
        return lgamma(a + 1) - lgamma(b + 1) - lgamma(a - b + 1)

    log_denom = log_comb(M, N)
    if math.isinf(log_denom) or log_denom == 0:
        return 1.0

    p = 0.0
    for i in range(k, min(n, N) + 1):
        lp = log_comb(n, i) + log_comb(M - n, N - i) - log_denom
        p += math.exp(lp) if lp > -700 else 0.0
    return min(p, 1.0)


def _bh_correct(pvalues: list[float]) -> list[float]:
    n = len(pvalues)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvalues[i])
    ranks = [0] * n
    for r, i in enumerate(order, 1):
        ranks[i] = r
    adjusted = [min(pvalues[i] * n / ranks[i], 1.0) for i in range(n)]
    # Enforce monotonicity (step-down)
    min_adj = 1.0
    for i in sorted(range(n), key=lambda i: -ranks[i]):
        min_adj = min(min_adj, adjusted[i])
        adjusted[i] = min_adj
    return adjusted


def _run_ora(
    query_genes: set[str],
    gene_sets: dict[str, frozenset[str]],
    background_size: int,
    min_overlap: int = _MIN_OVERLAP,
    max_fdr: float = _MAX_FDR,
) -> list[dict]:
    """Fast hypergeometric ORA. Returns list of {term, overlap, p, fdr, genes}."""
    results = []
    for term, gs in gene_sets.items():
        overlap_genes = query_genes & gs
        k = len(overlap_genes)
        if k < min_overlap:
            continue
        p = _hypergeometric_p(k, background_size, len(gs), len(query_genes))
        results.append({"term": term, "overlap": k, "p": p, "genes": sorted(overlap_genes)})

    if not results:
        return []

    pvals = [r["p"] for r in results]
    fdrs  = _bh_correct(pvals)
    for r, fdr in zip(results, fdrs):
        r["fdr"] = round(fdr, 4)
        r["p"]   = round(r["p"], 6)

    return sorted(
        [r for r in results if r["fdr"] <= max_fdr],
        key=lambda r: r["p"],
    )


def _cell_state_label(program: str, activation_biases: dict[str, float] | None) -> str:
    """Resting / Activated / Mixed based on bias ratio. Mixed when no data."""
    if not activation_biases:
        return "Mixed"
    # activation_biases keys are gene symbols; average over top program genes is
    # computed by load_program_activation_biases — we receive the per-program value.
    bias = activation_biases.get(program)
    if bias is None:
        return "Mixed"
    from config.scoring_thresholds import TIMEPOINT_ACTIVATION_BIAS_MIN
    if bias >= TIMEPOINT_ACTIVATION_BIAS_MIN:
        return "Activated"
    elif bias <= (1.0 / TIMEPOINT_ACTIVATION_BIAS_MIN):
        return "Resting"
    return "Mixed"


# ---------------------------------------------------------------------------
# Phase 2 helpers
# ---------------------------------------------------------------------------

def _load_eqtl_betas(disease_key: str) -> dict[str, float]:
    """Load cis-eQTL betas from pre-built local indices (gene → beta of lead eQTL)."""
    betas: dict[str, float] = {}
    for path in _EQTL_IDX_DIR.glob(f"{disease_key.upper()}_*_top_eqtls.json"):
        try:
            with open(path) as f:
                idx = json.load(f)
            if isinstance(idx, dict):
                for gene, rec in idx.items():
                    if isinstance(rec, dict) and "beta" in rec:
                        betas.setdefault(gene, float(rec["beta"]))
        except Exception as e:
            log.warning("Could not load eQTL index %s: %s", path.name, e)
    return betas


def _load_wes_burden(disease_key: str) -> dict[str, dict]:
    """Load UKB WES gene burden betas from local JSON (Backman et al. 2021)."""
    path = _WES_DIR / f"{disease_key.upper()}_burden.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        log.warning("Could not load WES burden %s: %s", path.name, e)
        return {}


def _fisher_exact_enrichment(
    overlap: int,
    query_n: int,
    gwas_n: int,
    background_n: int,
) -> float:
    """One-sided Fisher exact p for GWAS enrichment in query set.
    Using hypergeometric as Fisher (equivalent for one-sided enrichment).
    """
    if overlap == 0 or gwas_n == 0:
        return 1.0
    return _hypergeometric_p(overlap, background_n, gwas_n, query_n)


def _genetic_direction_score(
    top_genes: list[str],
    top_betas: dict[str, float],
    eqtl_betas: dict[str, float],
    gwas_proximal: set[str],
) -> tuple[float, int]:
    """Genetic direction vector: Σ sign(eQTL_β) × sign(program_β) for GWAS genes.

    Returns (score, n_genes_used).
    score > 0 → risk alleles activate this program (more expression → more disease)
    score < 0 → risk alleles suppress this program (less expression → more disease)
    """
    signs = []
    for gene in top_genes:
        if gene not in gwas_proximal:
            continue
        eqtl_b = eqtl_betas.get(gene)
        prog_b = top_betas.get(gene)
        if eqtl_b is None or prog_b is None or prog_b == 0.0:
            continue
        signs.append(math.copysign(1.0, eqtl_b) * math.copysign(1.0, prog_b))
    if not signs:
        return 0.0, 0
    return round(sum(signs) / len(signs), 3), len(signs)


def _wes_direction_stats(
    top_genes: list[str],
    direction_int: int,
    wes_burden: dict[str, dict],
) -> tuple[float | None, int]:
    """WES burden direction concordance with program direction.

    Concordant: sign(-burden_beta) == direction_int
      burden_beta > 0 (LoF → more disease) → gene is protective → should be in protective program (dir=-1)
      burden_beta < 0 (LoF → less disease) → gene is atherogenic → should be in atherogenic program (dir=+1)

    Returns (concordance_fraction, n_wes_nominal_hits).
    concordance_fraction=None if no genes with WES data found.
    """
    if direction_int == 0:
        # Depleted program — no directional expectation
        n_hits = sum(
            1 for g in top_genes
            if (rec := wes_burden.get(g)) and rec.get("burden_p", 1.0) < _WES_NOMINAL_P
        )
        return None, n_hits

    concordant = 0
    tested = 0
    n_hits = 0
    for gene in top_genes:
        rec = wes_burden.get(gene)
        if rec is None:
            continue
        burden_beta = rec.get("burden_beta")
        burden_p = rec.get("burden_p", 1.0)
        if burden_beta is None:
            continue
        if burden_p < _WES_NOMINAL_P:
            n_hits += 1
        tested += 1
        # LoF increases disease (beta>0) → gene is protective → concordant with dir=-1
        # LoF decreases disease (beta<0) → gene is atherogenic → concordant with dir=+1
        expected_sign = -math.copysign(1.0, burden_beta)  # -sign(burden_beta) = program_dir
        if expected_sign == direction_int:
            concordant += 1

    if tested == 0:
        return None, n_hits
    return round(concordant / tested, 3), n_hits


# ---------------------------------------------------------------------------
# Phase 3: mechanistic narrative
# ---------------------------------------------------------------------------

# Thresholds for narrative language
_GEN_DIR_STRONG  = 0.4   # |score| ≥ this → confident statement about direction
_WES_CONCORDANT  = 0.60  # ≥ this fraction → "WES burden concordant"
_WES_DISCORDANT  = 0.40  # ≤ this fraction → "WES burden discordant"


def _make_mechanistic_narrative(ann: dict) -> str:
    """Assemble a 2–3 sentence mechanistic hypothesis from Phase 1+2 annotation fields.

    All logic is deterministic template-filling — no API calls.
    """
    direction    = ann.get("direction", "depleted")
    direction_int= ann.get("direction_int", 0)
    cell_state   = ann.get("cell_state", "Mixed")
    gamma        = ann.get("gamma", 0.0)
    top_genes    = ann.get("top_genes", [])
    func_terms   = ann.get("functional_terms", [])
    gen_dir      = ann.get("genetic_direction_score", 0.0) or 0.0
    n_eqtl       = ann.get("n_eqtl_genes", 0)
    wes_score    = ann.get("wes_direction_score")   # None = no data
    n_wes        = ann.get("n_wes_hits", 0)
    gwas_anchors = ann.get("gwas_anchor_overlap", [])

    key_genes = (gwas_anchors[:3] if gwas_anchors else top_genes[:3])
    gene_str  = ", ".join(key_genes) if key_genes else "no GWAS-proximal genes identified"

    # --- Sentence 1: program identity + functional enrichment ---
    state_adj = {"Activated": "activated", "Resting": "resting"}.get(cell_state, "mixed-state")
    dir_adj   = {"atherogenic": "atherogenic", "protective": "protective", "depleted": "statistically depleted"}.get(direction, "depleted")

    if func_terms:
        top_term = func_terms[0]["term"].replace("HALLMARK_", "").replace("_", " ").title()
        s1 = (
            f"This {state_adj} {dir_adj} program (γ={gamma:+.3f}) is enriched for "
            f"{top_term} (FDR={func_terms[0]['fdr']:.2f}), "
            f"with top GWAS-proximal genes: {gene_str}."
        )
    else:
        s1 = (
            f"This {state_adj} {dir_adj} program (γ={gamma:+.3f}) shows no significant "
            f"Hallmark enrichment; top GWAS-proximal genes: {gene_str}."
        )

    if direction == "depleted":
        return s1 + " No directional genetic interpretation applies to a depleted program."

    # --- Sentence 2: genetic direction (eQTL-based) ---
    if n_eqtl == 0:
        s2 = "No cis-eQTL data available to determine whether risk alleles activate or suppress this program."
    elif abs(gen_dir) >= _GEN_DIR_STRONG:
        verb = "activate" if gen_dir > 0 else "suppress"
        s2 = (
            f"GWAS risk alleles coordinately {verb} this program "
            f"(genetic direction score {gen_dir:+.2f} across {n_eqtl} eQTL genes), "
            f"consistent with the program's {direction} classification."
        )
    else:
        s2 = (
            f"GWAS risk alleles show mixed directionality with respect to this program "
            f"(score {gen_dir:+.2f}, {n_eqtl} eQTL genes), suggesting heterogeneous "
            f"genetic architecture or partial program overlap."
        )

    # --- Sentence 3: WES concordance + therapeutic implication ---
    inhibit_or_activate = "inhibiting" if direction_int == 1 else "activating"
    expected_outcome    = "protective" if direction_int == 1 else "atherogenic-risk-reducing"

    if wes_score is None:
        s3 = (
            f"WES rare-variant burden data are unavailable for this gene set. "
            f"If the eQTL-direction signal is correct, {inhibit_or_activate} key regulators "
            f"of this program is predicted to be {expected_outcome}."
        )
    elif wes_score >= _WES_CONCORDANT:
        s3 = (
            f"WES burden direction is concordant ({wes_score:.0%} of tested genes, "
            f"{n_wes} nominally significant), independently supporting that "
            f"{inhibit_or_activate} this program would be {expected_outcome}."
        )
    elif wes_score <= _WES_DISCORDANT:
        s3 = (
            f"WES burden direction is discordant ({wes_score:.0%} concordance, "
            f"{n_wes} nominally significant) — rare LoF variants do not support "
            f"the expected {direction} role; interpret this program with caution."
        )
    else:
        s3 = (
            f"WES burden direction is equivocal ({wes_score:.0%} concordance, "
            f"{n_wes} nominally significant); rare-variant evidence neither strongly "
            f"supports nor contradicts the {direction} classification."
        )

    return " ".join([s1, s2, s3])


# ---------------------------------------------------------------------------
# Main annotation function
# ---------------------------------------------------------------------------

def annotate_programs(disease_key: str) -> dict[str, dict]:
    """
    Compute Phase 1+2 program annotations for disease_key (e.g. "cad", "ra").

    Returns dict {program_name → annotation_dict} and writes
    data/perturbseq/{dataset_id}/program_annotations.json.
    """
    from graph.schema import DISEASE_CELL_TYPE_MAP, _DISEASE_SHORT_NAMES_FOR_ANCHORS
    from pipelines.ldsc.gamma_loader import get_genetic_nmf_program_gammas, get_locus_program_gammas
    from config.scoring_thresholds import SLDSC_GAMMA_FLOOR

    # Resolve dataset_id
    short_key = _DISEASE_SHORT_NAMES_FOR_ANCHORS.get(disease_key.lower(), disease_key.upper())
    ctx = DISEASE_CELL_TYPE_MAP.get(short_key, {})
    dataset_id = ctx.get("scperturb_dataset")
    if not dataset_id:
        raise ValueError(f"No scperturb_dataset for disease_key={disease_key!r}")

    ds_dir = _PERTURBSEQ / dataset_id
    # Resolve actual data directory: check both scperturb_dataset and perturb_seq_source dirs
    # (CAD: scperturb_dataset=Schnitzler_GSE210681, but npz lives in schnitzler_cad_vascular)
    def _find_ds_dir(fname: str) -> pathlib.Path | None:
        if (ds_dir / fname).exists():
            return ds_dir
        for alt_key in ("perturb_seq_source", "perturbseq_server_id"):
            alt_id = ctx.get(alt_key)
            if alt_id and (_PERTURBSEQ / alt_id / fname).exists():
                return _PERTURBSEQ / alt_id
        return None

    resolved = _find_ds_dir("genetic_nmf_loadings.npz") or _find_ds_dir("svd_loadings.npz")
    if resolved is None:
        raise FileNotFoundError(f"No genetic_nmf_loadings.npz or svd_loadings.npz for {dataset_id!r}")
    ds_dir = resolved

    npz_file = "genetic_nmf_loadings.npz" if (ds_dir / "genetic_nmf_loadings.npz").exists() else "svd_loadings.npz"
    prog_source = "GeneticNMF" if npz_file == "genetic_nmf_loadings.npz" else "SVD"
    log.info("Loading %s from %s (%s)", npz_file, ds_dir.name, prog_source)

    npz = np.load(ds_dir / npz_file)
    Vt          = npz["Vt"]           # (n_components × n_perts)
    pert_names  = list(npz["pert_names"].tolist())
    n_progs, n_perts = Vt.shape
    prefix     = f"{short_key}_{prog_source}"
    prog_names = [f"{prefix}_C{c+1:02d}" for c in range(n_progs)]

    gammas = get_genetic_nmf_program_gammas(short_key)
    if not gammas:
        log.info("GeneticNMF gammas not yet computed for %s — run run_genetic_nmf_programs()", short_key)
    # Merge locus-anchored gammas (LOCUS_* programs take precedence when present)
    gammas = {**gammas, **get_locus_program_gammas(short_key)}
    gamma_map = {p: d["gamma"] for p, d in gammas.items()}

    # --- GWAS alignment scores ---
    gwas_t_map: dict[str, float] = {}
    gwa_path = ds_dir / "gwas_aligned_programs.json"
    if gwa_path.exists():
        with open(gwa_path) as f:
            raw = json.load(f)
        # Format: list of [prog_name, score], list of dicts, or dict
        if isinstance(raw, list):
            for entry in raw:
                if isinstance(entry, dict):
                    pid = entry.get("program_id") or entry.get("program")
                    score = entry.get("gwas_t_stat") or entry.get("gwas_t") or entry.get("score", 0.0)
                    if pid:
                        gwas_t_map[pid] = float(score)
                elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    gwas_t_map[entry[0]] = float(entry[1])
        elif isinstance(raw, dict):
            for k, v in raw.items():
                if isinstance(v, (int, float)):
                    gwas_t_map[k] = float(v)
                elif isinstance(v, dict):
                    gwas_t_map[k] = float(v.get("gwas_t", v.get("score", 0.0)))

    # --- Activation biases (per-program) ---
    prog_activation_biases: dict[str, float] = {}
    bias_path = ds_dir / "gene_activation_biases.json"
    if bias_path.exists():
        try:
            from mcp_servers.perturbseq_server import load_program_activation_biases
            # Build program→gene-list dict from top Vt loadings (perturbed gene symbols)
            _top_n = 50
            _prog_gene_sets: dict[str, list[str]] = {}
            for _pi, _pname in enumerate(prog_names):
                _row = np.abs(Vt[_pi, :])
                _top_idx = np.argsort(_row)[::-1][:_top_n]
                _prog_gene_sets[_pname] = [pert_names[i] for i in _top_idx]
            result = load_program_activation_biases(dataset_id, _prog_gene_sets)
            if result:
                prog_activation_biases = result
        except Exception as e:
            log.warning("Could not load activation biases for %s: %s", dataset_id, e)

    # --- MSigDB gene sets ---
    msigdb = _load_msigdb()
    # Background = all perturbed gene symbols
    background_genes = set(pert_names)
    background_size  = len(background_genes)

    # --- Phase 2: GWAS eQTL + WES data ---
    eqtl_betas  = _load_eqtl_betas(short_key)
    wes_burden  = _load_wes_burden(short_key)
    # GWAS-proximal = genes with a cis-eQTL in the GWAS-relevant tissue
    gwas_proximal: set[str] = set(eqtl_betas.keys()) & background_genes

    # --- Annotate each program ---
    annotations: dict[str, dict] = {}

    for c_idx, prog in enumerate(prog_names):
        row = Vt[c_idx]  # (n_perts,) — loading of each gene onto this component

        # Top-N genes by |loading|
        top_idx   = np.argsort(np.abs(row))[::-1][:_TOP_N_GENES]
        top_genes = [pert_names[i] for i in top_idx]
        top_betas = {pert_names[i]: float(row[i]) for i in top_idx}

        # ORA
        ora_hits = _run_ora(set(top_genes), msigdb, background_size)
        functional_terms = [
            {"term": h["term"], "overlap": h["overlap"], "fdr": h["fdr"], "p": h["p"]}
            for h in ora_hits[:3]
        ]

        # Direction from gamma (unsigned |τ*|: all enriched programs are disease_relevant)
        g = gamma_map.get(prog, 0.0)
        if abs(g) < SLDSC_GAMMA_FLOOR:
            direction = "depleted"
            direction_int = 0
        else:
            direction = "disease_relevant"
            direction_int = 1

        # Cell state
        cell_state = _cell_state_label(prog, prog_activation_biases)

        # --- Phase 2: genetic signals ---
        # GWAS anchor enrichment (Fisher / hypergeometric)
        anchor_overlap = sorted(set(top_genes) & gwas_proximal)
        gwas_enrich_p  = _fisher_exact_enrichment(
            len(anchor_overlap), _TOP_N_GENES, len(gwas_proximal), background_size
        )

        # Genetic direction vector
        gen_dir_score, n_eqtl_genes = _genetic_direction_score(
            top_genes, top_betas, eqtl_betas, gwas_proximal
        )

        # WES burden concordance
        wes_concordance, n_wes_hits = _wes_direction_stats(top_genes, direction_int, wes_burden)

        ann_record = {
            "program":                  prog,
            "gamma":                    round(g, 5),
            "direction":                direction,
            "direction_int":            direction_int,
            "gwas_t":                   round(gwas_t_map.get(prog, 0.0), 4),
            "cell_state":               cell_state,
            "top_genes":                top_genes,
            "top_betas":                top_betas,
            "functional_terms":         functional_terms,
            "n_background":             background_size,
            # Phase 2 — genetic interpretation
            "gwas_anchor_enrichment_p": round(gwas_enrich_p, 5),
            "gwas_anchor_overlap":      anchor_overlap,
            "genetic_direction_score":  gen_dir_score,
            "n_eqtl_genes":             n_eqtl_genes,
            "wes_direction_score":      wes_concordance,
            "n_wes_hits":               n_wes_hits,
        }
        # Phase 3 — mechanistic narrative (built after all fields are set)
        ann_record["mechanistic_narrative"] = _make_mechanistic_narrative(ann_record)
        annotations[prog] = ann_record

    # Write output
    out_path = ds_dir / "program_annotations.json"
    with open(out_path, "w") as f:
        json.dump(annotations, f, indent=2)
    log.info("Program annotations written: %d programs → %s", len(annotations), out_path)

    return annotations


def load_program_annotations(disease_key: str) -> dict[str, dict]:
    """Load cached annotations; recompute if missing."""
    from graph.schema import DISEASE_CELL_TYPE_MAP, _DISEASE_SHORT_NAMES_FOR_ANCHORS
    short_key  = _DISEASE_SHORT_NAMES_FOR_ANCHORS.get(disease_key.lower(), disease_key.upper())
    ctx        = DISEASE_CELL_TYPE_MAP.get(short_key, {})
    dataset_id = ctx.get("scperturb_dataset")
    if not dataset_id:
        return {}
    out_path = _PERTURBSEQ / dataset_id / "program_annotations.json"
    if out_path.exists():
        with open(out_path) as f:
            return json.load(f)
    return annotate_programs(disease_key)
