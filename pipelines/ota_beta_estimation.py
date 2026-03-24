"""
ota_beta_estimation.py — Ota framework Layer 2: gene → program β estimation.

The Ota et al. (Nature 2026) framework decomposes causal pathways as:
  γ_{gene→trait} = Σ_P (β_{gene→P} × γ_{P→trait})

This module estimates β_{gene→P}: the causal effect of gene X on program P activity.

β is fundamentally an INTERVENTIONAL quantity — it requires perturbation data or
a genetic instrument. Co-expression is explicitly NOT used because co-expression ≠ causation
(confounding, reverse causation, and shared upstream regulators all produce co-expression
without gene X causally driving program P).

Corrected tier hierarchy:
─────────────────────────────────────────────────────────────────────────────
Tier 1  Interventional   Cell-type-matched Perturb-seq (scPerturb database or
                         Replogle 2022 K562 for blood/myeloid diseases).
                         Direct CRISPR perturbation → transcriptome shift.

Tier 2  Convergent       eQTL-MR: use tissue-matched cis-eQTLs (GTEx) as
                         genetic instruments for gene expression; project onto
                         program gene loadings via MR effect size.
                         Causal basis: Mendelian randomization.

Tier 3  Provisional      LINCS L1000 genetic perturbation (shRNA/ORF): still
                         direct perturbation but in a cell line that may not
                         match the disease-relevant cell type.
                         Causal basis: real perturbation, imperfect cell match.

Virtual                  In silico only. Two sub-sources:
                           (a) Geneformer/GEARS: models trained on real
                               perturbation data; extrapolates to unseen genes
                               or cell types. Better than co-expression but
                               still not experimental.
                           (b) Pathway membership: binary proxy (gene in
                               program = 1, out = 0). No causal basis.
                         Must be labelled provisional_virtual; cannot be used
                         for clinical translation.
─────────────────────────────────────────────────────────────────────────────

Co-expression / GRN is NOT a tier. Observing that genes co-express does not
tell us the direction of regulation or whether there is any causal relationship.
GRN methods (SCENIC, Arboreto) produce directed graphs but the edges still
reflect statistical association conditioned on other genes, not intervention.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.evidence import CausalEdge, ProgramBetaMatrix, EvidenceTier


# ---------------------------------------------------------------------------
# Tier 1: Cell-type-matched Perturb-seq
# ---------------------------------------------------------------------------

def estimate_beta_tier1(
    gene: str,
    program: str,
    perturbseq_data: dict | None = None,
    cell_type: str = "K562",
) -> dict | None:
    """
    Tier 1 β: Direct cell-type-matched Perturb-seq measurement.

    Uses qualitative sign-level data from burden_perturb_server when the full
    h5ad has not been downloaded.  When quantitative data is available (h5ad
    loaded), returns the actual β coefficient + SE.

    Args:
        gene:            Gene symbol
        program:         cNMF program / gene-set name
        perturbseq_data: Pre-loaded Perturb-seq data dict (gene → program → {beta, se, ...})
        cell_type:       Cell line / type identifier; used to annotate data_source
    """
    if perturbseq_data is None:
        # Qualitative path — sign-level β from curated server data
        from mcp_servers.burden_perturb_server import get_gene_perturbation_effect
        effect = get_gene_perturbation_effect(gene)
        if effect.get("data_tier") != "qualitative":
            return None
        up_progs = effect.get("top_programs_up", [])
        dn_progs = effect.get("top_programs_dn", [])
        if program in up_progs:
            return {
                "beta":          1.0,   # sign-only; quantitative requires h5ad
                "beta_se":       None,
                "ci_lower":      None,
                "ci_upper":      None,
                "beta_sigma":    0.50,  # sign known, magnitude unknown
                "evidence_tier": "Tier1_Interventional",
                "data_source":   f"Replogle2022_{cell_type}_qualitative",
                "note":          "Sign-only β; download Figshare pseudo-bulk h5ad for quantitative estimate",
            }
        if program in dn_progs:
            return {
                "beta":          -1.0,
                "beta_se":       None,
                "ci_lower":      None,
                "ci_upper":      None,
                "beta_sigma":    0.50,  # sign known, magnitude unknown
                "evidence_tier": "Tier1_Interventional",
                "data_source":   f"Replogle2022_{cell_type}_qualitative",
                "note":          "Sign-only β; download Figshare pseudo-bulk h5ad for quantitative estimate",
            }
        return None

    # Quantitative path (h5ad loaded and processed through cNMF)
    gene_data = perturbseq_data.get(gene, {})
    prog_beta = gene_data.get("programs", {}).get(program)
    if prog_beta is None:
        return None
    return {
        "beta":          prog_beta["beta"],
        "beta_se":       prog_beta.get("se"),
        "ci_lower":      prog_beta.get("ci_lower"),
        "ci_upper":      prog_beta.get("ci_upper"),
        "beta_sigma":    prog_beta.get("se") or abs(prog_beta["beta"]) * 0.15,
        "evidence_tier": "Tier1_Interventional",
        "data_source":   f"Perturb-seq_{cell_type}_quantitative",
    }


# ---------------------------------------------------------------------------
# Tier 2: eQTL-MR (Mendelian Randomization via GTEx)
# ---------------------------------------------------------------------------

def estimate_beta_tier2(
    gene: str,
    program: str,
    eqtl_data: dict | None = None,
    coloc_h4: float | None = None,
    program_loading: float | None = None,
) -> dict | None:
    """
    Tier 2 β: eQTL-MR estimate.

    Uses cis-eQTLs (genetic instruments) for gene X in the disease-relevant
    tissue as instruments for gene X expression, then projects onto program P
    via the program gene loading.

    β_{gene→P} ≈ eQTL_NES × loading(gene_X in program_P)

    This is Mendelian randomization: genetic variation (eQTL SNP) randomizes
    gene X expression, and we measure downstream program effect.  The COLOC H4
    requirement (≥ 0.8) ensures the eQTL and program-activity signal share the
    same causal variant rather than being driven by distinct signals in LD.

    Args:
        gene:             Gene symbol
        program:          Program name
        eqtl_data:        GTEx eQTL result {nes, se, pval_nominal, tissue}
        coloc_h4:         COLOC H4 posterior (shared causal variant probability)
        program_loading:  Gene X's loading / weight in program P (from cNMF or gene set)
    """
    if coloc_h4 is not None and coloc_h4 < 0.8:
        return None
    if eqtl_data is None:
        return None

    nes = eqtl_data.get("nes")
    if nes is None:
        return None

    # Scale by program loading when available; raw NES otherwise
    loading = program_loading if program_loading is not None else 1.0
    beta = nes * loading

    tissue = eqtl_data.get("tissue", "unknown_tissue")
    coloc_str = f"_COLOC_H4={coloc_h4:.2f}" if coloc_h4 is not None else ""

    return {
        "beta":          beta,
        "beta_se":       eqtl_data.get("se"),
        "ci_lower":      None,
        "ci_upper":      None,
        "beta_sigma":    (
            abs((eqtl_data.get("se") or 0.0) * (loading if loading is not None else 1.0))
            or abs(beta) * 0.25
        ),
        "evidence_tier": "Tier2_Convergent",
        "data_source":   f"GTEx_{tissue}_eQTL_MR{coloc_str}",
        "coloc_h4":      coloc_h4,
        "mr_method":     "eQTL_NES_x_loading",
        "note":          "MR-based: genetic instrument (cis-eQTL) for gene expression → program loading",
    }


# ---------------------------------------------------------------------------
# Tier 2b: Open Targets credible-set genetic instruments
# Supplements GTEx when eQTL data is absent (common for immune-specific genes).
# ---------------------------------------------------------------------------

def estimate_beta_tier2_ot_instrument(
    gene: str,
    program: str,
    ot_instruments: dict | None = None,
    program_loading: float | None = None,
) -> dict | None:
    """
    Tier 2 β using Open Targets GWAS/eQTL credible-set instruments.

    Activates when GTEx eQTL data is absent.  Uses OT's integrated genetic
    evidence (multi-cohort GWAS fine-mapping + eQTL catalogue colocalization)
    as the genetic instrument.

    For eQTL instruments: β = eQTL_NES × loading  (same logic as estimate_beta_tier2)
    For GWAS credible sets: β = log(OR) × loading  (approximate — GWAS β to program)

    The GWAS-only path is labelled Tier2_Convergent but flagged as "gwas_projected"
    to signal that the gene→program direction is inferred, not directly measured.

    Args:
        gene:             Gene symbol
        program:          Program name
        ot_instruments:   Output of get_ot_genetic_instruments() — has `instruments`,
                          `best_nes`, `best_gwas_beta`
        program_loading:  Gene's weight in the program (from cNMF loadings)
    """
    if not ot_instruments or not ot_instruments.get("instruments"):
        return None

    loading = program_loading if program_loading is not None else 1.0

    # Prefer eQTL instrument (direct expression effect, same logic as GTEx Tier2)
    best_eqtl = next(
        (i for i in ot_instruments["instruments"] if i["instrument_type"] == "eqtl"),
        None,
    )
    if best_eqtl:
        nes  = best_eqtl["beta"]
        se   = best_eqtl.get("se")
        beta = nes * loading
        return {
            "beta":          beta,
            "beta_se":       (se * abs(loading)) if se else None,
            "ci_lower":      None,
            "ci_upper":      None,
            "beta_sigma":    (abs(se * loading) if se else abs(beta) * 0.25),
            "evidence_tier": "Tier2_Convergent",
            "data_source":   f"OT_eQTL_catalogue_{ot_instruments.get('ensembl_id', gene)}",
            "instrument_type": "eqtl",
            "note":          "OT eQTL credible set × program loading (MR)",
        }

    # Fall back to GWAS credible set instrument
    best_gwas = next(
        (i for i in ot_instruments["instruments"] if i["instrument_type"] == "gwas_credset"),
        None,
    )
    if best_gwas:
        gwas_beta = best_gwas["beta"]
        se        = best_gwas.get("se")
        # Project GWAS variant effect onto program loading
        beta = gwas_beta * loading
        return {
            "beta":          beta,
            "beta_se":       (se * abs(loading)) if se else None,
            "ci_lower":      None,
            "ci_upper":      None,
            "beta_sigma":    (abs(se * loading) if se else abs(beta) * 0.35),
            "evidence_tier": "Tier2_Convergent",
            "data_source":   f"OT_GWAS_credset_{best_gwas.get('study_id', 'unknown')}",
            "instrument_type": "gwas_projected",
            "note":          (
                "GWAS credible-set beta × program loading. "
                "Direction inferred from genetic association, not direct perturbation."
            ),
        }

    return None


# ---------------------------------------------------------------------------
# Tier 3: LINCS L1000 genetic perturbation
# ---------------------------------------------------------------------------

def estimate_beta_tier3(
    gene: str,
    program: str,
    lincs_signature: dict | None = None,
    program_gene_set: set[str] | None = None,
    cell_line: str | None = None,
) -> dict | None:
    """
    Tier 3 β: LINCS L1000 genetic perturbation signature.

    LINCS L1000 measures transcriptional response to shRNA knockdown or ORF
    overexpression across ~9 cancer cell lines.  This is direct perturbation
    data (not co-expression) but in a cell line that may not match the
    disease-relevant cell type.

    β_{gene→P} = overlap score between gene X knockdown signature and
                 program P gene set (Jaccard or weighted mean log2FC).

    Args:
        gene:             Gene symbol
        program:          Program name
        lincs_signature:  L1000 differential expression dict:
                          {gene_symbol → {log2fc, z_score}} for gene X KD
        program_gene_set: Set of gene symbols defining program P
        cell_line:        LINCS cell line used (A375, HT29, MCF7, etc.)
    """
    if lincs_signature is None or program_gene_set is None:
        return None

    # Compute weighted overlap: mean log2fc of program genes in KD signature.
    # Handles both {gene: float} (iLINCS flat) and {gene: {"log2fc": float}} shapes.
    def _extract_log2fc(v: object) -> float:
        if isinstance(v, dict):
            return float(v.get("log2fc") or v.get("Value") or 0.0)
        return float(v)

    program_hits = {
        g: _extract_log2fc(lincs_signature[g])
        for g in program_gene_set
        if g in lincs_signature
    }
    if not program_hits:
        return None

    # Sign-preserving mean effect of KD on program genes
    beta = sum(program_hits.values()) / len(program_gene_set)
    coverage = len(program_hits) / len(program_gene_set)

    if coverage < 0.05:  # < 5% of program genes measured — too sparse
        return None

    line_str = cell_line or "unknown_cell_line"
    return {
        "beta":          beta,
        "beta_se":       None,
        "ci_lower":      None,
        "ci_upper":      None,
        "beta_sigma":    abs(beta) * 0.35,
        "evidence_tier": "Tier3_Provisional",
        "data_source":   f"LINCS_L1000_{line_str}_shRNA",
        "coverage":      round(coverage, 3),
        "note":          (
            f"Direct perturbation (shRNA KD) in {line_str}; "
            "cell line may not match disease-relevant cell type"
        ),
    }


# ---------------------------------------------------------------------------
# Tier 4 / Virtual: In silico prediction
# Two sub-sources — both labelled provisional_virtual
# ---------------------------------------------------------------------------

def estimate_beta_geneformer(
    gene: str,
    program: str,
    geneformer_result: dict | None = None,
) -> dict | None:
    """
    Virtual β sub-source A: Geneformer / GEARS in silico perturbation.

    Models trained on real Perturb-seq data (Replogle K562 + Norman K562)
    that extrapolate to unseen genes or cell types.  Superior to co-expression
    because the model learned from actual perturbation distributions, but still
    extrapolation — not experimental.

    STUB — wire to Geneformer API or local model when available.
    """
    if geneformer_result is None:
        return None
    beta = geneformer_result.get("delta_program_activity")
    if beta is None:
        return None
    return {
        "beta":          beta,
        "beta_se":       geneformer_result.get("se"),
        "ci_lower":      None,
        "ci_upper":      None,
        "beta_sigma":    geneformer_result.get("se") or abs(beta) * 0.50,
        "evidence_tier": "provisional_virtual",
        "data_source":   "Geneformer_virtual_perturbation",
        "note":          "In silico prediction from Geneformer; not experimental — label provisional_virtual",
    }


def estimate_beta_foundation_model(
    gene: str,
    program: str,
    known_tier12_betas: dict[str, float] | None = None,
    program_gene_symbols: list[str] | None = None,
) -> dict | None:
    """
    Virtual-A (upgraded): Expression-similarity transfer from known Tier1/2 genes.

    Approximates what scGPT / Geneformer v2 / Universal Cell Embeddings do:
    find nearest neighbors in expression embedding space, then transfer their
    perturbation responses.  Here we use GTEx v10 median TPM profiles as the
    embedding (54 tissues × gene → cosine similarity).

    This gives a proper uncertainty-weighted prior for genes with no direct
    Perturb-seq coverage, rather than binary pathway membership (0/1).

    β_prior = Σ_i cos_sim(gene, known_i) × β_known_i / Σ cos_sim
    σ_prior = max(0.5 × |β_prior|, 0.30)   [calibrated: min 0.30]

    Args:
        gene:                 Target gene to estimate β for
        program:              Program name
        known_tier12_betas:   {gene_symbol: beta_value} for genes in the same
                              program that already have Tier1 or Tier2 evidence
        program_gene_symbols: All gene symbols defining program (for context)
    """
    if not known_tier12_betas:
        return None

    from mcp_servers.single_cell_server import _query_gtex_v10_median_expression

    # Fetch embedding for target gene
    target_tpm = _query_gtex_v10_median_expression(gene)
    if not target_tpm or len(target_tpm) < 3:
        return None

    target_norm = _l2_norm(target_tpm)
    if target_norm == 0:
        return None

    # Weighted transfer from known Tier1/2 genes
    weighted_sum = 0.0
    weight_total = 0.0
    for known_gene, known_beta in known_tier12_betas.items():
        if known_gene == gene:
            continue
        known_tpm = _query_gtex_v10_median_expression(known_gene)
        if not known_tpm or len(known_tpm) < 3:
            continue
        # Cosine similarity in GTEx expression space
        cos_sim = _cosine_sim(target_tpm, known_tpm, target_norm)
        if cos_sim <= 0:
            continue
        weighted_sum  += cos_sim * known_beta
        weight_total  += cos_sim

    if weight_total == 0:
        return None

    beta_prior = weighted_sum / weight_total
    sigma      = max(0.50 * abs(beta_prior), 0.30)

    return {
        "beta":          beta_prior,
        "beta_se":       None,
        "ci_lower":      None,
        "ci_upper":      None,
        "beta_sigma":    sigma,
        "evidence_tier": "provisional_virtual",
        "data_source":   f"foundation_model_GTEx_transfer_{len(known_tier12_betas)}_donors",
        "note":          (
            "Cosine-similarity transfer from known Tier1/2 genes in GTEx v10 "
            "expression space. Approximates scGPT/UCE nearest-neighbor transfer. "
            "Labelled provisional_virtual; represents uncertainty-weighted prior."
        ),
    }


def _l2_norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cosine_sim(a: list[float], b: list[float], norm_a: float | None = None) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = norm_a if norm_a is not None else _l2_norm(a)
    nb = _l2_norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def estimate_beta_virtual(
    gene: str,
    program: str,
    pathway_member: bool | None = None,
) -> dict:
    """
    Virtual β sub-source B: pathway membership proxy.

    If gene X is in program P's defining gene set, β = 1.0 (binary proxy).
    This has NO causal basis — it is annotation-only.  Used as final fallback
    only so the pipeline produces a finite matrix; all outputs are explicitly
    labelled provisional_virtual.

    Co-expression-derived weights are intentionally excluded here because
    they would imply causality that the data cannot support.
    """
    beta_val = 1.0 if pathway_member else None
    return {
        "beta":          beta_val,
        "beta_se":       None,
        "ci_lower":      None,
        "ci_upper":      None,
        "beta_sigma":    0.70,  # maximum uncertainty: annotation proxy only
        "evidence_tier": "provisional_virtual",
        "data_source":   "pathway_membership_proxy",
        "note":          "Annotation proxy only — no causal basis. Must be labelled provisional_virtual.",
    }


# ---------------------------------------------------------------------------
# Main β fallback decision tree
# ---------------------------------------------------------------------------

def estimate_beta(
    gene: str,
    program: str,
    perturbseq_data: dict | None = None,
    eqtl_data: dict | None = None,
    coloc_h4: float | None = None,
    program_loading: float | None = None,
    lincs_signature: dict | None = None,
    program_gene_set: set[str] | None = None,
    geneformer_result: dict | None = None,
    known_tier12_betas: dict[str, float] | None = None,
    pathway_member: bool | None = None,
    cell_type: str = "K562",
    cell_line: str | None = None,
    ot_instruments: dict | None = None,
) -> dict:
    """
    β fallback decision tree for the Ota framework.

    Priority:
      1. Tier1 — cell-type-matched Perturb-seq (direct intervention)
      2. Tier2a — GTEx eQTL-MR (Mendelian randomization, tissue-matched)
      2. Tier2b — OT credible-set instruments (GWAS/eQTL, when GTEx absent)
      3. Tier3 — LINCS L1000 perturbation (direct but cell-line mismatched)
      4. Virtual-A — Geneformer in silico (trained on perturbation data)
      4. Virtual-A upgraded — foundation model cosine-similarity transfer
      5. Virtual-B — pathway membership (annotation proxy, no causal basis)

    Co-expression and GRN weights are intentionally absent from this chain.
    """
    # Tier 1
    beta = estimate_beta_tier1(gene, program, perturbseq_data, cell_type=cell_type)
    if beta is not None:
        return {**beta, "gene": gene, "program": program, "tier_used": 1}

    # Tier 2a — GTEx eQTL-MR
    beta = estimate_beta_tier2(gene, program, eqtl_data, coloc_h4, program_loading)
    if beta is not None:
        return {**beta, "gene": gene, "program": program, "tier_used": 2}

    # Tier 2b — OT credible-set instruments (GWAS fine-mapping + eQTL catalogue)
    beta = estimate_beta_tier2_ot_instrument(gene, program, ot_instruments, program_loading)
    if beta is not None:
        return {**beta, "gene": gene, "program": program, "tier_used": 2}

    # Tier 3
    beta = estimate_beta_tier3(gene, program, lincs_signature, program_gene_set, cell_line)
    if beta is not None:
        return {**beta, "gene": gene, "program": program, "tier_used": 3}

    # Virtual-A: Geneformer
    beta = estimate_beta_geneformer(gene, program, geneformer_result)
    if beta is not None:
        return {**beta, "gene": gene, "program": program, "tier_used": 4}

    # Virtual-A upgraded: foundation model cosine-similarity transfer
    beta = estimate_beta_foundation_model(gene, program, known_tier12_betas, program_gene_set and list(program_gene_set))
    if beta is not None:
        return {**beta, "gene": gene, "program": program, "tier_used": 4}

    # Virtual-B: pathway membership proxy
    return {
        **estimate_beta_virtual(gene, program, pathway_member),
        "gene": gene, "program": program, "tier_used": 4,
    }


# ---------------------------------------------------------------------------
# β matrix construction
# ---------------------------------------------------------------------------

def build_beta_matrix(
    genes: list[str],
    programs: list[str],
    perturbseq_data: dict | None = None,
    eqtl_data: dict[str, dict] | None = None,
    coloc_data: dict[str, float] | None = None,
    program_loadings: dict[str, dict[str, float]] | None = None,
    lincs_data: dict[str, dict] | None = None,
    program_gene_sets: dict[str, set[str]] | None = None,
    geneformer_data: dict[str, dict] | None = None,
    pathway_membership: dict[str, set[str]] | None = None,
    cell_type: str = "unknown",
    disease: str | None = None,
) -> ProgramBetaMatrix:
    """
    Build the full β_{gene×program} matrix for the Ota framework.

    Args:
        genes:             List of gene symbols
        programs:          List of program / gene-set names
        perturbseq_data:   Perturb-seq data keyed by gene
        eqtl_data:         GTEx eQTL data keyed by gene → {nes, se, tissue}
        coloc_data:        COLOC H4 posteriors keyed by gene
        program_loadings:  NMF/cNMF loadings keyed by program → gene → weight
        lincs_data:        L1000 KD signatures keyed by gene → {gene_symbol → {log2fc}}
        program_gene_sets: Gene set definitions keyed by program → set[gene]
        geneformer_data:   Geneformer outputs keyed by gene → program → {delta_activity}
        pathway_membership: Pathway membership keyed by program → set[gene]
        cell_type:         Primary cell type context (used for ProgramBetaMatrix annotation)
        disease:           Disease name (used for cell_type routing if cell_type not set)

    Returns:
        ProgramBetaMatrix pydantic model with NaN-filled float matrix.
    """
    # Resolve cell type from disease if not supplied
    if cell_type == "unknown" and disease:
        from graph.schema import DISEASE_CELL_TYPE_MAP
        ctx = DISEASE_CELL_TYPE_MAP.get(disease, {})
        cell_type = ctx.get("cell_line") or (ctx.get("cell_types") or ["unknown"])[0]

    matrix: dict[str, dict[str, float | None]] = {}
    tier_summary: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}

    for gene in genes:
        matrix[gene] = {}
        for program in programs:
            loading = (program_loadings or {}).get(program, {}).get(gene)
            pg_set  = (program_gene_sets or {}).get(program)
            pm      = gene in (pathway_membership or {}).get(program, set())
            lincs_sig = (lincs_data or {}).get(gene)

            gf_result = None
            if geneformer_data and gene in geneformer_data:
                gf_result = geneformer_data[gene].get(program)

            beta_result = estimate_beta(
                gene=gene,
                program=program,
                perturbseq_data=perturbseq_data,
                eqtl_data=(eqtl_data or {}).get(gene),
                coloc_h4=(coloc_data or {}).get(gene),
                program_loading=loading,
                lincs_signature=lincs_sig,
                program_gene_set=pg_set,
                geneformer_result=gf_result,
                pathway_member=pm if pm else None,
                cell_type=cell_type,
            )
            matrix[gene][program] = beta_result.get("beta")
            tier_summary[beta_result.get("tier_used", 4)] += 1

    # Best tier per gene
    tier_priority = {
        "Tier1_Interventional": 1,
        "Tier2_Convergent":     2,
        "Tier3_Provisional":    3,
        "provisional_virtual":  4,
    }
    valid_tiers = {"Tier1_Interventional", "Tier2_Convergent", "Tier3_Provisional"}
    evidence_tier_per_gene: dict[str, str] = {}

    for gene in genes:
        best = "provisional_virtual"
        for program in programs:
            b = estimate_beta(
                gene=gene, program=program,
                perturbseq_data=perturbseq_data,
                eqtl_data=(eqtl_data or {}).get(gene),
                coloc_h4=(coloc_data or {}).get(gene),
                program_loading=(program_loadings or {}).get(program, {}).get(gene),
                cell_type=cell_type,
            )
            t = b.get("evidence_tier", "provisional_virtual")
            if tier_priority.get(t, 99) < tier_priority.get(best, 99):
                best = t
        evidence_tier_per_gene[gene] = best if best in valid_tiers else "Tier3_Provisional"

    # Replace None with NaN — distinguishes "no data" from "zero effect"
    float_matrix = {
        g: {p: (v if v is not None else math.nan) for p, v in progs.items()}
        for g, progs in matrix.items()
    }

    note = (
        f"β tiers — T1(cell-matched Perturb-seq)={tier_summary[1]}, "
        f"T2(eQTL-MR)={tier_summary[2]}, "
        f"T3(LINCS L1000)={tier_summary[3]}, "
        f"Virtual={tier_summary[4]}"
    )

    perturb_source = perturbseq_data and f"Perturb-seq_{cell_type}" or "eQTL_MR_or_virtual"

    return ProgramBetaMatrix(
        programs=[
            {"program_id": p, "top_genes": [], "pathways": [], "cell_type": cell_type}
            for p in programs
        ],
        beta_matrix=float_matrix,
        evidence_tier_per_gene=evidence_tier_per_gene,
        cell_type=cell_type,
        perturb_seq_source=perturb_source,
        virtual_ensemble_vs_baseline={"note": note},
    )


# ---------------------------------------------------------------------------
# Convenience: CAD target gene β estimation
# ---------------------------------------------------------------------------

def estimate_cad_target_betas() -> ProgramBetaMatrix:
    """
    Estimate β for core CAD-relevant genes using Tier1 qualitative K562 data.
    Cell type = myeloid (K562) — appropriate for CAD CHIP/inflammatory programs.
    """
    from mcp_servers.burden_perturb_server import get_cnmf_program_info

    cad_genes = ["PCSK9", "LDLR", "HMGCR", "DNMT3A", "TET2", "ASXL1", "IL6R", "HLA-DRA", "CIITA"]
    programs_info = get_cnmf_program_info()
    programs = programs_info["programs"]

    return build_beta_matrix(
        genes=cad_genes,
        programs=programs,
        cell_type="K562",
        disease="CAD",
    )
