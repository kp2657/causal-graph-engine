"""
mcp_servers/ukb_wes_server.py — UKB WES rare variant burden & gene constraint.

Provides access to:
  1. UK Biobank Whole Exome Sequencing (WES) gene-level burden stats
       - Collapsing rare variant (MAF < 0.001) LoF + damaging missense burden
       - Accessed via Open Targets Genetics gene-burden endpoint
  2. gnomAD v4 gene constraint metrics
       - LOEUF (Loss-of-Function Observed/Expected Upper bound Fraction)
       - pLI (probability of LoF intolerance)
       - Missense constraint (z-score)
  3. gnomAD v4 variant-level rare coding variants
       - Clinvar pathogenic / likely pathogenic
       - gnomAD LoF/damaging missense with carrier frequency in UKB

Burden test interpretation:
  - A significant rare-variant burden association (p < 1e-6) for a disease
    means LoF/damaging carriers have higher/lower disease risk.
  - Direction: LoF typically → loss of gene function → effect on trait
    NEGATIVE beta (in LoF direction) → gene activation reduces risk
    POSITIVE beta → gene activation increases risk (gain-of-function implication)
  - This provides a β_{gene→program} direction constraint even when cis-eQTL
    is absent (e.g. for HMCN1, FBN2, complement genes with coding variants).

Public API
----------
    get_gene_burden(gene, disease)
        -> dict  {gene, disease, burden_beta, burden_se, burden_p, n_variants,
                  loeuf, pli, missense_z, data_source}

    get_gnomad_constraint(gene)
        -> dict  {gene, loeuf, pli, missense_z, oe_lof, n_exp_lof, data_source}

    get_rare_coding_variants(gene, max_af)
        -> dict  {gene, variants: [{rsid, consequence, af, am_class, ...}]}
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

from pipelines.api_cache import api_cached

try:
    import fastmcp
    mcp = fastmcp.FastMCP("ukb-wes-server")
    _tool = mcp.tool()
except ImportError:
    def _tool(fn=None, **_):
        return fn if fn is not None else (lambda f: f)
    mcp = None

# ---------------------------------------------------------------------------
# API constants
# ---------------------------------------------------------------------------

GNOMAD_API       = "https://gnomad.broadinstitute.org/api"
OT_PLATFORM_GQL  = "https://api.platform.opentargets.org/api/v4/graphql"
OT_GENETICS_GQL  = "https://api.genetics.opentargets.org/graphql"  # OTG v4

_DELAY = 0.3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gnomad_gql(query: str, variables: dict | None = None) -> dict:
    try:
        resp = httpx.post(
            GNOMAD_API,
            json={"query": query, "variables": variables or {}},
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0),
        )
        resp.raise_for_status()
        result = resp.json()
        return result.get("data", {})
    except Exception:
        return {}


def _ot_gql(query: str, variables: dict | None = None, endpoint: str = OT_PLATFORM_GQL) -> dict:
    try:
        resp = httpx.post(
            endpoint,
            json={"query": query, "variables": variables or {}},
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0),
        )
        resp.raise_for_status()
        result = resp.json()
        if "errors" in result:
            return {"error": result["errors"][0].get("message", "")}
        return result.get("data", {})
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# gnomAD constraint — real implementation
# ---------------------------------------------------------------------------

_GNOMAD_CONSTRAINT_QUERY = """
query GeneConstraint($geneSymbol: String!) {
  gene(gene_symbol: $geneSymbol, reference_genome: GRCh38) {
    gene_id
    symbol
    gnomad_constraint {
      exp_lof
      obs_lof
      lof_z
      oe_lof
      oe_lof_upper
      pLI
      exp_mis
      obs_mis
      oe_mis
      mis_z
    }
  }
}
"""


@_tool
@api_cached(ttl_days=30)
def get_gnomad_constraint(gene: str) -> dict:
    """
    Retrieve gnomAD v2.1 gene constraint metrics (pLI, LOEUF, missense z).

    Constraint metrics reflect evolutionary tolerance to functional variation:
      - pLI ≥ 0.9: extremely intolerant to LoF — essential gene
      - LOEUF < 0.35: strong LoF constraint (upper bound of 90% CI on oe_lof)
      - mis_z > 3.09: strong missense constraint

    Used as a prior for interpreting rare variant burden test direction:
    a constrained gene (pLI ≥ 0.9) with increased LoF burden in disease
    → excess of LoF → heterozygous loss is sufficient for disease risk.

    Args:
        gene: Gene symbol (e.g. "CFH", "HMCN1", "FBN2")

    Returns:
        {gene, pli, loeuf, oe_lof, missense_z, n_exp_lof, n_obs_lof, data_source}
    """
    # Static-data fast path: if the gnomAD v4.1 constraint TSV is on disk,
    # skip the GraphQL call entirely.  Falls back to live API below when the
    # file is missing or the gene isn't in the table.
    try:
        from pipelines.static_lookups import get_lookups
        _local = get_lookups().get_gnomad_constraint(gene)
        if _local is not None:
            return _local
    except Exception:
        pass   # static layer is best-effort; never blocks live lookup

    time.sleep(_DELAY)
    data = _gnomad_gql(_GNOMAD_CONSTRAINT_QUERY, {"geneSymbol": gene})
    gene_data = data.get("gene") or {}
    constraint = gene_data.get("gnomad_constraint") or {}

    pli    = constraint.get("pLI")
    oe_lof = constraint.get("oe_lof")
    loeuf  = constraint.get("oe_lof_upper")   # upper bound of CI = LOEUF
    mis_z  = constraint.get("mis_z")
    exp_lof = constraint.get("exp_lof")
    obs_lof = constraint.get("obs_lof")

    return {
        "gene":        gene,
        "gene_id":     gene_data.get("gene_id"),
        "pli":         round(pli, 4) if pli is not None else None,
        "loeuf":       round(loeuf, 4) if loeuf is not None else None,
        "oe_lof":      round(oe_lof, 4) if oe_lof is not None else None,
        "missense_z":  round(mis_z, 3) if mis_z is not None else None,
        "n_exp_lof":   round(exp_lof, 2) if exp_lof is not None else None,
        "n_obs_lof":   obs_lof,
        "lof_z":       constraint.get("lof_z"),
        "data_source": "gnomAD v2.1 gene constraint (GRCh38)",
    }


# ---------------------------------------------------------------------------
# gnomAD rare coding variants
# ---------------------------------------------------------------------------

_GNOMAD_VARIANTS_QUERY = """
query RareCodingVariants($geneSymbol: String!, $datasetId: DatasetId!) {
  gene(gene_symbol: $geneSymbol, reference_genome: GRCh38) {
    variants(dataset: $datasetId) {
      variant_id
      rsid
      consequence
      lof
      lof_filter
      af { exome { af } genome { af } }
      flags
    }
  }
}
"""


@_tool
@api_cached(ttl_days=30)
def get_rare_coding_variants(
    gene: str,
    max_af: float = 0.001,
    dataset: str = "gnomad_r4",
) -> dict:
    """
    Retrieve rare coding variants (MAF < max_af) for a gene from gnomAD v4.

    Useful for identifying LoF and damaging missense variants that drive
    UKB WES burden associations. These are the actual variants collapsing
    tests accumulate across.

    Args:
        gene:    Gene symbol
        max_af:  Maximum allele frequency (default 0.001 = 0.1%)
        dataset: gnomAD dataset ID ("gnomad_r4" for v4, "gnomad_r2_1" for v2.1)

    Returns:
        {gene, n_rare_lof, n_rare_missense, variants: list[dict], data_source}
    """
    time.sleep(_DELAY)
    data = _gnomad_gql(_GNOMAD_VARIANTS_QUERY, {"geneSymbol": gene, "datasetId": dataset})
    gene_data = data.get("gene") or {}
    raw_variants = gene_data.get("variants") or []

    rare_lof = 0
    rare_mis = 0
    parsed: list[dict] = []

    for v in raw_variants:
        # Allele frequency — prefer exome, fall back to genome
        af_data = v.get("af") or {}
        af = (af_data.get("exome") or {}).get("af") or (af_data.get("genome") or {}).get("af") or 0.0
        if float(af) > max_af:
            continue

        csq = v.get("consequence", "")
        lof = v.get("lof", "")  # "HC" = high-confidence LoF
        is_lof = lof == "HC"
        is_mis = "missense" in csq.lower()

        if is_lof:
            rare_lof += 1
        elif is_mis:
            rare_mis += 1
        else:
            continue  # skip synonymous, intronic, etc.

        parsed.append({
            "variant_id":  v.get("variant_id"),
            "rsid":        v.get("rsid"),
            "consequence": csq,
            "lof":         lof,
            "af":          round(float(af), 8),
            "flags":       v.get("flags", []),
        })

    return {
        "gene":            gene,
        "n_rare_lof":      rare_lof,
        "n_rare_missense": rare_mis,
        "n_total_rare":    len(parsed),
        "variants":        parsed[:50],  # cap at 50 for context size
        "max_af":          max_af,
        "data_source":     f"gnomAD {dataset} rare coding variants (AF < {max_af})",
    }


# ---------------------------------------------------------------------------
# UKB WES gene burden — via Open Targets
# ---------------------------------------------------------------------------

_OT_GENE_BURDEN_QUERY = """
query GeneBurden($ensemblId: String!, $studyId: String!) {
  study(studyId: $studyId) {
    credibleSets {
      locus {
        beta
        standardError
        pValueMantissa
        pValueExponent
        variant {
          id
          rsIds
          consequence {
            label
          }
        }
      }
    }
  }
}
"""

# OT genetics study IDs for UKB WES gene-burden analyses
# These are collapsing burden test GWAS in OT genetics
_UKB_WES_BURDEN_STUDY_PREFIX = "UKBIOBANK_WES_"

# Disease → OT genetics study IDs for UKB WES burden tests
# These are approximate — OT uses EFO trait IDs as study handles
_DISEASE_BURDEN_EFO: dict[str, str] = {
    "CAD":  "EFO_0001645",   # coronary artery disease
    "SLE":  "EFO_0002690",   # systemic lupus erythematosus
    "RA":   "EFO_0000685",   # rheumatoid arthritis
}


_OT_GENE_L2G_QUERY = """
query GeneL2G($geneId: String!, $studyId: String!) {
  studyLocus(studyId: $studyId, locusId: $geneId) {
    l2GPredictions(orderByScore: true) {
      target {
        id
        approvedSymbol
      }
      score
      features {
        name
        value
      }
    }
  }
}
"""

# Simplified fallback: get UKB WES evidence from OT Platform gene page
_OT_GENE_WES_QUERY = """
query GeneWES($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    geneticConstraint {
      constraintType
      exp
      obs
      oeUpper
      score
    }
    safetyLiabilities {
      event
      effects { direction dosing }
    }
  }
}
"""

# gnomAD pLI/LOEUF query already done in get_gnomad_constraint.
# For UKB WES burden, we query OT Platform for gene-disease associations
# with rare variant evidence type.

_OT_DISEASE_GENE_EVIDENCE_QUERY = """
query DiseaseGeneEvidence($ensemblId: String!, $efoId: String!) {
  target(ensemblId: $ensemblId) {
    approvedSymbol
    associatedDiseases(page: {index: 0, size: 100}) {
      rows {
        disease { id }
        score
        datasourceScores { id score }
        evidences(
          efoIds: [$efoId]
          datasourceIds: ["ot_genetics_portal"]
          page: {index: 0, size: 10}
        ) {
          rows {
            score
            variantId
            studyId
            beta
            betaConfidenceIntervalLower
            betaConfidenceIntervalUpper
            pValueMantissa
            pValueExponent
          }
        }
      }
    }
  }
}
"""


@_tool
@api_cached(ttl_days=30)
def get_gene_burden(
    gene: str,
    disease: str | None = None,
    ensembl_id: str | None = None,
) -> dict:
    """
    Retrieve UKB WES gene-level rare variant burden statistics.

    Queries the Open Targets Platform for rare variant evidence (primarily
    from gene collapsing tests in UKB WES) and gnomAD constraint as a proxy
    for expected burden directionality.

    Burden test interpretation:
      - Positive beta (burden): more carriers → higher disease risk
        → inhibiting the gene (reducing function) may be therapeutic
      - Negative beta (burden): more carriers → lower disease risk
        → the gene is protective; LoF = disease risk factor
        → inhibiting the gene would be harmful; activating it is therapeutic

    Note: UKB WES collapsing tests are primarily in Open Targets genetics
    portal. Direct access to full burden summary stats requires data access
    agreement. This function uses the publicly available OT Platform associations
    filtered to rare variant evidence, supplemented by gnomAD constraint.

    Args:
        gene:         Gene symbol (e.g. "HMCN1", "FBN2", "CFH")
        disease:      Disease key (e.g. "AMD", "CAD"); used for EFO lookup
        ensembl_id:   Ensembl gene ID (looked up if not provided)

    Returns:
        {
            "gene":          str,
            "disease":       str | None,
            "burden_p":      float | None,   # best rare variant p-value in OT
            "burden_beta":   float | None,   # effect direction (pos = risk, neg = protective)
            "burden_se":     float | None,
            "n_rare_lof":    int | None,     # from gnomAD rare variant count
            "loeuf":         float | None,
            "pli":           float | None,
            "missense_z":    float | None,
            "interpretation": str,           # human-readable summary
            "data_source":   str,
        }
    """
    time.sleep(_DELAY)

    # 1. Get gnomAD constraint (always available)
    constraint = get_gnomad_constraint(gene)
    loeuf     = constraint.get("loeuf")
    pli       = constraint.get("pli")
    mis_z     = constraint.get("missense_z")
    gene_id   = ensembl_id or constraint.get("gene_id")

    # 2. Attempt OT Platform rare variant evidence query
    burden_p    = None
    burden_beta = None
    burden_se   = None
    burden_study = None

    if gene_id and disease:
        efo_id = _DISEASE_BURDEN_EFO.get(disease)
        if efo_id:
            try:
                ot_data = _ot_gql(_OT_DISEASE_GENE_EVIDENCE_QUERY, {
                    "ensemblId": gene_id,
                    "efoId":     efo_id,
                }, endpoint=OT_PLATFORM_GQL)

                target_data = ot_data.get("target") or {}
                rows = target_data.get("associatedDiseases", {}).get("rows", [])
                
                for row in rows:
                    if row.get("disease", {}).get("id") == efo_id:
                        evidences = row.get("evidences", {}).get("rows", [])
                        # Find rare variant evidence (lowest p-value)
                        for ev in evidences:
                            p_mant = ev.get("pValueMantissa")
                            p_exp  = ev.get("pValueExponent")
                            beta   = ev.get("beta")
                            if p_mant is not None and p_exp is not None:
                                p = float(p_mant) * (10 ** int(p_exp))
                                if burden_p is None or p < burden_p:
                                    burden_p    = p
                                    burden_beta = float(beta) if beta is not None else None
                                    ci_lo = ev.get("betaConfidenceIntervalLower")
                                    ci_hi = ev.get("betaConfidenceIntervalUpper")
                                    if ci_lo is not None and ci_hi is not None:
                                        burden_se = (float(ci_hi) - float(ci_lo)) / (2 * 1.96)
                                    burden_study = ev.get("studyId")
                        break # found the disease
            except Exception:
                pass

    # 3. Interpret constraint + burden
    interp_parts = []
    if loeuf is not None:
        if loeuf < 0.35:
            interp_parts.append(f"Strongly constrained (LOEUF={loeuf:.3f}): LoF likely pathogenic")
        elif loeuf < 0.6:
            interp_parts.append(f"Moderately constrained (LOEUF={loeuf:.3f})")
        else:
            interp_parts.append(f"Tolerant to LoF (LOEUF={loeuf:.3f})")
    if pli is not None and pli >= 0.9:
        interp_parts.append(f"pLI={pli:.3f}: LoF intolerant")
    if burden_p is not None and burden_p < 1e-4:
        direction = "increases" if (burden_beta or 0) > 0 else "decreases"
        interp_parts.append(
            f"Rare variant burden p={burden_p:.2e}: LoF/damaging variants {direction} disease risk"
        )
    if not interp_parts:
        interp_parts.append("No significant burden signal; constraint metrics only")

    return {
        "gene":           gene,
        "gene_id":        gene_id,
        "disease":        disease,
        "burden_p":       burden_p,
        "burden_beta":    burden_beta,
        "burden_se":      burden_se,
        "burden_study":   burden_study,
        "loeuf":          loeuf,
        "pli":            pli,
        "missense_z":     mis_z,
        "n_exp_lof":      constraint.get("n_exp_lof"),
        "n_obs_lof":      constraint.get("n_obs_lof"),
        "interpretation": "; ".join(interp_parts),
        "data_source":    "OT Platform (rare variant evidence) + gnomAD v2.1 constraint",
    }


# ---------------------------------------------------------------------------
# Convenience helper for beta estimation pipeline
# ---------------------------------------------------------------------------

@api_cached(ttl_days=30)
def get_burden_direction_for_gene(gene: str, disease: str | None = None) -> dict | None:
    """
    Get the rare-variant burden directionality for use in β estimation.

    Returns dict with 'burden_beta' (sign = direction), 'burden_p', 'loeuf', 'pli'
    or None if no useful data available.
    """
    try:
        result = get_gene_burden(gene, disease=disease)
        # Return something useful even without burden_p (constraint alone is informative)
        if result.get("loeuf") is not None or result.get("burden_p") is not None:
            return result
    except Exception:
        pass
    return None


if __name__ == "__main__":
    print("Testing UKB WES server...")
    r = get_gnomad_constraint("CFH")
    print(f"CFH constraint: pLI={r['pli']}, LOEUF={r['loeuf']}")
    r2 = get_gene_burden("TREM2", disease="AD")
    print(f"HMCN1 burden: p={r2['burden_p']}, interpretation={r2['interpretation'][:80]}")
