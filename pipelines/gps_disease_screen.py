"""
gps_disease_screen.py — GPS disease-state and NMF-program reversal screens.

Two complementary GPS screens that go beyond per-target emulation (gps_screen.py):

1. DISEASE-STATE REVERSAL
   Input:  whole AMD disease transcriptional signature (disease_log2fc per gene,
           from CELLxGENE AMD-vs-healthy RPE differential expression).
   GPS finds compounds that REVERSE the AMD transcriptional state → healthy.
   Phenotypic, target-agnostic. Equivalent to asking: "what has a similar mechanism
   to moving the cell from the AMD attractor back to the healthy attractor?"

2. NMF PROGRAM REVERSAL
   Input:  gene loading vectors of the top AMD-causal programs (highest |γ_{P→AMD}|),
           signed by causal direction.
   GPS finds compounds that REVERSE each causal transcriptional program individually.
   Mechanistic: this maps the OTA formula directly into chemical space.

   OTA formula link:
     γ_{gene→AMD} = Σ_P (β_{gene→P} × γ_{P→AMD})
   Programs with high |γ_{P→AMD}| are the causal transcriptional axes.
   The program loading vector encodes the genes that define that axis.
   GPS reversal of the program signature identifies compounds that pharmacologically
   suppress (risk programs) or restore (protective programs) the causal axis.

   Signature sources (in priority order):
   a. NMF gene loadings from cNMF h5ad analysis (transcriptomic, log2FC-like).
      These are the correct GPS inputs — RGES similarity is defined in expression space.
   b. OTA contribution fractions + MSigDB Hallmark gene expansion (fallback).
      Used when cNMF data is unavailable or GPS 2198 overlap is too sparse (<5 genes).
      OTA fractions approximate the "activity" of each target in that program axis.

   Direction logic:
     - Risk program  (net contribution > 0 across top targets):
         pass positive loadings → GPS reversers inhibit the risk program → therapeutic
     - Protective program (net contribution < 0):
         pass negative loadings → GPS reversers reinforce the protective program
         (i.e. reversers of the negated protective signature = activators) → therapeutic

3. TARGET EMULATION  (in gps_screen.py, not this file)
   Input:  Perturb-seq KO signature for a specific gene (negated: KO mimics = reversers of -KO).
   GPS finds compounds that mimic therapeutic knockdown of that gene.
   Per-target, mechanistic at single-gene resolution.

Compound overlap between disease-reversal and program-reversal lists provides the
strongest evidence: those compounds act on the specific causal mechanism and also
correct the downstream disease phenotype.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any

log = logging.getLogger(__name__)

# GPS requires enough differentially expressed genes for RGES to have statistical power.
# Below ~50 genes the normalized expression rank vector has insufficient dynamic range
# and RGES approaches 0 for all compounds. OTA proxy signatures (~88 genes) satisfy this;
# h5ad DEG signatures (~700-1000 genes) are preferred.
_MIN_DISEASE_SIG_GENES = 50

# Minimum intersection of an NMF program's gene loadings with the GPS 2198-gene landmark
# set (Xing et al., Cell 2026). Programs with fewer matching genes produce RGES estimates
# dominated by noise from the sparse ranked list.
_MIN_PROGRAM_SIG_GENES = 5

# GPS sets n_permutations = n_sig_genes internally (Xing et al., Cell 2026, Methods).
# ≥700 permutations ensures the Z_RGES null distribution approximates N(0,1) via CLT
# (confirmed empirically: 183 genes → 183 perms → 0 hits; ~1000 genes → well-calibrated).
# 1000 cap keeps BGRD computation under 4500s, within the 7200s timeout.
_GPS_BGRD_MIN_PERMS = 700
_GPS_BGRD_MAX_PERMS = 1000

# Programs are screened only if their net causal weight is ≥10% of the strongest program's
# weight. This is a relative threshold: it adapts to the scale of causal signal in each
# disease rather than imposing an absolute cutoff that could either screen noise (if too low)
# or miss meaningful programs (if too high for a data-sparse disease).
_PROGRAM_WEIGHT_FRACTION = 0.10


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_gps_disease_screens(
    targets: list[dict],
    disease_query: dict,
    gamma_estimates: dict | None = None,   # {prog → {trait → gamma_dict}}; optional
    top_n_programs: int = 5,
    top_n_compounds: int = 20,
    z_threshold: float = 2.0,
    max_hits: int = 100,
    library: str = "HTS",
) -> dict:
    """
    Run GPS at disease-state and NMF-program level.

    Args:
        targets:         List of target dicts from target_prioritization_result
        disease_query:   Disease context dict
        gamma_estimates: Optional program→trait gamma matrix (adds direct γ_P→AMD)
        top_n_programs:  Max number of NMF programs to screen
        top_n_compounds: Fallback top-N when Z_RGES threshold is not applicable
        z_threshold:     |Z_RGES| cutoff for hit selection. With ≥700 permutations,
                         Z_RGES ~ N(0,1) under the null (Xing et al., Cell 2026);
                         |Z| > 2.0 gives two-tailed p < 0.05 ≈ FDR 5%. Falls back
                         to top_n_compounds if fewer than 3 compounds pass this threshold.
        max_hits:        Hard cap on returned hits regardless of threshold
        library:         GPS compound library ("HTS" or "ZINC")

    Returns:
        {
            "disease_reversers":  list[compound_dict],        # whole-disease GPS
            "program_reversers":  {prog_id: list[compound]},  # per-program GPS
            "disease_sig_n_genes": int,
            "programs_screened":  list[{id, direction, weight, n_genes}],
            "warnings":           list[str],
        }
    """
    from pipelines.gps_screen import _docker_available, screen_disease_for_reversers

    warnings: list[str] = []
    disease_reversers: list[dict] = []
    program_reversers: dict[str, list[dict]] = {}
    programs_screened: list[dict] = []

    if not _docker_available():
        warnings.append("GPS disease screen skipped: Docker / GPS image unavailable")
        return _empty(warnings)

    disease_name = disease_query.get("disease_name", "disease")
    disease_key  = disease_query.get("disease_key") or disease_name.lower().replace(" ", "_")[:12]

    # ------------------------------------------------------------------
    # 1. Build disease-level transcriptional signature
    # ------------------------------------------------------------------
    disease_sig = _build_disease_signature(targets, disease_query=disease_query)
    disease_sig_n = len(disease_sig)

    if disease_sig_n >= _MIN_DISEASE_SIG_GENES:
        log.info("GPS disease reversal: %d-gene signature for %s", disease_sig_n, disease_name)
        print(f"[GPS] disease_state screen: {disease_sig_n} genes | {disease_name}", flush=True)  # progress
        disease_reversers = screen_disease_for_reversers(
            disease_sig,
            label=f"{disease_key}_disease_state",
            library=library,
            top_n=top_n_compounds,
            z_threshold=z_threshold,
            max_hits=max_hits,
        )
        for hit in disease_reversers:
            hit["screen_type"] = "disease_state_reversal"
        log.info("GPS disease_state done: %d reversers", len(disease_reversers))
        print(f"[GPS] disease_state done: {len(disease_reversers)} reversers", flush=True)  # progress
        if disease_reversers:
            log.info("GPS: annotating %d disease-state reversers via PubChem + ChEMBL", len(disease_reversers))
            disease_reversers = annotate_gps_compounds(disease_reversers)
    else:
        warnings.append(
            f"Disease signature too sparse ({disease_sig_n} genes < {_MIN_DISEASE_SIG_GENES}); "
            "skipping whole-disease GPS. Need CELLxGENE AMD h5ad for disease_log2fc."
        )

    # ------------------------------------------------------------------
    # 2. Identify top AMD-causal programs and build program signatures
    # ------------------------------------------------------------------
    program_directions = _aggregate_program_directions(targets, gamma_estimates, disease_query)

    # Screen programs with weight ≥ _PROGRAM_WEIGHT_FRACTION × max program weight.
    # Relative threshold: adapts to disease signal scale rather than imposing an
    # absolute cutoff. Programs below 10% of the top program's weight are too weak
    # to produce interpretable GPS reversal hits.
    all_weights = [w for _, (_, w) in program_directions.items()]
    max_weight = max(all_weights, default=0.0)
    min_weight_cutoff = _PROGRAM_WEIGHT_FRACTION * max_weight

    top_programs = sorted(
        [(p, d, w) for p, (d, w) in program_directions.items() if w >= min_weight_cutoff and w > 0],
        key=lambda x: -x[2],
    )[:top_n_programs]

    # Warn if most programs have zero gamma — likely missing Perturb-seq coverage for GWAS genes.
    all_program_ids = list(program_directions.keys())
    if all_program_ids:
        zero_gamma = sum(1 for p in all_program_ids if program_directions[p][1] == 0)
        if zero_gamma / len(all_program_ids) > 0.5:
            warnings.append(
                f"{zero_gamma}/{len(all_program_ids)} programs have γ=0. "
                "GWAS genes for this disease may not be perturbed in the available Perturb-seq dataset. "
                "GPS program screens will be limited."
            )
            log.warning(
                "GPS: %d/%d programs have γ=0 (missing Perturb-seq coverage?)",
                zero_gamma, len(all_program_ids),
            )

    for prog_id, direction, weight in top_programs:
        # Prefer NMF gene loadings (transcriptomic; matches GPS RGES expectation).
        # Fall back to OTA contribution fractions + MSigDB Hallmark expansion when
        # NMF loadings have insufficient GPS 2198 overlap (structural/cell-type genes
        # are often absent from the GPS landmark set).
        prog_sig = _build_program_signature(prog_id, direction)
        sig_source = "nmf_loadings"
        if len(prog_sig) < _MIN_PROGRAM_SIG_GENES:
            prog_sig = _build_program_signature_from_ota(prog_id, direction, targets)
            sig_source = "ota_proxy"
        n_genes = len(prog_sig)

        programs_screened.append({
            "program_id":  prog_id,
            "direction":   "risk" if direction > 0 else "protective",
            "net_weight":  round(weight, 4),
            "n_sig_genes": n_genes,
            "sig_source":  sig_source,
        })

        if n_genes < _MIN_PROGRAM_SIG_GENES:
            warnings.append(
                f"Program {prog_id}: only {n_genes} loading genes < {_MIN_PROGRAM_SIG_GENES}; skipping GPS"
            )
            continue

        log.info(
            "GPS program reversal: %s (%s, weight=%.3f, %d genes, source=%s)",
            prog_id, "risk" if direction > 0 else "protective", weight, n_genes, sig_source,
        )
        print(  # progress — GPS program screens run 30-90 min each
            f"[GPS] program screen: {prog_id} | {n_genes} genes ({sig_source}) | "
            f"{'risk' if direction > 0 else 'protective'}",
            flush=True,
        )
        hits = screen_disease_for_reversers(
            prog_sig,
            label=f"{disease_key}_prog_{hashlib.sha1(prog_id.encode()).hexdigest()[:8]}",
            library=library,
            top_n=top_n_compounds,
            z_threshold=z_threshold,
            max_hits=max_hits,
        )
        for hit in hits:
            hit["screen_type"] = "program_reversal"
            hit["program_id"]  = prog_id
            hit["direction"]   = "risk" if direction > 0 else "protective"
        log.info("GPS program done: %s → %d hits", prog_id, len(hits))
        print(f"[GPS] program done: {prog_id} → {len(hits)} hits", flush=True)  # progress
        program_reversers[prog_id] = hits

    # ------------------------------------------------------------------
    # Annotate all program reversal hits in bulk (deduplicated).
    # Compounds appearing in multiple programs are only annotated once,
    # then the annotation is backfilled into every hit dict that shares
    # the compound_id. This feeds putative_targets into priority compounds.
    # ------------------------------------------------------------------
    all_prog_hits: list[dict] = [h for hits in program_reversers.values() for h in hits]
    if all_prog_hits:
        seen_ids: set[str] = set()
        unique_prog_hits: list[dict] = []
        for h in all_prog_hits:
            cid = h.get("compound_id", "")
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique_prog_hits.append(h)
        log.info(
            "GPS annotating %d unique program compounds (%d total across %d programs) "
            "via PubChem + ChEMBL",
            len(unique_prog_hits), len(all_prog_hits), len(program_reversers),
        )
        print(
            f"[GPS] annotating {len(unique_prog_hits)} unique program compounds "
            f"({len(all_prog_hits)} total across {len(program_reversers)} programs) "
            "via PubChem + ChEMBL",
            flush=True,
        )
        annotated_unique = annotate_gps_compounds(unique_prog_hits)
        ann_by_id: dict[str, dict] = {
            a.get("annotation", {}).get("compound_id", a.get("compound_id", "")): a.get("annotation", {})
            for a in annotated_unique
        }
        # Backfill annotation into all hits (including duplicates across programs)
        for prog_id, hits in program_reversers.items():
            program_reversers[prog_id] = [
                {**h, "annotation": ann_by_id.get(h.get("compound_id", ""), {})}
                for h in hits
            ]

    return {
        "disease_reversers":  disease_reversers,
        "program_reversers":  program_reversers,
        "disease_sig_n_genes": disease_sig_n,
        "programs_screened":  programs_screened,
        "warnings":           warnings,
    }


# ---------------------------------------------------------------------------
# Disease signature
# ---------------------------------------------------------------------------

def _build_disease_signature(targets: list[dict], disease_query: dict | None = None) -> dict[str, float]:
    """
    Build {gene: log2FC} restricted to GPS's selected_genes_2198 set.

    Priority:
    1. CELLxGENE h5ad pseudo-bulk DEG (disease vs normal) — full transcriptome,
       best GPS overlap. Requires cached h5ad from pipeline run.
    2. tau_disease_log2fc from target records (from CELLxGENE DEG in pipeline).
    3. OTA gamma proxy — available without h5ad, ~72 GPS-compatible genes for AMD.
    """
    from pipelines.gps_screen import _get_gps_genes
    gps_genes = _get_gps_genes()

    def _in_gps(gene: str) -> bool:
        return gps_genes is None or gene in gps_genes

    # Pass 1: full h5ad DEG — best coverage of GPS 2198 genes
    if disease_query:
        sig = _build_sig_from_h5ad(disease_query, gps_genes)
        if len(sig) >= _MIN_DISEASE_SIG_GENES:
            return sig

    sig: dict[str, float] = {}

    # Pass 2: per-target tau_disease_log2fc
    for t in targets:
        gene   = t.get("target_gene", "")
        log2fc = t.get("tau_disease_log2fc") or t.get("disease_log2fc")
        if gene and log2fc is not None and _in_gps(gene):
            try:
                sig[gene] = float(log2fc)
            except (TypeError, ValueError):
                pass

    if len(sig) >= _MIN_DISEASE_SIG_GENES:
        return sig

    # Pass 3: OTA gamma proxy
    gammas = [abs(t.get("ota_gamma", 0.0) or 0.0) for t in targets]
    max_g  = max(gammas) if gammas else 1.0
    if max_g < 1e-6:
        return sig

    for t in targets:
        gene  = t.get("target_gene", "")
        gamma = t.get("ota_gamma", 0.0) or 0.0
        if gene and abs(gamma) > 1e-6 and _in_gps(gene):
            sig[gene] = gamma / max_g

    return sig


# Map disease_key prefixes → cellxgene subdir name(s) to search first.
# Prevents fallback scan from picking a different disease's h5ad alphabetically.
_DISEASE_H5AD_SUBDIRS: dict[str, list[str]] = {
    "coronary_art": ["CAD"],
    "coronary_artery": ["CAD"],
    "cad": ["CAD"],
    "age-related": ["AMD"],
    "age_related": ["AMD"],
    "amd": ["AMD"],
    "macular": ["AMD"],
    "ibd": ["IBD"],
    "inflammatory_bowel": ["IBD"],
    "crohn": ["IBD"],
}


def _build_sig_from_h5ad(disease_query: dict, gps_genes: set | None) -> dict[str, float]:
    """
    Compute pseudo-bulk log2FC (disease vs normal) from cached CELLxGENE h5ad.

    Averages raw counts per condition, adds pseudocount 1, takes log2 ratio.
    Returns only genes in gps_genes (if provided).
    Tries all candidate h5ads in priority order and returns the first one that
    has both ≥10 disease cells AND ≥10 normal cells.
    """
    import numpy as np
    from pathlib import Path

    disease_name = disease_query.get("disease_name", "")
    # Derive disease_key from disease_name when absent from checkpoint phenotype_result
    disease_key = (
        disease_query.get("disease_key")
        or disease_name.lower().replace(" ", "_")
    )

    base = Path(__file__).parent.parent / "data" / "cellxgene"
    candidates: list[Path] = []

    def _add_subdir(subdir: Path) -> None:
        candidates.extend(p for p in sorted(subdir.glob("*retinal_pigment*.h5ad"))
                          if "latent_cache" not in p.name and "state_cache" not in p.name)
        candidates.extend(p for p in sorted(subdir.glob("*.h5ad"))
                          if "latent_cache" not in p.name and "state_cache" not in p.name)

    # 1. Disease-specific abbreviation map (prevents AMD/ from matching CAD queries)
    dk_lower = disease_key.lower()
    for prefix, subdirs in _DISEASE_H5AD_SUBDIRS.items():
        if dk_lower.startswith(prefix) or dk_lower == prefix:
            for subdir_name in subdirs:
                d = base / subdir_name
                if d.exists():
                    _add_subdir(d)

    # 2. Explicit disease_key / disease_name subdir match
    for key in [disease_key, disease_name.replace(" ", "_")]:
        if not key:
            continue
        d = base / key.lower()
        if d.exists():
            _add_subdir(d)

    # 3. Fallback: scan ALL subdirs only when no disease-specific candidates found
    if not candidates:
        for subdir in sorted(base.iterdir()):
            if subdir.is_dir():
                _add_subdir(subdir)

    # Deduplicate preserving order
    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            unique_candidates.append(p)

    if not unique_candidates:
        log.debug("No h5ad found for GPS disease sig (%s)", disease_key)
        return {}

    # Build disease keyword set for relevance filtering (e.g. {"coronary", "artery", "atherosclerosis"})
    _CAD_SYNONYMS = {"coronary", "artery", "arterial", "atherosclerosis", "atherosclerotic", "cad"}
    _AMD_SYNONYMS = {"macular", "degeneration", "amd", "retinal", "retina", "drusen"}
    _IBD_SYNONYMS = {"crohn", "colitis", "ibd", "inflammatory bowel", "bowel"}
    _DISEASE_SYNONYMS: dict[str, set[str]] = {
        "coronary_art": _CAD_SYNONYMS, "coronary_artery": _CAD_SYNONYMS, "cad": _CAD_SYNONYMS,
        "age-related": _AMD_SYNONYMS, "age_related": _AMD_SYNONYMS, "amd": _AMD_SYNONYMS,
        "macular": _AMD_SYNONYMS,
        "ibd": _IBD_SYNONYMS, "inflammatory_bowel": _IBD_SYNONYMS, "crohn": _IBD_SYNONYMS,
    }
    disease_synonyms: set[str] = set()
    for prefix, synonyms in _DISEASE_SYNONYMS.items():
        if dk_lower.startswith(prefix) or dk_lower == prefix:
            disease_synonyms = synonyms
            break
    # Also add words from disease_name itself — filter out generic stopwords
    # that would match any disease h5ad ("disease", "related", "associated")
    _STOPWORDS = {"disease", "related", "associated", "disorder", "syndrome", "condition"}
    disease_synonyms |= {w for w in disease_name.lower().split() if w not in _STOPWORDS}

    # Try each candidate; return first with enough disease AND normal cells of matching disease type
    for h5ad_path in unique_candidates:
        sig = _try_h5ad_sig(h5ad_path, gps_genes, disease_synonyms)
        if sig:
            return sig

    log.debug("No usable h5ad for GPS disease sig (%s) — all candidates lacked disease/normal split",
              disease_key)
    return {}


def _try_h5ad_sig(
    h5ad_path: "Path",
    gps_genes: "set | None",
    disease_synonyms: "set[str] | None" = None,
) -> "dict[str, float]":
    """Attempt to compute GPS disease sig from one h5ad. Returns {} if not usable.

    disease_synonyms: if provided, at least one disease label in the h5ad must contain
    one of these keywords — prevents using a breast-cancer h5ad for a CAD query.
    """
    import numpy as np

    try:
        import anndata as ad
        adata = ad.read_h5ad(h5ad_path, backed="r")

        if "disease" not in adata.obs.columns:
            return {}

        disease_vals = adata.obs["disease"].unique().tolist()
        disease_labels = [v for v in disease_vals if v != "normal" and "normal" not in v.lower()]
        if not disease_labels:
            return {}

        # Reject h5ad if none of its disease labels match the target disease
        if disease_synonyms:
            def _label_matches(label: str) -> bool:
                label_lower = label.lower()
                return any(kw in label_lower for kw in disease_synonyms)
            if not any(_label_matches(lbl) for lbl in disease_labels):
                log.debug("GPS h5ad %s: disease labels %s don't match query — skipping",
                          h5ad_path.name, disease_labels[:3])
                return {}

        disease_mask = adata.obs["disease"].isin(disease_labels).values
        normal_mask  = (adata.obs["disease"] == "normal").values

        n_dis = int(disease_mask.sum())
        n_nrm = int(normal_mask.sum())
        if n_dis < 10 or n_nrm < 10:
            log.debug("GPS h5ad %s: too few cells (disease=%d, normal=%d) — skipping",
                      h5ad_path.name, n_dis, n_nrm)
            return {}

        dis_idx = np.where(disease_mask)[0]
        nrm_idx = np.where(normal_mask)[0]
        try:
            import scipy.sparse as sp
            X_dis = adata.X[dis_idx]
            X_nrm = adata.X[nrm_idx]
            if sp.issparse(X_dis):
                dis_mean = np.asarray(X_dis.mean(axis=0)).flatten()
                nrm_mean = np.asarray(X_nrm.mean(axis=0)).flatten()
            else:
                dis_mean = np.asarray(X_dis).mean(axis=0).flatten()
                nrm_mean = np.asarray(X_nrm).mean(axis=0).flatten()
        except Exception:
            return {}

        log2fc = np.log2(dis_mean + 1) - np.log2(nrm_mean + 1)
        if "feature_name" in adata.var.columns:
            gene_names = list(adata.var["feature_name"])
        else:
            gene_names = list(adata.var_names)

        sig: dict[str, float] = {}
        for gene, lfc in zip(gene_names, log2fc):
            if abs(lfc) > 1e-8 and (gps_genes is None or gene in gps_genes):
                sig[gene] = float(lfc)

        # Elbow-based gene selection: mirroring how PCA elbow plots choose the
        # number of components.  Sort genes by |log2FC| descending (analogous to
        # variance explained per PC) and find the point of maximum curvature —
        # the knee where adding more genes gives diminishing differential signal.
        #
        # Implementation: kneedle algorithm.  Normalize ranks and |log2FC| to
        # [0,1], then find the gene maximally distant from the diagonal connecting
        # the first and last points.  That index is the natural "bend" in the
        # sorted |log2FC| curve.
        #
        # Bounds: min 50 (statistical power), max 200 (GPS was designed for 978
        # GPS sets n_permutations = n_sig_genes internally. <500 permutations →
        # null distribution too sparse → 0 hits (session-58 confirmed: 183 genes
        # → 183 perms → 0 hits; session-56 used ~1000 genes → 20 hits).
        # min=700 ensures ≥700 perms; max=1000 caps screening at ~4500s < 7200s.
        n_before = len(sig)
        sig = _elbow_trim_sig(sig, min_genes=_GPS_BGRD_MIN_PERMS, max_genes=_GPS_BGRD_MAX_PERMS)
        log.info(
            "GPS h5ad sig built: %d DEGs → %d after elbow trim (min=%d, max=%d) | %s "
            "(disease=%d, normal=%d cells)",
            n_before, len(sig), _GPS_BGRD_MIN_PERMS, _GPS_BGRD_MAX_PERMS,
            h5ad_path.name, n_dis, n_nrm,
        )
        print(
            f"[GPS] h5ad sig built: {n_before} DEGs → {len(sig)} after elbow trim "
            f"(min={_GPS_BGRD_MIN_PERMS}, max={_GPS_BGRD_MAX_PERMS}) | {h5ad_path.name} "
            f"(dis={n_dis} cells, nrm={n_nrm} cells)",
            flush=True,
        )

        log.info(
            "GPS disease sig from h5ad (%s): %d GPS-compatible genes after elbow trim "
            "(disease=%d, normal=%d cells)",
            h5ad_path.name, len(sig), n_dis, n_nrm,
        )
        return sig

    except Exception as exc:
        log.debug("GPS h5ad DEG failed for %s: %s", h5ad_path, exc)
        return {}


# ---------------------------------------------------------------------------
# Elbow-based signature trimming
# ---------------------------------------------------------------------------

def _elbow_trim_sig(
    sig: dict[str, float],
    min_genes: int = 700,
    max_genes: int = 1000,
) -> dict[str, float]:
    """
    Return the top-N most differentially expressed genes using an elbow heuristic.

    Motivation: GPS sets n_permutations = n_sig_genes internally, so signature
    size controls both BGRD statistical power and screening runtime. Too few genes
    (< ~500) → sparse null distribution → 0 hits. Too many (> 1000) → screening
    exceeds 7200s WITH_BGRD timeout. Default [700, 1000] balances power and speed.

    Algorithm (kneedle):
      1. Sort genes by |log2FC| descending (= scree curve).
      2. Normalise both axes to [0, 1].
      3. The diagonal connects (0, 1) to (1, 0) — the "straight line" null.
      4. Elbow = gene with maximum perpendicular distance from that diagonal.
         Distance_i ∝ |rank_norm_i + |lfc|_norm_i − 1|  (no sqrt(2) needed).
      5. Clamp result to [min_genes, max_genes].
    """
    if len(sig) <= min_genes:
        return sig

    import numpy as np

    # Sorted (gene, |lfc|) descending
    ranked = sorted(sig.items(), key=lambda kv: -abs(kv[1]))
    abs_lfcs = np.array([abs(v) for _, v in ranked], dtype=float)

    n = len(abs_lfcs)
    x = np.linspace(0.0, 1.0, n)                           # normalised rank axis
    lfc_min, lfc_max = abs_lfcs[-1], abs_lfcs[0]
    y = (abs_lfcs - lfc_min) / max(lfc_max - lfc_min, 1e-10)  # normalised |lfc| axis

    # Perpendicular distance from the line x + y = 1
    distances = np.abs(x + y - 1.0)

    # Only consider indices within [min_genes, max_genes]
    lo = max(0, min_genes - 1)
    hi = min(n - 1, max_genes - 1)
    elbow_idx = int(lo + np.argmax(distances[lo : hi + 1]))

    n_keep = elbow_idx + 1  # 0-indexed → count
    n_keep = max(min_genes, min(max_genes, n_keep))

    trimmed = dict(ranked[:n_keep])
    log.debug(
        "GPS sig elbow trim: %d → %d genes (elbow at rank %d, |lfc|=%.4f)",
        len(sig), n_keep, elbow_idx, abs_lfcs[elbow_idx],
    )
    return trimmed


# ---------------------------------------------------------------------------
# Program direction aggregation
# ---------------------------------------------------------------------------

def _aggregate_program_directions(
    targets: list[dict],
    gamma_estimates: dict | None,
    disease_query: dict,
) -> dict[str, tuple[float, float]]:
    """
    For each NMF program, compute (direction, weight):
      direction: +1 (risk-increasing) or -1 (protective)
      weight:    sum of |contribution| across all targets that use this program

    Sources (in priority order):
    1. gamma_estimates[prog][trait].get("gamma")  — direct γ_{P→AMD} if passed
    2. top_programs contributions aggregated across targets (derive net direction)
    """
    program_net: dict[str, float] = {}   # prog → sum of signed contributions

    # Source 1: explicit gamma_estimates
    if gamma_estimates:
        disease_name = disease_query.get("disease_name", "")
        traits = [disease_name, disease_name.lower(), disease_name.upper()]
        for prog, trait_gammas in gamma_estimates.items():
            for trait, gdict in trait_gammas.items():
                if any(t in trait for t in traits) or any(trait in t for t in traits):
                    gval = gdict.get("gamma") if isinstance(gdict, dict) else gdict
                    if gval is not None:
                        try:
                            program_net[prog] = float(gval)
                        except (TypeError, ValueError):
                            pass

    # Source 2: aggregate top_programs contributions from target records
    # After Fix 4, top_programs = {prog: contribution} for each target
    for t in targets:
        tp = t.get("top_programs") or {}
        if not isinstance(tp, dict):
            continue
        for prog, contrib in tp.items():
            try:
                program_net[prog] = program_net.get(prog, 0.0) + float(contrib)
            except (TypeError, ValueError):
                pass

    # Convert net contribution to (direction, |weight|)
    result: dict[str, tuple[float, float]] = {}
    for prog, net in program_net.items():
        if abs(net) < 1e-8:
            continue
        result[prog] = (1.0 if net >= 0 else -1.0, abs(net))

    return result


# ---------------------------------------------------------------------------
# Program signature
# ---------------------------------------------------------------------------

def _build_program_signature_from_ota(
    program_id: str,
    direction: float,
    targets: list[dict],
) -> dict[str, float]:
    """
    Build a GPS program signature from OTA analysis results.

    For program P, the signature is the set of OTA-ranked target genes whose
    causal contribution runs through P (i.e. β_{gene→P} × γ_{P→disease} is large).
    These are experimentally perturbed genes from Perturb-seq — far better GPS
    overlap than NMF loading vectors, which are dominated by cell-type-specific
    structural genes absent from the GPS 2198 set.

    GPS weight = signed contribution of program P to the gene's OTA γ:
        weight = direction × (program_P_contribution / total_ota_gamma)

    direction > 0: risk program → GPS reversers suppress it → therapeutic
    direction < 0: protective program → GPS reversers reinforce it → therapeutic
    """
    from pipelines.gps_screen import _get_gps_genes
    gps_genes = _get_gps_genes()

    sig: dict[str, float] = {}

    for t in targets:
        gene = t.get("target_gene") or t.get("gene", "")
        if not gene:
            continue
        if gps_genes is not None and gene not in gps_genes:
            continue

        top_progs = t.get("top_programs")
        if not isinstance(top_progs, dict) or program_id not in top_progs:
            continue

        prog_contribution = top_progs.get(program_id, 0.0)
        if not prog_contribution or abs(prog_contribution) < 1e-6:
            continue

        total_gamma = abs(t.get("ota_gamma") or t.get("causal_gamma") or 0.0)
        if total_gamma < 1e-6:
            continue

        # Weight = fraction of γ carried by this program, signed by causal direction
        frac = abs(prog_contribution) / total_gamma
        sig[gene] = direction * frac

    # Augment with MSigDB Hallmark gene set members for this program.
    # OTA-derived weights (above) are kept; Hallmark genes not yet in sig get
    # a lower fixed weight of 0.3 × direction.  This expands sparse program
    # signatures (1–5 OTA genes) to the full 50–200 gene Hallmark set so the
    # GPS threshold of 5 genes is reliably met.
    try:
        from pipelines.cnmf_programs import PROGRAM_TO_HALLMARKS, get_msigdb_hallmark_programs
        hallmark_names = set(PROGRAM_TO_HALLMARKS.get(program_id, []))
        if hallmark_names:
            msigdb = get_msigdb_hallmark_programs()  # in-process cached after first call
            for prog in msigdb.get("programs", []):
                if prog.get("hallmark_name") not in hallmark_names:
                    continue
                for gene in prog.get("gene_set", []):
                    if gps_genes is not None and gene not in gps_genes:
                        continue
                    if gene not in sig:  # don't overwrite OTA-derived weights
                        sig[gene] = direction * 0.3
    except Exception:
        pass

    return sig


def _build_program_signature(program_id: str, direction: float) -> dict[str, float]:
    """
    Build a GPS-compatible {gene: signed_loading} from a program's gene loadings,
    restricted to GPS's selected_genes_2198 set.

    direction > 0: risk program → positive loadings → GPS reversers are therapeutic
    direction < 0: protective program → negate loadings → GPS reversers are therapeutic
    """
    from pipelines.gps_screen import _get_gps_genes
    gps_genes = _get_gps_genes()

    try:
        from mcp_servers.burden_perturb_server import get_program_gene_loadings
        info = get_program_gene_loadings(program_id)
    except Exception as exc:
        log.debug("Could not load gene loadings for %s: %s", program_id, exc)
        return {}

    sig: dict[str, float] = {}
    for g in info.get("top_genes", []):
        if isinstance(g, str):
            gene   = g
            weight = 1.0
        elif isinstance(g, dict):
            gene   = g.get("gene", "") or g.get("symbol", "")
            weight = float(g.get("weight") or g.get("loading") or g.get("score") or 1.0)
        else:
            continue
        if gene and abs(weight) > 1e-6:
            if gps_genes is None or gene in gps_genes:
                sig[gene] = direction * weight

    return sig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty(warnings: list[str]) -> dict:
    return {
        "disease_reversers":  [],
        "program_reversers":  {},
        "disease_sig_n_genes": 0,
        "programs_screened":  [],
        "warnings":           warnings,
    }


# ---------------------------------------------------------------------------
# Convenience: cross-reference overlap
# ---------------------------------------------------------------------------

def annotate_gps_compounds(hits: list[dict], max_workers: int = 4) -> list[dict]:
    """
    Annotate GPS hit compounds (Enamine Z-numbers) with structural info and predicted targets.

    Pipeline per compound:
    1. PubChem by Z-number name → CID, SMILES, InChIKey, MW, formula
    2. ChEMBL 50% Tanimoto similarity → similar known compounds + targets
    3. Falls back gracefully: "novel chemical matter" when no ChEMBL data found.
    """
    import urllib.parse
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import httpx

    PUBCHEM_PROP        = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/InChIKey,IsomericSMILES,CanonicalSMILES,MolecularFormula,MolecularWeight/JSON"
    PUBCHEM_BY_CID      = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IsomericSMILES,CanonicalSMILES/JSON"
    PUBCHEM_BY_CID_CAN  = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
    PUBCHEM_BY_INCHIKEY = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/property/IsomericSMILES,CanonicalSMILES/JSON"
    CHEMBL_SIM       = "https://www.ebi.ac.uk/chembl/api/data/similarity/{smiles}/{threshold}"
    CHEMBL_ACT   = "https://www.ebi.ac.uk/chembl/api/data/activity"
    TIMEOUT      = httpx.Timeout(connect=8.0, read=20.0, write=5.0, pool=5.0)

    def _annotate_one(hit: dict) -> dict:
        compound_id = hit.get("compound_id", "")
        annotation: dict = {
            "compound_id":      compound_id,
            "pubchem_cid":      None,
            "smiles":           None,
            "inchikey":         None,
            "molecular_formula": None,
            "molecular_weight": None,
            "putative_targets":  [],
            "chembl_similarity_hits": [],
            "target_note":      "novel chemical matter — no ChEMBL target data",
        }
        try:
            # Step 1: PubChem lookup by Z-number name
            r = httpx.get(PUBCHEM_PROP.format(name=compound_id), timeout=TIMEOUT)
            if r.status_code == 200:
                props = r.json().get("PropertyTable", {}).get("Properties", [{}])[0]
                annotation["pubchem_cid"]       = props.get("CID")
                annotation["smiles"]            = (props.get("IsomericSMILES") or props.get("CanonicalSMILES")
                                                   or props.get("SMILES"))
                annotation["inchikey"]          = props.get("InChIKey")
                annotation["molecular_formula"] = props.get("MolecularFormula")
                annotation["molecular_weight"]  = props.get("MolecularWeight")
        except Exception:
            pass

        # Step 1b: if name lookup found a CID but no SMILES, retry by CID directly
        # (Enamine Z-numbers often register CID without IsomericSMILES in PubChem)
        if not annotation["smiles"] and annotation["pubchem_cid"]:
            try:
                r_cid = httpx.get(
                    PUBCHEM_BY_CID.format(cid=annotation["pubchem_cid"]),
                    timeout=TIMEOUT,
                )
                if r_cid.status_code == 200:
                    cp = r_cid.json().get("PropertyTable", {}).get("Properties", [{}])[0]
                    annotation["smiles"] = (cp.get("IsomericSMILES") or cp.get("CanonicalSMILES")
                                           or cp.get("SMILES"))
            except Exception:
                pass

        # Step 1c: CanonicalSMILES-only CID request — some Enamine entries omit IsomericSMILES
        # but have CanonicalSMILES when requested alone (avoids 404 from missing stereo property)
        if not annotation["smiles"] and annotation["pubchem_cid"]:
            try:
                r_can = httpx.get(
                    PUBCHEM_BY_CID_CAN.format(cid=annotation["pubchem_cid"]),
                    timeout=TIMEOUT,
                )
                if r_can.status_code == 200:
                    cp = r_can.json().get("PropertyTable", {}).get("Properties", [{}])[0]
                    annotation["smiles"] = cp.get("CanonicalSMILES") or cp.get("SMILES")
            except Exception:
                pass

        # Step 1d: InChIKey-based lookup — last resort for vendors that register InChIKey
        # in PubChem without directly linking IsomericSMILES to the Z-number name
        if not annotation["smiles"] and annotation["inchikey"]:
            try:
                r_ik = httpx.get(
                    PUBCHEM_BY_INCHIKEY.format(inchikey=annotation["inchikey"]),
                    timeout=TIMEOUT,
                )
                if r_ik.status_code == 200:
                    cp = r_ik.json().get("PropertyTable", {}).get("Properties", [{}])[0]
                    annotation["smiles"] = (cp.get("IsomericSMILES") or cp.get("CanonicalSMILES")
                                           or cp.get("SMILES"))
            except Exception:
                pass

        smiles = annotation.get("smiles")
        if not smiles:
            return {**hit, "annotation": annotation}

        try:
            # Step 2: ChEMBL 50% Tanimoto similarity
            encoded = urllib.parse.quote(smiles, safe="")
            r2 = httpx.get(
                CHEMBL_SIM.format(smiles=encoded, threshold=50),
                params={"format": "json", "limit": 5},
                timeout=TIMEOUT,
            )
            if r2.status_code == 200:
                sim_mols = r2.json().get("molecules", [])
                chembl_hits = []
                for m in sim_mols:
                    cid = m.get("molecule_chembl_id")
                    if not cid:
                        continue
                    sim = round(float(m.get("similarity") or 0), 1)
                    # Get activity targets for this similar compound
                    targets: list[str] = []
                    try:
                        r3 = httpx.get(
                            CHEMBL_ACT,
                            params={"molecule_chembl_id": cid, "format": "json", "limit": 10},
                            timeout=TIMEOUT,
                        )
                        if r3.status_code == 200:
                            acts = r3.json().get("activities", [])
                            targets = list(dict.fromkeys(
                                a.get("target_pref_name")
                                for a in acts
                                if a.get("target_pref_name")
                            ))[:4]
                    except Exception:
                        pass
                    chembl_hits.append({
                        "chembl_id":  cid,
                        "name":       m.get("pref_name"),
                        "max_phase":  m.get("max_phase"),
                        "similarity": sim,
                        "targets":    targets,
                    })
                annotation["chembl_similarity_hits"] = chembl_hits

                # Collect unique targets across all similar compounds
                all_targets = list(dict.fromkeys(
                    t for h in chembl_hits for t in h["targets"]
                ))
                if all_targets:
                    annotation["putative_targets"] = all_targets
                    annotation["target_note"] = (
                        f"predicted from ChEMBL structural similarity (≥50% Tanimoto): "
                        f"{', '.join(all_targets[:3])}"
                    )
        except Exception:
            pass

        return {**hit, "annotation": annotation}

    if not hits:
        return hits

    annotated: list[dict] = [None] * len(hits)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_annotate_one, h): i for i, h in enumerate(hits)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                annotated[idx] = fut.result()
            except Exception:
                annotated[idx] = hits[idx]
    return annotated


def find_overlapping_compounds(
    disease_reversers: list[dict],
    emulation_candidates: list[dict],
    program_reversers: dict[str, list[dict]] | None = None,
) -> list[dict]:
    """
    Find compounds appearing in multiple GPS screens.
    Higher overlap = stronger multi-evidence support.

    Returns list of {compound_id, screens, avg_rges, min_rank}.
    """
    from collections import defaultdict

    evidence: dict[str, dict] = defaultdict(lambda: {"screens": [], "rges": [], "ranks": []})

    for hit in disease_reversers:
        cid = hit.get("compound_id", "")
        if cid:
            evidence[cid]["screens"].append("disease_state_reversal")
            evidence[cid]["rges"].append(hit.get("rges", 0.0))
            evidence[cid]["ranks"].append(hit.get("rank", 999))

    for hit in emulation_candidates:
        cid = hit.get("compound_id", "")
        if cid:
            label = f"target_emulation:{hit.get('target','?')}"
            evidence[cid]["screens"].append(label)
            evidence[cid]["rges"].append(hit.get("rges", 0.0))
            evidence[cid]["ranks"].append(hit.get("rank", 999))

    for prog, hits in (program_reversers or {}).items():
        for hit in hits:
            cid = hit.get("compound_id", "")
            if cid:
                evidence[cid]["screens"].append(f"program_reversal:{prog}")
                evidence[cid]["rges"].append(hit.get("rges", 0.0))
                evidence[cid]["ranks"].append(hit.get("rank", 999))

    overlapping = [
        {
            "compound_id": cid,
            "n_screens":   len(set(ev["screens"])),
            "screens":     sorted(set(ev["screens"])),
            "avg_rges":    round(sum(ev["rges"]) / len(ev["rges"]), 4),
            "best_rank":   min(ev["ranks"]),
        }
        for cid, ev in evidence.items()
        if len(set(ev["screens"])) > 1   # only compounds in multiple screens
    ]

    overlapping.sort(key=lambda x: (-x["n_screens"], -x["avg_rges"]))
    return overlapping
