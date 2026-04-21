"""
validate_results.py — Cell-type-aware validation of AMD and CAD pipeline outputs.

Checks whether the ranked target list recovers known disease-relevant genes for
the specific cell types used in the pipeline:
  - AMD: RPE1 Perturb-seq + RPE CELLxGENE h5ad
  - CAD: Schnitzler HCASMC/HAEC Perturb-seq + SMC CELLxGENE h5ad

Usage:
    python scripts/validate_results.py
    python scripts/validate_results.py --disease amd
    python scripts/validate_results.py --disease cad
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Known disease-relevant genes by cell type
# ---------------------------------------------------------------------------

# AMD — validated in RPE / RPE1 biology
# Sources: AMD GWAS (Fritsche et al. 2016 Nat Genet), RPE biology literature
AMD_RPE_VALIDATED = {
    # Complement cascade (top AMD GWAS locus; CFH Y402H is the canonical AMD variant)
    "CFH":   "complement factor H — top AMD GWAS gene (Y402H variant, OR ~7); druggable",
    "CFHR1": "CFH-related protein 1 — CFH locus deletion protective for AMD",
    "CFHR3": "CFH-related protein 3 — CFH locus; CFHR1/CFHR3 deletion modifies complement",
    "C3":    "complement C3 — AMD GWAS; APL-9 (C3 inhibitor) in Phase 2 trials",
    "CFB":   "complement factor B — AMD GWAS; Iptacopan (CFB inhibitor) Phase 2",
    "CFI":   "complement factor I — AMD GWAS; gene therapy trial (FOCUS study)",
    # RPE-specific biology
    "RDH5":  "retinol dehydrogenase 5 — RPE visual cycle; mutations cause fundus albipunctatus",
    "TSPAN10": "tetraspanin 10 — AMD GWAS hit; RPE-expressed",
    "BEST1": "bestrophin 1 — RPE chloride channel; mutations cause Best macular dystrophy",
    # Lipid / druggable
    "LIPC":  "hepatic lipase — AMD GWAS; lipid metabolism in RPE",
    "ABO":   "ABO blood group — AMD GWAS association",
    # Growth factors / druggable
    "VEGFA": "VEGF-A — anti-VEGF drugs (ranibizumab, bevacizumab) approved for wet AMD",
    # Stress response
    "NFE2L2": "NRF2 — master antioxidant regulator; RPE oxidative stress in AMD",
    "KEAP1":  "KEAP1 — NRF2 regulator; RPE-relevant druggable node",
}

# CAD — validated in SMC / endothelial biology
# Sources: CAD GWAS (van der Harst & Verweij 2018; Aragam et al. 2022),
#          SMC Perturb-seq (Schnitzler et al. GSE210681), HAEC (Natsume 2023)
CAD_VASCULAR_VALIDATED = {
    # Lipid metabolism — top CAD drugged targets
    "PCSK9":  "PCSK9 — LDL receptor degradation; evolocumab/alirocumab approved",
    "LDLR":   "LDL receptor — familial hypercholesterolemia; statin target",
    "HMGCR":  "HMG-CoA reductase — statin target; top CAD genetic signal",
    "SORT1":  "sortilin — top CAD GWAS locus (1p13); regulates PCSK9 secretion",
    "APOB":   "apolipoprotein B — LDL particle; inclisiran/mipomersen targets",
    "LPA":    "lipoprotein(a) — independent CAD risk; olpasiran (RNA silencer) Phase 3",
    "LIPC":   "hepatic lipase — HDL metabolism; CAD GWAS hit",
    "CETP":   "CETP — HDL/LDL exchange; anacetrapib (trial), dalcetrapib (failed)",
    # Vascular SMC biology (Schnitzler HCASMC-relevant)
    "TGFB1":  "TGF-beta 1 — SMC differentiation and fibrosis; CAD GWAS",
    "MMP9":   "matrix metalloprotease 9 — plaque instability; SMC-expressed",
    "PDGFRB": "PDGF receptor beta — SMC proliferation; imatinib inhibitor",
    "KLF4":   "Krüppel-like factor 4 — SMC phenotype switching; GWAS signal",
    "NOTCH1": "Notch 1 — SMC/endothelial fate; CAD GWAS locus",
    # Endothelial biology (Natsume HAEC-relevant)
    "VWF":    "von Willebrand factor — endothelial thrombosis marker; CAD GWAS",
    "PROCR":  "protein C receptor — endothelial anticoagulation; CAD GWAS",
    # Inflammation
    "IL6R":   "IL-6 receptor — tocilizumab target; Mendelian randomization validated",
    "CRP":    "C-reactive protein — causal in CAD by MR; anti-inflammatory target",
}

# ---------------------------------------------------------------------------
# GPS reverser biology hints
# ---------------------------------------------------------------------------

# Compound classes known to affect SMC biology — for CAD GPS validation
CAD_SMC_RELEVANT_MECHANISMS = [
    "mTOR inhibitor",     # rapamycin analogs reduce SMC proliferation (RAPA-based stents)
    "PDGFR inhibitor",    # imatinib, sunitinib — SMC growth
    "TGF-beta inhibitor", # SMC fibrosis
    "statin",             # HMGCR; SMC lipid metabolism
    "ACE inhibitor",      # vascular remodeling
    "ARB",                # angiotensin; SMC
]

AMD_RPE_RELEVANT_MECHANISMS = [
    "complement inhibitor",  # CFH/C3/CFB pathway
    "anti-VEGF",             # ranibizumab class
    "antioxidant",           # NRF2 activators for RPE oxidative stress
    "visual cycle",          # RDH5/RPE65 pathway
]


# ---------------------------------------------------------------------------
# Load and validate
# ---------------------------------------------------------------------------

def validate_disease(disease: str) -> None:
    # Find latest output JSON
    json_candidates = sorted(
        DATA_DIR.glob(f"analyze_{disease.lower().replace(' ', '_').replace('-', '_')}*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    # Also try hyphenated name
    json_candidates += sorted(
        DATA_DIR.glob(f"analyze_{disease.lower().replace(' ', '-')}*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    path = None
    for p in json_candidates:
        if p.stat().st_size > 10_000:  # skip near-empty outputs
            path = p
            break

    if not path:
        print(f"[{disease.upper()}] No output JSON found in {DATA_DIR}")
        return

    print(f"\n{'='*70}")
    print(f"VALIDATION: {disease.upper()}  ({path.name})")
    print(f"{'='*70}")

    with open(path) as f:
        d = json.load(f)

    # Find target list
    targets: list[dict] = []
    pr = d.get("prioritization_result", {})
    for key in ("targets", "target_list", "genetic_anchors"):
        candidates = pr.get(key) or d.get(key, [])
        if candidates:
            targets = candidates
            break

    if not targets:
        print("  No target list found in output.")
        return

    gene_field = "target_gene" if "target_gene" in targets[0] else "gene_symbol"
    print(f"  Total targets ranked: {len(targets)}")

    # Select known gene set for this disease
    is_amd = "macular" in disease.lower() or "amd" in disease.lower()
    known_genes = AMD_RPE_VALIDATED if is_amd else CAD_VASCULAR_VALIDATED
    cell_type_label = "RPE / RPE1 Perturb-seq" if is_amd else "SMC / HCASMC / HAEC"

    print(f"\n  Cell type context: {cell_type_label}")
    print(f"\n  Known disease genes recovered in ranked list:")
    print(f"  {'Gene':<10} {'Rank':>6}  {'Score':>7}  {'Tier':<25}  Evidence")
    print(f"  {'-'*80}")

    gene_index = {t[gene_field]: (i + 1, t) for i, t in enumerate(targets) if t.get(gene_field)}
    n_recovered = 0
    n_top20 = 0
    n_top100 = 0

    for gene, note in sorted(known_genes.items(), key=lambda x: gene_index.get(x[0], (9999, {}))[0]):
        if gene in gene_index:
            rank, t = gene_index[gene]
            score = round(t.get("target_score", t.get("causal_gamma", 0)), 3)
            tier = t.get("evidence_tier", "?")
            n_recovered += 1
            if rank <= 20:
                n_top20 += 1
            if rank <= 100:
                n_top100 += 1
            flag = "  ★" if rank <= 20 else (" ●" if rank <= 100 else "")
            print(f"  {gene:<10} {rank:>6}  {score:>7.3f}  {tier:<25}  {note[:55]}{flag}")
        else:
            print(f"  {gene:<10} {'—':>6}  {'—':>7}  {'not in output':<25}  {note[:55]}")

    total_known = len(known_genes)
    print(f"\n  Recovery: {n_recovered}/{total_known} known genes in ranked list")
    print(f"           {n_top20}/{total_known} in top 20  |  {n_top100}/{total_known} in top 100")

    # GPS results — check top-level keys (v0.2.0) then fall back to nested (legacy)
    chem = d.get("chemistry_result", {})
    gps_reversers = d.get("gps_disease_state_reversers") or chem.get("gps_disease_reversers", [])
    prog_reversers_list = d.get("gps_program_reversers") or []
    prog_reversers_dict = chem.get("gps_program_reversers", {})

    print(f"\n  GPS disease-state reversers: {len(gps_reversers)}")
    if gps_reversers:
        for h in gps_reversers[:5]:
            cid = h.get("compound_id", "?")
            z = h.get("z_rges", h.get("rges", "?"))
            ann = h.get("annotation", {})
            target = ann.get("putative_targets", "?")
            mech = ann.get("mechanism_of_action", ann.get("target_class", "?"))
            print(f"    {cid:<20}  Z_RGES={z:<6}  target={target}  mech={mech}")

    if prog_reversers_list:
        print(f"  GPS program reversers: {len(prog_reversers_list)}")
    elif prog_reversers_dict:
        total_prog = sum(len(v) for v in prog_reversers_dict.values())
        print(f"  GPS program reversers: {total_prog} across {len(prog_reversers_dict)} programs")

    # Top 5 targets narrative
    print(f"\n  Top 10 ranked targets:")
    for t in targets[:10]:
        g = t.get(gene_field, "?")
        s = round(t.get("target_score", t.get("causal_gamma", 0)), 3)
        tier = t.get("evidence_tier", "?")
        is_known = "  [KNOWN]" if g in known_genes else ""
        print(f"    {t.get('rank', '?'):>4}. {g:<12} {s:.3f}  {tier}{is_known}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate AMD/CAD pipeline outputs")
    parser.add_argument("--disease", choices=["amd", "cad", "both"], default="both")
    args = parser.parse_args()

    diseases = []
    if args.disease in ("amd", "both"):
        diseases.append("age_related_macular_degeneration")
    if args.disease in ("cad", "both"):
        diseases.append("coronary_artery_disease")

    for d in diseases:
        validate_disease(d)

    print()


if __name__ == "__main__":
    main()
