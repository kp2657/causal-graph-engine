"""
steps/tier5_writer/drug_target_validator.py — Clinical drug target validation report.

Runs before plot generation to verify that known approved and validated drug targets:
  - Are present in the ranked target list
  - Show correct γ direction (sign concordance with expected inhibitor/agonist class)
  - Fall in the expected long_island quadrant

Sign convention (OTA framework):
  γ < 0 : gene drives disease (KO suppresses disease) → inhibitor drug target
  γ > 0 : gene protects against disease (KO amplifies disease) → agonist/boost target

Prints a formatted table. Does NOT raise exceptions on failures — all checks are advisory.
"""
from __future__ import annotations

import math


def validate_drug_targets(
    disease_key: str,
    targets: list[dict],
    *,
    gene_key: str = "target_gene",
) -> dict:
    """
    Check clinically validated drug targets against the ranked target list.

    Returns a dict with:
      n_found:          int — validated targets present in output
      n_sign_correct:   int — of those found, how many have the expected γ sign
      n_sign_wrong:     int
      n_no_perturb:     int — genes flagged as virtual/imputed β
      anchor_recovery:  float — fraction of validated targets found
      rows:             list[dict] — per-gene details
    """
    from config.drug_target_registry import get_validated_targets

    validated = get_validated_targets(disease_key)
    if not validated:
        return {"n_found": 0, "n_sign_correct": 0, "n_sign_wrong": 0,
                "n_no_perturb": 0, "anchor_recovery": 0.0, "rows": []}

    # Build gene → target index
    gene_map: dict[str, dict] = {}
    for i, t in enumerate(targets):
        g = t.get(gene_key) or t.get("gene") or ""
        if g:
            gene_map[g] = dict(t, _rank0=i)

    rows: list[dict] = []
    n_found = n_sign_correct = n_sign_wrong = n_no_perturb = 0

    for spec in validated:
        gene = spec["gene"]
        expected_sign = spec["expected_sign"]
        t = gene_map.get(gene)

        gamma      = t.get("ota_gamma") if t else None
        gamma_rest = t.get("ota_gamma_rest") if t else None
        rank  = t.get("rank") if t else None
        tier  = t.get("dominant_tier", "") if t else "ABSENT"
        partition = t.get("partition", "") if t else ""

        if t is None:
            sign_ok = None
            sign_ok_rest = None
        else:
            n_found += 1
            try:
                g = float(gamma)
                if math.isnan(g) or abs(g) < 1e-6:
                    sign_ok = None
                else:
                    actual_sign = 1 if g > 0 else -1
                    sign_ok = actual_sign == expected_sign
                    if sign_ok:
                        n_sign_correct += 1
                    else:
                        n_sign_wrong += 1
            except (TypeError, ValueError):
                sign_ok = None

            try:
                gr = float(gamma_rest) if gamma_rest is not None else float('nan')
                if math.isnan(gr) or abs(gr) < 1e-6:
                    sign_ok_rest = None
                else:
                    sign_ok_rest = (1 if gr > 0 else -1) == expected_sign
            except (TypeError, ValueError):
                sign_ok_rest = None

        # Check for virtual/imputed β dominance: uniform betas = no real Perturb-seq data
        no_perturb = False
        if t:
            top_prog_betas = list((t.get("top_programs") or {}).values())
            if top_prog_betas and len(set(round(b, 4) for b in top_prog_betas)) <= 1:
                no_perturb = True
                n_no_perturb += 1

        # Check bystander confounding: is γ dominated by non-cell-type-specific programs?
        # Uses the same gamma source as the OTA sum for the disease:
        #   RA  → GeneticNMF gammas (SVD excluded from OTA sum)
        #   CAD → SVD gammas (have cell_type_specificity_weight annotations)
        # GeneticNMF programs have no specificity annotation yet, so they never
        # trigger bystander — which is correct: the confounded SVD programs are
        # no longer in the RA OTA sum.
        bystander_dominated = False
        bystander_programs: list[str] = []
        if t and not no_perturb:
            top_progs = t.get("top_programs") or {}
            if isinstance(top_progs, dict) and top_progs:
                try:
                    from pipelines.ldsc.gamma_loader import (
                        get_genetic_nmf_program_gammas,
                        get_cnmf_program_gammas,
                    )
                    from models.disease_registry import get_disease_key
                    _dk = (get_disease_key(disease_key) or disease_key).upper()
                    _prog_gammas = {
                        **get_genetic_nmf_program_gammas(_dk),
                        **get_cnmf_program_gammas(_dk),
                    }
                    total_abs = 0.0
                    bystander_abs = 0.0
                    for pname, beta_val in top_progs.items():
                        pg = _prog_gammas.get(pname, {})
                        p_gamma = pg.get("gamma") or 0.0
                        spec_w = pg.get("cell_type_specificity_weight", 1.0)
                        contrib = abs(float(beta_val) * float(p_gamma))
                        total_abs += contrib
                        if spec_w < 0.5:
                            bystander_abs += contrib
                            if contrib > 0.01 and pname not in bystander_programs:
                                bystander_programs.append(pname.split("_")[-1])
                    if total_abs > 0 and bystander_abs / total_abs > 0.40:
                        bystander_dominated = True
                except Exception:
                    pass

        rows.append(dict(
            gene=gene,
            rank=rank,
            gamma=gamma,
            gamma_rest=gamma_rest,
            expected_sign=expected_sign,
            sign_ok=sign_ok,
            sign_ok_rest=sign_ok_rest,
            partition=partition,
            tier=tier,
            drug=spec["drug"],
            status=spec["status"],
            no_perturb=no_perturb,
            bystander_dominated=bystander_dominated,
            bystander_programs=bystander_programs,
            note=spec["note"],
        ))

    anchor_recovery = n_found / len(validated) if validated else 0.0

    return dict(
        n_found=n_found,
        n_sign_correct=n_sign_correct,
        n_sign_wrong=n_sign_wrong,
        n_no_perturb=n_no_perturb,
        anchor_recovery=anchor_recovery,
        rows=rows,
    )


def print_drug_target_report(disease_key: str, targets: list[dict], gene_key: str = "target_gene") -> dict:
    """Validate and print a formatted drug target validation table. Returns summary dict."""
    result = validate_drug_targets(disease_key, targets, gene_key=gene_key)
    rows = result["rows"]
    n_validated = len(rows)

    n_bystander = sum(1 for r in rows if r.get("bystander_dominated"))

    print(f"\n{'='*70}")
    print(f"DRUG TARGET VALIDATION — {disease_key.upper()}")
    print(f"  Validated targets checked: {n_validated}")
    print(f"  Found in output:           {result['n_found']} / {n_validated} ({100*result['anchor_recovery']:.0f}%)")
    print(f"  Direction correct (γ sign): {result['n_sign_correct']} / {result['n_found']} found")
    if result["n_no_perturb"]:
        print(f"  Virtual/imputed β only:    {result['n_no_perturb']} (direction unreliable)")
    if n_bystander:
        print(f"  Bystander-confounded γ:    {n_bystander} (>40% γ from non-cell-type programs)")
    print(f"{'='*70}")

    _has_rest = any(r.get("gamma_rest") is not None for r in rows)

    # Header
    if _has_rest:
        print(f"  {'Gene':10s} {'Rank':>6} {'γ Stim':>8} {'Dir':>4}  {'γ Rest':>8} {'Dir':>4}  {'Exp':>4}  {'Drug / Status':38s}  Partition")
        print(f"  {'-'*10} {'-'*6} {'-'*8} {'-'*4}  {'-'*8} {'-'*4}  {'-'*4}  {'-'*38}  ---------")
    else:
        print(f"  {'Gene':10s} {'Rank':>6} {'γ':>7}  {'Exp':>4} {'Dir':>4}  {'Drug / Status':40s}  Partition")
        print(f"  {'-'*10} {'-'*6} {'-'*7}  {'-'*4} {'-'*4}  {'-'*40}  ---------")

    def _dir_flag(sign_ok, no_perturb) -> str:
        if no_perturb:
            return "~V"
        return "  "

    for r in sorted(rows, key=lambda x: (x["rank"] is None, x["rank"] or 9999)):
        rank_str   = f"#{r['rank']}" if r["rank"] is not None else "ABSENT"
        gamma_str  = f"{r['gamma']:+.3f}" if r["gamma"] is not None else "  n/a "
        exp_str    = "  +" if r["expected_sign"] == 1 else "  -"
        dir_flag   = _dir_flag(r.get("sign_ok"), r["no_perturb"])

        virt_flag  = " [virt]" if r["no_perturb"] else (
            f" [byst:{','.join(r.get('bystander_programs', [])[:2])}]"
            if r.get("bystander_dominated") else ""
        )
        drug_str   = f"{r['drug']} ({r['status']})"
        part_short = (r["partition"] or "")[:18]

        if _has_rest:
            gr = r.get("gamma_rest")
            gr_str   = f"{gr:+.3f}" if gr is not None else "  n/a "
            dir_rest = _dir_flag(r.get("sign_ok_rest"), r["no_perturb"])
            print(
                f"  {r['gene']:10s} {rank_str:>6} {gamma_str:>8} {dir_flag:>4}  "
                f"{gr_str:>8} {dir_rest:>4}  {exp_str:>4}  {drug_str[:38]:38s}  {part_short}{virt_flag}"
            )
        else:
            print(f"  {r['gene']:10s} {rank_str:>6} {gamma_str:>7}  {exp_str:>4} {dir_flag:>4}  {drug_str[:40]:40s}  {part_short}{virt_flag}")

    print()

    return result
