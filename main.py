"""
main.py — Causal Graph Engine CLI entry point.

Usage:
  python main.py analyze <disease>           # Full Ota pipeline
  python main.py update <disease> <type>     # Incremental update
  python main.py validate <disease>          # Graph quality report
  python main.py export <disease> [dir]      # Export to RDF/JSON-LD/CSV
  python main.py snapshot <version> [notes]  # Create versioned snapshot
  python main.py snapshots                   # List all snapshots
  python main.py rollback <version>          # Restore a snapshot
  python main.py schedule [disease]          # Start update scheduler

Environment:
  GRAPH_DB_PATH   — Kùzu database path (default: ./data/graph.kuzu)
  SNAPSHOT_DIR    — Snapshot storage (default: ./data/snapshots)
  TARGET_DISEASE  — Default disease for scheduler (default: coronary artery disease)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("causal-graph-engine")


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _disease_slug(disease_name: str) -> str:
    return disease_name.lower().replace(" ", "_")


def _write_markdown_report(result: dict, path: Path) -> None:
    """Write a human-readable Markdown report from a GraphOutput dict."""
    disease   = result.get("disease_name", "Unknown")
    efo       = result.get("efo_id", "")
    generated = result.get("generated_at", "")[:10]
    version   = result.get("pipeline_version", "0.1.0")
    status    = result.get("pipeline_status", "")
    recovery  = result.get("anchor_edge_recovery", 0.0)
    duration  = result.get("pipeline_duration_s")

    # Import narrative generator so reports are always fresh even from old JSON
    from agents.tier5_writer.scientific_writer_agent import _causal_narrative

    lines: list[str] = [
        f"# Causal Graph Analysis: {disease.title()}",
        f"",
        f"**EFO ID:** {efo}  |  **Generated:** {generated}  |  **Version:** {version}",
        f"**Pipeline status:** {status}  |  **Anchor recovery:** {recovery:.0%}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        result.get("executive_summary", ""),
        "",
        "---",
        "",
        "## Target Rankings",
        "",
        result.get("target_table", "_No targets ranked._"),
        "",
        "---",
        "",
        "## Causal Pathway Narratives",
        "",
    ]

    # Re-generate narratives from target_list so the report is always current
    target_list = result.get("target_list", result.get("targets", []))
    chemistry   = result.get("chemistry_result", {})
    trials      = result.get("trials_result", {})
    for i, target in enumerate(target_list[:3], 1):
        narrative = _causal_narrative(target, chemistry, trials)
        lines.append(f"### Target {i}: {target.get('target_gene', '?')}")
        lines.append("")
        lines.append(narrative)
        lines.append("")

    eq = result.get("evidence_quality", {})
    lines += [
        "---",
        "",
        "## Evidence Quality",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Anchor edge recovery | {eq.get('anchor_edge_recovery_rate', 'N/A')} |",
        f"| Tier 1 edges (interventional) | {eq.get('n_tier1_edges', 0)} |",
        f"| Tier 2 edges (convergent MR) | {eq.get('n_tier2_edges', 0)} |",
        f"| Tier 3 edges (provisional) | {eq.get('n_tier3_edges', 0)} |",
        f"| Virtual edges (in silico) | {eq.get('n_virtual_edges', 0)} |",
        f"| SHD from reference | {eq.get('shd_from_reference', 'N/A')} |",
        f"| Pipeline duration | {f'{duration:.0f}s' if duration else 'N/A'} |",
        "",
        "---",
        "",
        "## Limitations",
        "",
        result.get("limitations", ""),
        "",
    ]

    warnings = result.get("warnings", [])
    if warnings:
        lines += [
            "---",
            "",
            "## Pipeline Warnings",
            "",
        ]
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def cmd_analyze(disease_name: str) -> None:
    """Run full Ota pipeline for a disease."""
    from orchestrator.pi_orchestrator import analyze_disease

    print(f"\n[analyze] Running full pipeline for: {disease_name}")
    try:
        result = analyze_disease(disease_name)
    except ValueError as exc:
        print(f"\n[HALTED] Pipeline stopped: {exc}")
        sys.exit(1)

    slug = _disease_slug(disease_name)
    data_dir = Path("./data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Pretty-print summary
    targets = result.get("target_list", result.get("targets", []))
    print(f"\n{'='*60}")
    print(f"  Disease:          {disease_name}")
    print(f"  Pipeline status:  {result.get('pipeline_status', 'N/A')}")
    print(f"  Anchor recovery:  {result.get('anchor_edge_recovery', 0):.0%}")
    print(f"  Edges written:    {result.get('total_edges_written', result.get('n_edges', 'N/A'))}")
    print(f"  Top targets ({len(targets)}):")
    for t in targets[:5]:
        gene  = t.get("target_gene", t.get("gene", t.get("target", "?")))
        gamma = t.get("ota_gamma", t.get("composite_score", t.get("score", "—")))
        tier  = t.get("evidence_tier", "")
        print(f"    {t.get('rank', '?')}. {gene}  γ={gamma}  [{tier}]")
    print(f"{'='*60}\n")

    # 1. JSON — full machine-readable result
    json_path = data_dir / f"analyze_{slug}.json"
    with json_path.open("w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"JSON result:      {json_path}")

    # 2. Markdown — human-readable report
    md_path = data_dir / f"analyze_{slug}.md"
    _write_markdown_report(result, md_path)
    print(f"Markdown report:  {md_path}")

    # 3. Figures — target rankings, causal network, evidence summary
    try:
        from graph.visualize import generate_report_figures
        figures_dir = data_dir / "figures"
        fig_paths = generate_report_figures(result, output_dir=str(figures_dir))
        for name, p in fig_paths.items():
            print(f"Figure ({name}): {p}")
    except Exception as exc:
        logger.warning(f"Figure generation failed (non-fatal): {exc}")

    # 4. Graph export — RDF/Turtle, JSON-LD, CSV
    try:
        from graph.export import export_disease_graph
        exports_dir = data_dir / "exports"
        export_result = export_disease_graph(disease_name, output_dir=str(exports_dir))
        print(f"RDF/Turtle:       {export_result.get('turtle_path', 'N/A')}")
        print(f"JSON-LD:          {export_result.get('jsonld_path', 'N/A')}")
        print(f"CSV edge list:    {export_result.get('csv_path', 'N/A')}")
    except Exception as exc:
        logger.warning(f"Graph export failed (non-fatal): {exc}")


def cmd_update(disease_name: str, update_type: str) -> None:
    """Run an incremental graph update."""
    from graph.update_pipeline import run_update

    valid_types = {"gwas", "literature", "clinical_trials", "full"}
    if update_type not in valid_types:
        print(f"[ERROR] update_type must be one of: {valid_types}")
        sys.exit(1)

    print(f"\n[update] Running {update_type} update for: {disease_name}")
    result = run_update(disease_name, update_type=update_type, auto_snapshot=True)  # type: ignore[arg-type]

    print(f"\n  Status:    {result['status']}")
    print(f"  Submitted: {result['n_new_edges']}")
    print(f"  Approved:  {result['n_approved']}")
    print(f"  Rejected:  {result['n_rejected']}")
    if result.get("snapshot", {}).get("version_tag"):
        print(f"  Snapshot:  {result['snapshot']['version_tag']}")


def cmd_validate(disease_name: str) -> None:
    """Run graph quality validation."""
    from graph.validation import validate_graph, validation_report_to_dict

    print(f"\n[validate] Running quality checks for: {disease_name}")
    report = validate_graph(disease_name)
    d = validation_report_to_dict(report)

    print(f"\n{'='*60}")
    print(f"  Anchor recovery: {d['anchor_recovery_rate']:.0%}  "
          f"({d['n_anchors_recovered']}/{d['n_anchors_total']})")
    print(f"  SHD:             {d['shd']}")
    print(f"  SID approx:      {d['sid']:.2f}" if d['sid'] is not None else "  SID approx: N/A")
    print(f"  Total edges:     {d['n_edges_total']}  (demoted: {d['n_edges_demoted']})")
    print(f"  Low E-value:     {d['n_low_evalue']}")
    status = "PASS" if d["passed"] else "FAIL"
    print(f"  Result:          {status}")
    if d["errors"]:
        for e in d["errors"]:
            print(f"  [ERROR] {e}")
    if d["warnings"]:
        for w in d["warnings"]:
            print(f"  [WARN]  {w}")
    print(f"{'='*60}\n")


def cmd_export(disease_name: str, output_dir: str = "./data/exports") -> None:
    """Export graph to RDF/Turtle, JSON-LD, and CSV."""
    from graph.export import export_disease_graph

    print(f"\n[export] Exporting graph for: {disease_name}")
    result = export_disease_graph(disease_name, output_dir=output_dir)

    print(f"\n  Edges exported: {result['n_edges']}")
    print(f"  Turtle:         {result['turtle_path']}")
    print(f"  JSON-LD:        {result['jsonld_path']}")
    print(f"  CSV:            {result['csv_path']}\n")


def cmd_snapshot(version_tag: str, release_notes: str = "") -> None:
    """Create a versioned snapshot of the current graph."""
    from graph.versioning import create_snapshot

    print(f"\n[snapshot] Creating snapshot: {version_tag}")
    result = create_snapshot(version_tag, release_notes=release_notes)

    print(f"\n  Version:  {result['version_tag']}")
    print(f"  Path:     {result['snapshot_path']}")
    print(f"  Created:  {result['created_at']}\n")


def cmd_list_snapshots() -> None:
    """List all available snapshots."""
    from graph.versioning import list_snapshots

    snapshots = list_snapshots()
    if not snapshots:
        print("\n  No snapshots found.\n")
        return

    print(f"\n  {'Version':<12} {'Created':<30} {'Release Notes'}")
    print("  " + "-" * 72)
    for s in snapshots:
        notes = s.get("release_notes", "")[:40]
        print(f"  {s['version_tag']:<12} {s['created_at'][:26]:<30} {notes}")
    print()


def cmd_rollback(version_tag: str) -> None:
    """Restore the graph database from a snapshot."""
    from graph.versioning import rollback_to_snapshot

    print(f"\n[rollback] Restoring snapshot: {version_tag}")
    result = rollback_to_snapshot(version_tag)

    print(f"\n  Status:      {result['status']}")
    print(f"  Version:     {result['version_tag']}")
    print(f"  Restored at: {result['restored_at']}\n")


def cmd_schedule(disease_name: str | None = None) -> None:
    """Start the APScheduler update daemon."""
    from scheduler.update_scheduler import start

    print(f"\n[schedule] Starting update scheduler (Ctrl-C to stop)")
    if disease_name:
        print(f"  Disease: {disease_name}")
    start(disease_name)


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------

def _usage() -> None:
    print(__doc__)
    sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]

    if not args:
        _usage()

    cmd = args[0].lower()

    if cmd == "analyze":
        if len(args) < 2:
            print("[ERROR] Usage: main.py analyze <disease>")
            sys.exit(1)
        cmd_analyze(" ".join(args[1:]))

    elif cmd == "update":
        if len(args) < 3:
            print("[ERROR] Usage: main.py update <disease> <gwas|literature|clinical_trials|full>")
            sys.exit(1)
        cmd_update(disease_name=args[1], update_type=args[2])

    elif cmd == "validate":
        if len(args) < 2:
            print("[ERROR] Usage: main.py validate <disease>")
            sys.exit(1)
        cmd_validate(" ".join(args[1:]))

    elif cmd == "export":
        if len(args) < 2:
            print("[ERROR] Usage: main.py export <disease> [output_dir]")
            sys.exit(1)
        disease = args[1]
        out_dir = args[2] if len(args) > 2 else "./data/exports"
        cmd_export(disease, out_dir)

    elif cmd == "snapshot":
        if len(args) < 2:
            print("[ERROR] Usage: main.py snapshot <version> [notes]")
            sys.exit(1)
        version = args[1]
        notes = " ".join(args[2:]) if len(args) > 2 else ""
        cmd_snapshot(version, notes)

    elif cmd == "snapshots":
        cmd_list_snapshots()

    elif cmd == "rollback":
        if len(args) < 2:
            print("[ERROR] Usage: main.py rollback <version>")
            sys.exit(1)
        cmd_rollback(args[1])

    elif cmd == "schedule":
        disease = " ".join(args[1:]) if len(args) > 1 else None
        cmd_schedule(disease)

    else:
        print(f"[ERROR] Unknown command: {cmd!r}")
        _usage()


if __name__ == "__main__":
    main()
