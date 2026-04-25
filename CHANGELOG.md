# Changelog

## v0.2.3 (2026-04-25)
- CAD: 1,586 targets, 4,896 causal edges (validated)
- RA: 244 targets, 488 causal edges (validated)
- SLE dropped (insufficient OT L2G coverage for EFO_0002690)
- GPS Z_RGES threshold raised to 3.5σ; max_hits cap removed from z-scored path
- Program γ edges (CellularProgram → DiseaseTrait) written to Kùzu graph
- Required anchor QC gate removed; polybic selection is now the sole edge filter
- Dead code removed: sdk_tools, scone_sensitivity (except polybic_selection), agent_runner, message_contracts

## v0.2.2 (2026-04-24)
- GPS program screens: bidirectional signature centering fix (was producing 0 reversers)
- Perturb-seq cache selection: largest-file wins (was selecting wrong cache)
- S-LDSC γ wired; cNMF real sklearn NMF; OT optimal_transport mode

## v0.2.1 (2026-04-21)
- Transition-zone signal (BP cell selection), pLI penalty, entropy discount, mechanistic filter
- OT API caching: 4800+ calls → ~26 per run (30-day SQLite cache)
- BGRD size-bucketed cache shared across disease runs

## v0.2.0 (2026-04-19)
- Public release readiness: synthetic data removed, disease_registry + scoring_thresholds created
- AMD removed; CAD + RA active
- OTA algorithm documented in docs/OTA_ALGORITHM.md
- CI/CD via GitHub Actions
