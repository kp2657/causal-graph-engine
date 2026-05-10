Causal graph analysis of rheumatoid arthritis (EFO: EFO_0000685) identified 152 significant causal edges spanning 496 therapeutic targets. Top-ranked targets are SMAD3, DENND1B, IL12RB2. 623 gene–program β estimate(s) use the `provisional_virtual` evidence tier (in silico / weak perturbation support); other evidence streams may still be experimental.


## Target Rankings

| Rank | Gene | Ota γ | Tier | OT Score | Max Phase | Key Evidence |
|------|------|-------|------|----------|-----------|--------------|
| 1 | SMAD3 | 0.867 | provisional_virtual | 0.00 | Phase 0 | — |
| 2 | DENND1B | 0.829 | Tier2_eQTL_direction | 0.00 | Phase 0 | — |
| 3 | IL12RB2 | 0.691 | provisional_virtual | 0.00 | Phase 0 | — |
| 4 | ICOS | 0.661 | provisional_virtual | 0.00 | Phase 0 | — |
| 5 | TPD52 | 0.543 | Tier2_eQTL_direction | 0.00 | Phase 0 | — |
| 6 | ZC3H11A | 0.524 | provisional_virtual | 0.00 | Phase 0 | — |
| 7 | MST1 | 0.492 | provisional_virtual | 0.00 | Phase 0 | — |
| 8 | NUGGC | 0.477 | Tier2_PerturbNominated | 0.00 | Phase 0 | — |
| 9 | MAGI3 | 0.467 | provisional_virtual | 0.00 | Phase 0 | — |
| 10 | SOX4 | 0.466 | Tier2_eQTL_direction | 0.00 | Phase 0 | — |


## Top Target Narratives

SMAD3 is a master regulator of RA_GeneticNMF_Stim48hr_C25 (β=1.12). Although no direct GWAS hit, it controls the RA_GeneticNMF_Stim48hr_C25 axis which is genetically causal for rheumatoid arthritis (program γ=1.00).
SMAD3 upregulates disease-relevant RA_GeneticNMF_Stim48hr_C03, RA_GeneticNMF_Stim48hr_C05, RA_GeneticNMF_Stim48hr_C08 → disease trait [provisional_virtual — in silico estimate, awaiting experimental data].
Evidence: GWAS + Perturb-seq.
Composite Ota estimate: γ_{SMAD3→trait} = 0.867 (provisional_virtual).
Best compound: no approved drug (no in vitro activity data; Phase 0).

DENND1B is a master regulator of RA_GeneticNMF_Stim48hr_C25 (β=0.70). Although no direct GWAS hit, it controls the RA_GeneticNMF_Stim48hr_C25 axis which is genetically causal for rheumatoid arthritis (program γ=1.00).
DENND1B upregulates disease-relevant RA_SVD_C17, RA_SVD_C09, RA_SVD_C15 → disease trait.
Evidence: GWAS + Perturb-seq.
Composite Ota estimate: γ_{DENND1B→trait} = 0.829 (Tier2_eQTL_direction).
Best compound: no approved drug (no in vitro activity data; Phase 0).

IL12RB2 is a master regulator of RA_GeneticNMF_Stim48hr_C25 (β=0.87). Although no direct GWAS hit, it controls the RA_GeneticNMF_Stim48hr_C25 axis which is genetically causal for rheumatoid arthritis (program γ=1.00).
IL12RB2 upregulates disease-relevant RA_GeneticNMF_Stim48hr_C03, RA_GeneticNMF_Stim48hr_C04, RA_GeneticNMF_Stim48hr_C05 → disease trait [provisional_virtual — in silico estimate, awaiting experimental data].
Evidence: GWAS + Perturb-seq.
Composite Ota estimate: γ_{IL12RB2→trait} = 0.691 (provisional_virtual).
Best compound: no approved drug (no in vitro activity data; Phase 0).


## Limitations

About 623 gene–program β rows are labelled `provisional_virtual` (in silico or weakly supported cell-model estimates). That label applies to those β edges, not to the entire analysis: GWAS/L2G and other tiers may still be experimentally grounded. Treat virtual-tier rows as hypothesis-generating. Full quantitative Perturb-seq β coverage may require local GEO GSE246756 downloads (~50GB; Replogle 2022) when not cached. Program γ estimates are context-dependent; validate lead findings experimentally.