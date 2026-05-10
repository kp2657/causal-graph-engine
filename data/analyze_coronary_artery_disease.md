Causal graph analysis of coronary artery disease (EFO: EFO_0001645) identified 6661 significant causal edges spanning 1689 therapeutic targets. Top-ranked targets are INTS7, VEGFA, FGF2. 183 gene–program β estimate(s) use the `provisional_virtual` evidence tier (in silico / weak perturbation support); other evidence streams may still be experimental.


## Target Rankings

| Rank | Gene | Ota γ | Tier | OT Score | Max Phase | Key Evidence |
|------|------|-------|------|----------|-----------|--------------|
| 1 | INTS7 | 1.458 | Tier1_Interventional | 0.00 | Phase 0 | — |
| 2 | VEGFA | -1.265 | Tier1_Interventional | 0.00 | Phase 0 | — |
| 3 | FGF2 | 1.210 | Tier1_Interventional | 0.00 | Phase 0 | — |
| 4 | PMEL | 1.203 | Tier1_Interventional | 0.00 | Phase 0 | — |
| 5 | IFI27L1 | -1.162 | Tier1_Interventional | 0.00 | Phase 0 | — |
| 6 | ZMYND19 | -1.137 | Tier1_Interventional | 0.00 | Phase 0 | — |
| 7 | TNFAIP8 | -1.125 | Tier1_Interventional | 0.00 | Phase 0 | — |
| 8 | UBE2H | 1.117 | Tier1_Interventional | 0.00 | Phase 0 | — |
| 9 | RASIP1 | -1.061 | Tier1_Interventional | 0.00 | Phase 0 | — |
| 10 | DGCR8 | -1.058 | Tier1_Interventional | 0.00 | Phase 0 | — |


## Top Target Narratives

INTS7 is a master regulator of P29 (β=0.55). Although no direct GWAS hit, it controls the P29 axis which is genetically causal for CAD (program γ=0.40).
INTS7 upregulates disease-relevant P01, P02, P03 → disease trait.
Evidence: GWAS + Perturb-seq.
Composite Ota estimate: γ_{INTS7→trait} = 1.458 (Tier1_Interventional).
Best compound: no approved drug (no in vitro activity data; Phase 0).

VEGFA is a master regulator of P31 (β=-0.54). Although no direct GWAS hit, it controls the P31 axis which is genetically causal for CAD (program γ=0.63).
VEGFA downregulates disease-promoting P01, P03, P04 → disease trait.
Evidence: GWAS + Perturb-seq.
Composite Ota estimate: γ_{VEGFA→trait} = -1.265 (Tier1_Interventional).
Best compound: no approved drug (no in vitro activity data; Phase 2).

FGF2 is a master regulator of P11 (β=0.60). Although no direct GWAS hit, it controls the P11 axis which is genetically causal for CAD (program γ=0.46).
FGF2 upregulates disease-relevant P02, P03, P04 → disease trait.
Evidence: GWAS + Perturb-seq.
Composite Ota estimate: γ_{FGF2→trait} = 1.210 (Tier1_Interventional).
Best compound: no approved drug (no in vitro activity data; Phase 0).


## Limitations

About 183 gene–program β rows are labelled `provisional_virtual` (in silico or weakly supported cell-model estimates). That label applies to those β edges, not to the entire analysis: GWAS/L2G and other tiers may still be experimentally grounded. Treat virtual-tier rows as hypothesis-generating. Full quantitative Perturb-seq β coverage may require local GEO GSE246756 downloads (~50GB; Replogle 2022) when not cached. Program γ estimates are context-dependent; validate lead findings experimentally.