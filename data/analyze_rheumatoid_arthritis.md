Causal graph analysis of rheumatoid arthritis (EFO: EFO_0000685) identified 2131 significant causal edges spanning 2130 therapeutic targets. Top-ranked targets are PTPRC, MALT1, LMO4. 112 gene–program β estimate(s) use the `provisional_virtual` evidence tier (in silico / weak perturbation support); other evidence streams may still be experimental.


## Target Rankings

| Rank | Gene | Ota γ | Tier | OT Score | Max Phase | Key Evidence |
|------|------|-------|------|----------|-----------|--------------|
| 1 | PTPRC | -8.251 | Tier1_Interventional | 0.33 | Phase 2 | Drug: BC8 131I |
| 2 | MALT1 | -7.084 | Tier1_Interventional | 0.31 | Phase 0 | Drug: MEPAZINE ACETATE |
| 3 | LMO4 | 6.886 | Tier2_eQTL_direction | 0.33 | Phase 0 | — |
| 4 | PARK7 | 6.877 | Tier1_Interventional | 0.36 | Phase 0 | — |
| 5 | RTKN2 | 6.767 | Tier1_Interventional | 0.49 | Phase 0 | — |
| 6 | SMAD3 | 6.209 | Tier1_Interventional | 0.36 | Phase 0 | — |
| 7 | RAD51B | -6.001 | Tier1_Interventional | 0.43 | Phase 0 | — |
| 8 | SH3PXD2A | -5.999 | Tier1_Interventional | 0.26 | Phase 0 | — |
| 9 | BBS9 | -5.655 | Tier1_Interventional | 0.25 | Phase 0 | — |
| 10 | CSF2RB | 5.591 | Tier1_Interventional | 0.34 | Phase 4 | Drug: CIBINETIDE, SARGRAMOSTIM, MAVRILIMUMAB |


## Top Target Narratives

PTPRC downregulates disease-promoting RA_SVD_C17, RA_SVD_C09, RA_SVD_C15 → disease trait.
Evidence: Drug: BC8 131I.
Composite Ota estimate: γ_{PTPRC→trait} = -8.251 (Tier1_Interventional).
Best compound: BC8 131I (no in vitro activity data; Phase 2).

MALT1 downregulates disease-promoting RA_SVD_C17, RA_SVD_C09, RA_SVD_C15 → disease trait.
Evidence: Drug: MEPAZINE ACETATE.
Composite Ota estimate: γ_{MALT1→trait} = -7.084 (Tier1_Interventional).
Best compound: MEPAZINE ACETATE (no in vitro activity data; Phase 0).

LMO4 upregulates disease-relevant RA_SVD_C17, RA_SVD_C09, RA_SVD_C15 → disease trait.
Evidence: GWAS + Perturb-seq.
Composite Ota estimate: γ_{LMO4→trait} = 6.886 (Tier2_eQTL_direction).
Best compound: no approved drug (no in vitro activity data; Phase 0).


## Limitations

About 112 gene–program β rows are labelled `provisional_virtual` (in silico or weakly supported cell-model estimates). That label applies to those β edges, not to the entire analysis: GWAS/L2G and other tiers may still be experimentally grounded. Treat virtual-tier rows as hypothesis-generating. Full quantitative Perturb-seq β coverage may require local GEO GSE246756 downloads (~50GB; Replogle 2022) when not cached. Program γ estimates are context-dependent; validate lead findings experimentally.