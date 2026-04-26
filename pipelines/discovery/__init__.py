"""
pipelines/discovery — Data-driven program and γ discovery.

This package replaces pre-specified program registries and hardcoded γ values
with computations from real genomic data:

  1. cellxgene_downloader.py  — download sc-RNA from CELLxGENE census
  2. cnmf_runner.py           — run NMF on sc-RNA to discover programs de novo
"""
from __future__ import annotations

from pipelines.discovery.cellxgene_downloader import download_disease_scrna
from pipelines.discovery.cnmf_runner import run_nmf_programs, load_computed_programs
