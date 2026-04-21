"""
pipelines/state_space — Dynamical state-space models for disease cell biology.

Implements the state-space scoring layer of the OTA pipeline:
  - Latent NMF program model (latent_model.py)
  - Disease cell-state definition from h5ad (state_definition.py)
  - State transition graph and pseudotime scoring (transition_graph.py)
  - Therapeutic redirection per target gene (therapeutic_redirection.py)
  - Disease-axis influence score per gene (state_influence.py)

Currently validated on AMD (RPE h5ad, cNMF k=12) and CAD (SMC h5ad, cNMF k=8).

Usage:
    from pipelines.state_space.latent_model import build_disease_latent_space
    from pipelines.state_space.state_definition import define_cell_states
    from pipelines.state_space.transition_graph import infer_state_transition_graph
"""
