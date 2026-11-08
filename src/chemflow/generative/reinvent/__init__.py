"""REINVENT4 integration for ChemFlow.

A thin, config-driven wrapper: a ChemFlow YAML is validated and translated to a
REINVENT4 TOML, and the external ``reinvent`` executable (installed in its own
conda environment, see ``environment-reinvent.yaml``) is invoked. Nothing in
this package imports REINVENT4 or torch, so config / validate / stage /
dry-run work without it installed.
"""
