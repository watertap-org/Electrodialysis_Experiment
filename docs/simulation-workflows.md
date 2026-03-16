# Simulation Workflows

This note summarizes the main process-simulation entry points currently used in
the repository.

## Minimal demo workflows

Primary demo runners:

- `scripts/demos/demo_ossp_proc.py`
- `scripts/demos/demo_kssp_proc_k2.py`

Detailed walkthrough docs:

- `docs/demos/demo_ossp_proc.rst`
- `docs/demos/demo_kssp_proc_k2.rst`

Use these first when you want a compact process-only run (hardcoded feed
condition, YAML-driven model setup) before moving to parquet-driven workflows.

## One-stage single-pass process

Primary runner:

- `scripts/run_proc_ossp.py`

Typical flow:

1. Load an experimental dataset row from parquet.
2. Convert that row into a feed-state object and experiment-specific updates.
3. Build a `OneStageSinglePass` process from YAML.
4. Import scaling and initialization values.
5. Fix operating variables and estimated cation transport numbers.
6. Initialize units sequentially and solve the full flowsheet.

Use this workflow when you want one solved simulation of a single-stage
electrodialysis process under a chosen experimental condition.

## Two-stage single-pass process

Primary runner:

- `scripts/run_proc_tssp.py`

Typical flow:

1. Load an experimental dataset row.
2. Build stage-dependent updates for stages 1 and 2.
3. Assemble a two-stage configuration from the multi-stage YAML definition.
4. Instantiate `KStageSinglePass` with `num_stages = 2`.
5. Import initialization and scaling settings.
6. Fix stage-specific operating variables and transport numbers.
7. Initialize the coupled stages and solve the full flowsheet.

Use this workflow when you want one solved simulation of a coupled two-stage
single-pass process.

## Cross-validation workflow

Primary runner:

- `scripts/run_loocv.py`

This workflow fits the hybrid process model repeatedly under leave-one-out
cross-validation and writes prediction and parameter summaries to
`src/electrodialysis_experiment/data/output/`.

## Notes

- Many workflows read from parquet data in `src/electrodialysis_experiment/data/raw/`.
- Most process runners rely on YAML files in
  `src/electrodialysis_experiment/configs/`.
- Many workflow-generated figures and summaries are written under
  `src/electrodialysis_experiment/data/output/loocv/` or
  `src/electrodialysis_experiment/data/output/loocv_2_grouped/`.
