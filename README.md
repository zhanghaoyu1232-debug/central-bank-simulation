# Dynamic Interbank Network Simulation

This repository contains the executable program and reproduction workflow for
*Dynamic Interbank Network Formation and Systemic Risk*. It compares daily
centralized, market-wide rate matching with daily decentralized local RFQ
matching under common balance-sheet, project, settlement, rollover, and policy
rules.

## Archived thesis version

After the verified formal rerun, tag the thesis release as
`thesis-v3-revision-20260907`. Every result JSON
stores `artifact_schema_version`, `code_fingerprint`, realized seeds, feature
switches, matcher metadata, and run lengths. The comparison program rejects an
artifact whose fingerprint does not match the checked-out source.

## Environment

The reported runs used Windows 11, Python 3.10.11, NumPy 1.26.4, and
Matplotlib 3.10.1 on an Intel Core i5-12600KF, 32 GB RAM, and an NVIDIA
GeForce RTX 4060 Ti (8 GB). Install the remaining packages listed in
`requirements.txt` in the same Python environment.

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Formal reproduction

Place the trained local matcher at
`输入/gnn_pair_matcher_v6_local.pth` and its v6 input data at
`输入/bank_contagion_data_local_v6.json`. From the repository root run:

```powershell
python compare_decentralized_centralized.py --T 1000 --nsim 20 `
  --seed-root 42 --seed-namespace FORMAL --stop-alive-threshold 1 `
  --den-matcher-mode load --cen-matcher-mode off `
  --all-scenarios --force-rerun
```

This command regenerates the four rollover/support configurations and their
comparison outputs. The measurement sweeps reuse recorded states; they do not
rerun the financial system. The single-loan-cap robustness check reruns the
economic simulation at `B = 600, 1200, 1800`.

For figure-only regeneration after artifact validation:

```powershell
python compare_decentralized_centralized.py --T 1000 --nsim 20 `
  --seed-root 42 --seed-namespace FORMAL --stop-alive-threshold 1 `
  --den-matcher-mode load --cen-matcher-mode off `
  --all-scenarios --skip-run
```

Figures 7.8--7.9 can also be rebuilt without importing the simulation models:

```powershell
python scripts/rebuild_thesis_four_scenario_figures.py `
  --artifact-root 输出/figures --output-dir thesis_figures
```

## Formal paired seeds

The 20 ON/ON paired seeds are:

```text
576266650, 3288100970, 2598636728, 189077467, 3356860510,
730482077, 283194794, 3363951558, 235919858, 1558245143,
3381681453, 3716205366, 142333896, 4207354830, 742397554,
614242082, 1193124925, 1943491583, 3647799003, 1076140681
```

The order above is the realized execution order. Centralized and RFQ runs use
the same seed at each paired position.

## Files and outputs

- `bank_simulation_model_decentralized_central_policy.py`: decentralized RFQ
  simulator.
- `bank_simulation_model_centralized_central_policy.py`: centralized simulator.
- `interbank_installment_rollover.py`: contract schedules and one-system EN
  settlement.
- `interbank_resolution.py`: absorbing default, estate transfer, creditor
  recovery, and write-offs.
- `bank_regulatory.py`: common RWA and CAR definitions.
- `bank_econ_shared.py`: common economic, risk, and sweep helpers.
- `interbank_intentions.py` and `interbank_matcher_shared.py`: intentions and
  shared matching features.
- `compare_decentralized_centralized.py`: Monte Carlo orchestration, paired
  statistics, tables, and figures.
- `输出/figures/**/compare_artifacts.json`: raw per-replication paths and
  metadata used by the reported tables and figures.

Large input and raw-output files should be stored with Git LFS. Do not replace
them with manually edited summaries: the JSON artifacts are the source for
reported values.

## Verification

Before using cached outputs, compare the value printed by:

```powershell
python -c "from output_paths import simulation_code_fingerprint; print(simulation_code_fingerprint())"
```

with each artifact's `code_fingerprint`. A mismatch means that the simulations
must be rerun with `--force-rerun` before the thesis numbers are considered
final.
