# FedPURE: Pretrained Unsupervised Representation Encoder for Intrusion Detection under Non-IID Data and Noisy Labels

Code and run logs for the paper. This repository contains only the code that produced the reported results and the runs that the tables and figures are computed from.

In the code and in the result folders, **FedPURE has the algorithm id `fedus`**.

## Contents

```
dataset/                  scripts that build the client partitions from the raw data
  generate_<dataset>.py   preprocessing and partitioning (one per dataset)
  make_partitions.py      builds every partition used in the paper
  utils/                  partitioning helpers (pat, exDir)
system/
  main.py                 entry point: one run = one method, dataset, setting and seed
  flcore/                 servers and clients of all methods
  utils/                  data loading
  run_parallel.py         runs a job file (several runs in parallel)
  jobs/                   the job file of every table and figure
  summarize_tables.py     recomputes Tables 3-5 from the run logs
  summarize_cost.py       recomputes Tables 6-8 (model sizes and results/timing/)
  summarize_figures.py    recomputes the numbers of Figures 4-11
  budget_ft.py            extra local fine-tuning of FedRep (Figure 4)
  latejoin_per_client.py  per-client F1 of the late-joining client runs (Figure 5)
results/
  table_runs.csv          table entry -> run directory, for every seed
  <dataset>/<partition>/<method>/<experiment>/seed_<s>/
      tracking.log        full log of the run; the score is in "Best performance metrics"
      per_client_f1.json  late-joining client runs only (Figure 5)
  budget_ft/, budget_l2_logs/   Figure 4
  timing/                 Tables 7 and 8
requirements.txt
```

## 1. Environment

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

All runs use the CPU (`-dev cpu`).

## 2. Data

The raw datasets are not redistributed here. Download them from the official sources and put the files in `dataset/<id>/rawdata/`:

| Dataset | id | Source |
|---|---|---|
| NSL-KDD | `nslkdd` | https://www.unb.ca/cic/datasets/nsl.html |
| UNSW-NB15 | `unsw` | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| CIC-IDS2017 | `cic_ids` | https://www.unb.ca/cic/datasets/ids-2017.html |
| TON_IoT | `toniot` | https://research.unsw.edu.au/projects/toniot-datasets |

Then build the partitions (20 clients; `pat` with two classes per client and `exDir` with alpha = 0.1; partition seed 1):

```bash
cd dataset
python make_partitions.py nslkdd unsw cic_ids toniot
```

This writes `dataset/<id>/noniid20/` (`pat`), `dataset/<id>/noniidexdir20/` (`exDir`) and `dataset/<id>/centralized/`. The partition is fixed by the partition seed; the run seed (0, 1, 2) only changes model initialization and training.

## 3. Check the numbers of the paper (no training needed)

```bash
cd system
python summarize_tables.py
```

For every entry of Tables 3-5, the script reads the score of each seed from the `tracking.log` of the run listed in `results/table_runs.csv`, checks it against the value in the CSV, and prints the mean and standard deviation over the three seeds. The score of a run is the macro-F1 of the "Best performance metrics" block of its log (benign clients only in Tables 4 and 5).

Tables 6-8 and Figures 4-11 are checked in the same way:

```bash
python summarize_cost.py      # Table 6 (NSL-KDD models), Tables 7 and 8 (results/timing/)
python summarize_figures.py   # Figure 4 (budget), Figure 5 (late-joining client), Figures 6-11 (ablation)
```

- **Table 6** is computed from the models of this code on NSL-KDD (`pat`, two classes per client), with 32-bit floats and 1 MB = 2^20 bytes.
- **Tables 7 and 8** are computed from the timing runs in `results/timing/` (NSL-KDD, `pat`, seed 0), which record the local training time of every client in every round and the inference latency.
- **Figure 4**: FedPURE at L2 = 50-1000 from the Phase-2 milestones in `results/budget_l2_logs/`, at L2 = 2500 from the Table 3/4 runs; FedRep from the per-client F1 written by `budget_ft.py` in `results/budget_ft/`.
- **Figure 5**: the run log prints per-client F1 with two decimals only, so `latejoin_per_client.py` re-evaluates the client models of the reported round (from `models.pkl`, which every run writes but which is not included here) and writes `per_client_f1.json` (included) for each seed.
- **Figures 6-11**: the score of each ablation run log (seed 0) and of FedPURE (seed 0 of Tables 3 and 4).

## 4. Reproduce one table entry

Example: FedPURE on NSL-KDD, `pat` partition, clean labels, seed 0 (Table 3):

```bash
cd system
python main.py -algo fedus -data nslkdd -dt noniid -dev cpu -nt none -npw 0 -psr 0 --seed 0 --exp_name r2_table3
```

The log is written to `results/nslkdd/noniid20/fedus/r2_table3_noniid20_clean/seed_0/tracking.log`.

Main options:

| Option | Meaning |
|---|---|
| `-algo` | `fedavg`, `fedgh`, `fedhome`, `lg_fedavg`, `fedrep`, `fednoro`, `fedcorr`, `fedus` (FedPURE), `centralized` |
| `-data` | `nslkdd`, `unsw`, `cic_ids`, `toniot` |
| `-dt` | `noniid` (`pat`) or `exdir` (`exDir`) |
| `-nt`, `-npw`, `-psr` | noise: `none 0 0` (Table 3); `label_flipping 0.3/0.5 1.0` (Table 4, r_c = 0.3/0.5); `Asymmetric` or `Symmetric 0.3 0.4` (Table 5) |
| `--seed` | 0, 1, 2 |
| `--fedcorr_eval` | FedCorr only; `global` (default, as in the original FedCorr): every client is evaluated with the global model |
| `--ablation` | FedPURE only (Section 6.5): `exchange_all`, `head_only`, `single_phase`, `joint_loss` |

Other hyperparameters (20 clients, 5 local epochs per round, batch size, learning rate, rounds, and the 2,500 Phase-2 epochs of FedPURE) are the defaults of `main.py`; the values are listed in Table 2 of the paper.

## 5. Reproduce a whole table or figure

Every table and figure has a job file in `system/jobs/` (one line per run: `name | priority | arguments`):

| Job file | Paper |
|---|---|
| `table3.txt`, `table4.txt`, `table5.txt` | Tables 3-5 (8 methods x 4 datasets x 2 settings x 3 seeds; Table 3 also Centralized) |
| `budget_fedpure_fig4.txt`, `budget_fedrep_fig4.txt` | Figure 4 (supervised budget, NSL-KDD) |
| `latejoin_fig5.txt` | Figure 5 (late-joining client, NSL-KDD) |
| `ablation_fig6-11.txt` | Figures 6-11 (ablation, seed 0) |
| `timing_table7.txt` | Tables 7 and 8 (time per local epoch, local training time, inference) |

```bash
cd system
python run_parallel.py --jobs jobs/table3.txt --max-jobs 8 --log-dir logs/table3
```

Runs whose log folder already has a `<name>.done` file are skipped, so the command can be restarted. One run takes about 1-3 hours on one CPU core, so a full table takes several days on one machine.

The Figure 4 jobs start from finished Table 3/4 runs (`--resume_from`), and `budget_ft.py` needs the model files of those runs; run the corresponding Table 3/4 jobs first.

## 6. Notes on reproducibility

- **Small differences between runs.** Training runs on the CPU with several processes in parallel, and floating-point operations are not bit-exact between runs. Re-running an entry gives a very close but not always identical F1 (for example, two runs of FedPURE on CIC-IDS17, `pat`, seed 0, gave 0.9748 and 0.9753).
- **Run directories.** Some runs of Tables 3-5 were made before the experiment names were unified, so their folders are named by start time (e.g. `20261111-090615_noniid20_clean`). The settings of every run are printed at the top of its `tracking.log`, and `results/table_runs.csv` gives the run of every table entry.
- **Per-client results.** For all federated methods except FedCorr, `tracking.log` lists the accuracy, precision, recall, and macro-F1 of every client in every round (two decimals; the weighted mean recomputed from them matches the seed score to within 0.004). The FedCorr implementation logs only the weighted mean over the clients in each round.
- **What is stored per run.** Only `tracking.log` is kept: it contains every metric reported in the paper. A new run also writes `metrics_results.pkl` and model files; these are not included here and are not needed to check or retrain any entry.
- **Model files.** The model files (`best_model.pt`) are not included; every run can be retrained with the job files in `system/jobs/`.
