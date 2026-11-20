"""Per-client macro-F1 of the late-joining client runs (Figure 5).

The run log prints per-client F1 with two decimals only. This script re-evaluates the client
models of the reported round (the best round of the run) on each client's test set and writes
<run_dir>/per_client_f1.json, from which Figure 5 is computed (see summarize_figures.py).
It needs models.pkl, which main.py writes for every run but which is not included in the repository.

usage (from system/): python latejoin_per_client.py [--run ../results/nslkdd/noniid20/fedus/r2_latejoin_noniid20_clean]
"""
import argparse
import json
import os
import pickle

import torch
from torch.utils.data import DataLoader

from budget_ft import test_f1
from utils.data_utils import read_client_data

LATE_CLIENT = 19  # the client that joins after Phase 1 (the last client)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='../results/nslkdd/noniid20/fedus/r2_latejoin_noniid20_clean')
    ap.add_argument('--num_clients', type=int, default=20)
    a = ap.parse_args()
    torch.set_num_threads(1)
    for s in (0, 1, 2):
        d = f'{a.run}/seed_{s}'
        meta = pickle.load(open(f'{d}/metrics_results.pkl', 'rb'))
        models = pickle.load(open(f'{d}/models.pkl', 'rb'))
        br = meta['best_round']
        res = {}
        for i in range(a.num_clients):
            dl = DataLoader(read_client_data('nslkdd', i, a.num_clients, data_type='test', type='noniid20'),
                            batch_size=256)
            f1, n = test_f1(models[br][1 + i], dl)
            res[str(i)] = {'f1': f1, 'n_test': n}
        out = {'run_dir': d.split('results/', 1)[-1], 'seed': s, 'round': br, 'late_client': LATE_CLIENT,
               'logged_best_f1_normal': meta['best_f1_normal'], 'clients': res}
        json.dump(out, open(f'{d}/per_client_f1.json', 'w'), indent=1)
        tot = sum(c['f1'] * c['n_test'] for c in res.values()) / sum(c['n_test'] for c in res.values())
        print(f"seed {s}: round {br}, logged {meta['best_f1_normal']:.4f}, recomputed (20 clients) {tot:.4f}, "
              f"late client {res[str(LATE_CLIENT)]['f1']:.4f}")


if __name__ == '__main__':
    main()
