"""Tables 6-8 of the paper (communication cost, computational cost, training budget).

usage (from system/): python summarize_cost.py [--timing ../results/timing]

Table 6 is computed from the models of this code (NSL-KDD: 122 features, 5 classes, pat partition
with two classes per client). Tables 7 and 8 are computed from the timing runs in results/timing/
(NSL-KDD, pat, seed 0; job file jobs/timing_table7.txt):
  <algo>.json      per-round client training times (groups = rounds; per client: seconds of local training)
  inference.json   per-sample inference latency
"""
import argparse
import json
import os
import statistics as st
from collections import defaultdict

from flcore.trainmodel.models import AE, MLP

ORDER = ['fedavg', 'fedgh', 'fedrep', 'fedhome', 'lg_fedavg', 'fedcorr', 'fednoro', 'fedus']
NAMES = {'fedavg': 'FedAvg', 'fedgh': 'FedGH', 'fedrep': 'FedRep', 'fedhome': 'FedHome', 'lg_fedavg': 'LG-FedAvg',
         'fedcorr': 'FedCorr', 'fednoro': 'FedNoRo', 'fedus': 'FedPURE'}
E = 5  # local epochs per round
L2 = 2500  # FedPURE Phase-2 epochs


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def mb(p):  # 32-bit floats, 1 MB = 2^20 bytes
    return p * 4 / 2 ** 20


def table6():
    ae, mlp = AE(122, 5), MLP(122, 5)
    full, enc, pred, net = n_params(ae), n_params(ae.encoder), n_params(ae.predictor), n_params(mlp)
    proto = 2 * ae.latent_dim  # pat: one prototype per class, two classes per client
    rows = [('fedavg', 'full', full, 'full', full, 51), ('fedgh', 'proto', proto, 'pred', pred, 51),
            ('fedrep', 'encoder', enc, 'encoder', enc, 51), ('fedhome', 'full', full, 'full', full, 51),
            ('lg_fedavg', 'predictor', pred, 'predictor', pred, 51), ('fedcorr', 'full', net, 'full', net, 55),
            ('fednoro', 'full', net, 'full', net, 51)]
    print('Table 6 (NSL-KDD, pat): method | upload | params | download | params | MB/round | total MB')
    for a, cu, u, cd, d, r in rows:
        print(f'  {NAMES[a]:10s} {cu:9s} {u:>7,} {cd:9s} {d:>7,}  {mb(u + d):.5f}  {mb(u + d) * r:.3f}')
    print(f"  {'FedPURE':10s} {'full':9s} {full:>7,} {'full':9s} {full:>7,}  {mb(2 * full):.5f}  "
          f"{mb(full) * (2 * 49 + 1):.3f}   (49 rounds up+down, one final download)")
    print(f'  FedCorr, noisy client (no Stage 2, 45 rounds): {mb(2 * net) * 45:.3f} MB')


def per_epoch(groups, epochs):
    v = [t / epochs for g in groups for t in g.values()]
    return st.mean(v)


def table7_8(tdir):
    T = {}
    for a in ORDER:
        p = os.path.join(tdir, f'{a}.json')
        if os.path.isfile(p):
            T[a] = json.load(open(p))
    inf = {r['algorithm']: r['ms_per_sample'] for r in json.load(open(os.path.join(tdir, 'inference.json')))['runs']}
    print('\nTable 7 (NSL-KDD, pat, seed 0): method | stage | local epoch (s) | training time (s) | infer (ms/sample)')
    for a in ORDER:
        g = T[a]['groups']
        if a == 'fednoro':  # 10 warm-up rounds, then robust rounds (round 10 also contains the transition step)
            stages = [('Stage 1: warm-up (BCE)', per_epoch(g[:10], E)), ('Stage 2: robust (KD)', per_epoch(g[11:], E))]
        elif a == 'fedcorr':  # Stage 1: one client per group; Stage 2: the clean clients; Stage 3: all 20 clients
            stages = [('Stage 1: noise detection', per_epoch([x for x in g if len(x) == 1], E)),
                      ('Stage 2: fine-tune (clean)', per_epoch([x for x in g if 1 < len(x) < 20 and len(x) != 6], E)),
                      ('Stage 3: final training', per_epoch([x for x in g if len(x) == 20], E))]
        elif a == 'fedus':  # 49 Phase-1 rounds, then one Phase-2 group of L2 epochs
            stages = [('Phase 1: AE reconstruction', per_epoch(g[:-1], E)), ('Phase 2: CE fine-tune', per_epoch(g[-1:], L2))]
        else:
            stages = [('--', per_epoch(g, E))]
        for k, (name, v) in enumerate(stages):
            tail = f"{T[a]['parallel_s']:9.1f}  {inf[a]:.4f}" if k == 0 else ''
            print(f'  {NAMES[a] if k == 0 else "":10s} {name:28s} {v:.3f}  {tail}')
    epochs = {'fedcorr': (0, '275 (225 for a client excluded from Stage 2)'), 'fedus': (49 * E, str(L2))}
    print('\nTable 8: method | label-free epochs | supervised epochs | local time per client (s): mean (slowest)')
    for a in ORDER:
        tot = defaultdict(float)
        for grp in T[a]['groups']:
            for c, t in grp.items():
                tot[c] += t
        lf, sup = epochs.get(a, (0, str(51 * E)))
        print(f'  {NAMES[a]:10s} {lf:4d}  {sup:>5s}  {st.mean(tot.values()):.0f} ({max(tot.values()):.0f})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--timing', default='../results/timing')
    a = ap.parse_args()
    table6()
    table7_8(a.timing)
