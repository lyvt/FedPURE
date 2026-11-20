"""Numbers plotted in Figures 4-11 of the paper.

usage (from system/): python summarize_figures.py

Figure 4  budget-matched comparison on NSL-KDD (pat and r_c=0.5 flip), mean +- std over 3 seeds:
          FedPURE at L2 = 50..1000 from results/budget_l2_logs (milestones of Phase 2), at L2 = 2500
          from the Table 3/4 runs; FedRep after E extra local epochs from results/budget_ft (per-client
          F1 written by budget_ft.py), weighted by the number of test samples of each client.
Figure 5  late-joining client (NSL-KDD, pat): F1 of the late client and weighted F1 of the 19 original
          clients, from results/.../r2_latejoin_noniid20_clean/seed_<s>/per_client_f1.json
          (written by latejoin_per_client.py from the model files of the run).
Figs 6-11 ablation, seed 0: score of each run log (results/.../fedus/r2_abl_<variant>_<setting>/seed_0)
          and of FedPURE (seed 0 of Tables 3 and 4).
"""
import csv
import glob
import json
import re
import statistics as st

R = '../results'
D = ['nslkdd', 'unsw', 'cic_ids', 'toniot']


def best_f1(log):
    m = re.search(r'Best performance metrics.*?f1-normal:\s*([0-9.]+)', open(log).read(), re.S)
    return float(m.group(1))


def ms(v, rounded=False):
    """mean +- sample std over seeds; rounded=True first rounds each seed score to 4 decimals
    (as for the per-seed values listed in the Supplementary Material, e.g. Figure 5)."""
    if rounded:
        v = [round(x, 4) for x in v]
    return f'{st.mean(v):.4f} +- {st.stdev(v):.4f}'


def table_runs():
    return {(r['table'], r['setting'], r['dataset'], r['method'], r['seed']): float(r['f1'])
            for r in csv.DictReader(open(f'{R}/table_runs.csv'))}


def figure4(T):
    print('Figure 4 (NSL-KDD): budget | FedPURE (L2) | FedRep (+E local epochs)')
    for tag, setting, table, key in [('pat', 'r2_table3_noniid20_clean', 'Table 3', 'pat'),
                                     ('rc05', 'r2_table4_noniid20_label_flipping_npw0.5_psr1.0', 'Table 4', 'r_c=0.5')]:
        print(f'  {key}')
        for e in ['50', '100', '250', '500', '1000', '2500']:
            if e == '2500':
                p = [T[(table, key, 'nslkdd', 'fedus', str(s))] for s in range(3)]
            else:
                p = [float(re.search(rf'L2 milestone {e}: f1-normal ([0-9.]+)',
                                     open(f'{R}/budget_l2_logs/budget_l2_fedus_{tag}_s{s}.log').read()).group(1))
                     for s in range(3)]
            q = []
            for s in range(3):
                num = den = 0.0
                for f in glob.glob(f'{R}/budget_ft/fedrep__{setting}__seed{s}__c*.json'):
                    for c in json.load(open(f))['clients'].values():
                        num += c['f1'][e] * c['n_test']
                        den += c['n_test']
                q.append(num / den)
            print(f'    {e:>5s}  FedPURE {ms(p)}   FedRep {ms(q)}')


def figure5():
    late, orig = [], []
    for s in range(3):
        d = json.load(open(f'{R}/nslkdd/noniid20/fedus/r2_latejoin_noniid20_clean/seed_{s}/per_client_f1.json'))
        c = d['clients']
        late.append(c[str(d['late_client'])]['f1'])
        o = [v for k, v in c.items() if k != str(d['late_client'])]
        orig.append(sum(v['f1'] * v['n_test'] for v in o) / sum(v['n_test'] for v in o))
    print('\nFigure 5 (NSL-KDD, pat, FedPURE):')
    print(f'  late-joining client       {ms(late, True)}   seeds {[round(x, 4) for x in late]}')
    print(f'  original clients (19)     {ms(orig, True)}   seeds {[round(x, 4) for x in orig]}')


def figures6_11(T):
    print('\nFigures 6-11 (seed 0): setting | variant | NSL-KDD UNSW-NB15 CIC-IDS17 TON_IoT')
    for part, sc, name, table, key in [('noniid20', 'clean', 'pat', 'Table 3', 'pat'),
                                       ('noniidexdir20', 'clean', 'exDir', 'Table 3', 'exDir'),
                                       ('noniid20', 'label_flipping_npw0.3_psr1.0', 'r_c=0.3', 'Table 4', 'r_c=0.3'),
                                       ('noniid20', 'label_flipping_npw0.5_psr1.0', 'r_c=0.5', 'Table 4', 'r_c=0.5')]:
        print(f'  {name:8s} FedPURE       ' + '  '.join(f"{T[(table, key, d, 'fedus', '0')]:.4f}" for d in D))
        for v in ['exchange_all', 'head_only', 'single_phase', 'joint_loss']:
            vals = [best_f1(f'{R}/{d}/{part}/fedus/r2_abl_{v}_{part}_{sc}/seed_0/tracking.log') for d in D]
            print(f'  {"":8s} {v:13s} ' + '  '.join(f'{x:.4f}' for x in vals))


if __name__ == '__main__':
    T = table_runs()
    figure4(T)
    figure5()
    figures6_11(T)
