"""Equal-budget check (b): local fine-tuning of a finished baseline run.

Takes the final-round models saved in <run_dir>/models.pkl ([global, client_0, ..., client_{N-1}])
and fine-tunes, for every benign client, the model that client is evaluated with, for up to
--epochs local epochs with the same recipe as FedPURE's phase 2 (cross-entropy, plain SGD with the
run's local learning rate, batch size from the run, shuffle=False, drop_last=True). Test macro-F1
is recorded at each milestone; f1-normal = sample-weighted mean over benign clients, as in the
main tables. Milestone 0 is the model before fine-tuning.

Nothing is retrained federatedly: the federated part is reused from the saved run.

usage: python budget_ft.py --run_dir ../results/<ds>/<partition>/<algo>/<setting>/seed_<k>
                           [--clients 3,4] [--epochs 2500] [--milestones 50,100,250,500,1000,2500]
Writes <out_dir>/<algo>__<setting>__seed<k>__c<ids>.json
"""
import argparse, copy, json, os, pickle, sys, time

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_utils import read_client_data  # noqa: E402


def logits(model, x):
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


def test_f1(model, loader):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for x, y in loader:
            yp.append(torch.argmax(logits(model, x), dim=1).numpy())
            yt.append(y.numpy())
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return metrics.f1_score(yt, yp, average='macro'), len(yt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--clients', default='', help='comma-separated benign client ids (default: all benign)')
    ap.add_argument('--epochs', type=int, default=2500)
    ap.add_argument('--milestones', default='50,100,250,500,1000,2500')
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--num_clients', type=int, default=20)
    ap.add_argument('--out_dir', default='../results/budget_ft')
    a = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)

    meta = pickle.load(open(os.path.join(a.run_dir, 'metrics_results.pkl'), 'rb'))
    ds, dt, algo = meta['dataset'], meta['data_type'], meta['algorithm']
    poisoned = set(meta.get('poisoned_worker_ids') or [])
    models = pickle.load(open(os.path.join(a.run_dir, 'models.pkl'), 'rb'))
    final = models[max(models)]                       # [global, client_0, ...]
    benign = [i for i in range(a.num_clients) if i not in poisoned]
    todo = [int(c) for c in a.clients.split(',') if c.strip()] or benign
    todo = [c for c in todo if c in benign]
    ms = sorted({int(m) for m in a.milestones.split(',') if m.strip()} | {0})
    ms = [m for m in ms if m <= a.epochs]

    # which saved model each client is evaluated with: the per-client model if it reproduces the
    # run's last-round f1-normal better than the global one (FedRep: personal model; FedCorr: global)
    test = {i: DataLoader(read_client_data(ds, i, a.num_clients, data_type='test', type=dt),
                          batch_size=256, shuffle=False) for i in benign}

    def weighted(pick):
        num = den = 0.0
        for i in benign:
            f1, n = test_f1(pick(i), test[i]); num += f1 * n; den += n
        return num / den
    logged = meta['f1-normal'][-1]
    f_glob = weighted(lambda i: final[0])
    f_pers = weighted(lambda i: final[1 + i])
    use_personal = abs(f_pers - logged) <= abs(f_glob - logged)
    src = 'client' if use_personal else 'global'
    print(f"{algo} {a.run_dir}: logged last-round f1-normal {logged:.4f}; global {f_glob:.4f}, "
          f"per-client {f_pers:.4f} -> fine-tune the {src} model", flush=True)

    res = {'run_dir': os.path.abspath(a.run_dir), 'algo': algo, 'dataset': ds, 'data_type': dt,
           'setting': meta.get('exp_name'), 'seed': meta.get('seed'), 'start_model': src,
           'logged_last_f1_normal': logged, 'logged_best_f1_normal': max(meta['f1-normal']),
           'start_f1_global': f_glob, 'start_f1_client': f_pers, 'epochs': a.epochs,
           'batch_size': a.batch_size, 'lr': a.lr, 'clients': {}}
    for cid in todo:
        t0 = time.time()
        model = copy.deepcopy(final[1 + cid] if use_personal else final[0])
        for p in model.parameters():
            p.requires_grad = True
        tr = read_client_data(ds, cid, a.num_clients, data_type='train', type=dt)
        loader = DataLoader(tr, min(a.batch_size, len(tr)), drop_last=True, shuffle=False)
        opt = torch.optim.SGD(model.parameters(), lr=a.lr)
        ce = nn.CrossEntropyLoss()
        curve = {}
        f1, n = test_f1(model, test[cid]); curve[0] = f1
        for ep in range(1, a.epochs + 1):
            model.train()
            for x, y in loader:
                opt.zero_grad()
                ce(logits(model, x), y.long()).backward()
                opt.step()
            if ep in ms:
                curve[ep] = test_f1(model, test[cid])[0]
        res['clients'][cid] = {'n_test': n, 'f1': curve, 'minutes': (time.time() - t0) / 60}
        print(f"  client {cid}: " + ", ".join(f"{k}:{v:.4f}" for k, v in curve.items())
              + f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    os.makedirs(a.out_dir, exist_ok=True)
    tag = f"{algo}__{meta.get('exp_name')}__seed{meta.get('seed')}__c{'-'.join(map(str, todo))}"
    json.dump(res, open(os.path.join(a.out_dir, tag + '.json'), 'w'), indent=1)
    print('wrote', os.path.join(a.out_dir, tag + '.json'))


if __name__ == '__main__':
    main()
