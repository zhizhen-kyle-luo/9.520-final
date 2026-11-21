import torch, json, sys, os, itertools, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from train import LOSS_GRID, SEEDS, SEEDS_LR2

OUT = sys.argv[1] if len(sys.argv) > 1 else 'runs'
OPTS = ['sgd', 'adam', 'muon']
NAME = {'sgd': 'SGD', 'adam': 'Adam', 'muon': 'Muon'}
COLOR = {'sgd': 'tab:blue', 'adam': 'tab:orange', 'muon': 'tab:green'}
MAX_MISMATCH = 0.03   # a target is used only if every run's save is within this many nats of it
LRS = json.load(open(f'{OUT}/lrs.json'))

def load(o, which, s):
    path = f'{OUT}/{o}_lr{LRS[o][which]}_seed{s}'
    if not os.path.exists(path + '.json'): return None
    return json.load(open(path + '.json')), torch.load(path + '_logprobs.pt', weights_only=True)

# a run is keyed (optimizer, 'best' or 'second', seed); missing runs are skipped so partial results can be plotted
runs = {(o, w, s): r for o in OPTS for w, seeds in [('best', SEEDS), ('second', SEEDS_LR2)] for s in seeds
        if (r := load(o, w, s)) is not None}
main = [k for k in runs if k[1] == 'best']
seeds = sorted({k[2] for k in main if all((o, 'best', k[2]) in runs for o in OPTS)})
print('runs loaded', len(runs), 'seeds with all three optimizers', seeds)

def js(a, b):
    # jensen-shannon divergence in nats between next-token distributions, mean over held-out tokens
    a, b = a.float(), b.float()
    m = torch.logsumexp(torch.stack([a, b]), 0) - np.log(2)
    return (0.5 * (a.exp() * (a - m)).sum(-1) + 0.5 * (b.exp() * (b - m)).sum(-1)).mean().item()
def top1(a, b):
    return (a.argmax(-1) != b.argmax(-1)).float().mean().item()

# keep targets every run reached with a save close to the target; early targets fail this for fast runs,
# because muon can drop more than 0.1 nats in a single step there
targets = []
for j, t in enumerate(LOSS_GRID):
    saves = [runs[k][0]['grid_save'][j] for k in runs]
    if min(saves) < 0: continue
    if max(t - runs[k][0]['save_train'][runs[k][0]['grid_save'][j]] for k in runs) <= MAX_MISMATCH: targets.append(j)
print(f'targets used {len(targets)}, loss {LOSS_GRID[targets[0]]} to {LOSS_GRID[targets[-1]]}')

def at(k, j): return runs[k][1][runs[k][0]['grid_save'][j]]

def mean_dist(f, j, pairs): return np.mean([f(at(a, j), at(b, j)) for a, b in pairs])

def permutation_p(f, j, o1, o2):
    """p-value that optimizers o1 and o2 differ beyond seed noise at grid target j
    statistic is mean distance over different-seed pairs across optimizers minus mean over different-seed pairs
    within an optimizer. seed k of both optimizers shares init and data, so labels are only swapped inside each
    seed pair, over all 2^n swaps"""
    ks = [(o1, 'best', s) for s in seeds] + [(o2, 'best', s) for s in seeds]
    n = len(seeds)
    D = np.zeros((2 * n, 2 * n))
    for a, b in itertools.combinations(range(2 * n), 2): D[a, b] = D[b, a] = f(at(ks[a], j), at(ks[b], j))
    seed_of = np.array(list(range(n)) * 2)
    diff_seed = seed_of[:, None] != seed_of[None, :]
    swaps = np.array(list(itertools.product([0, 1], repeat=n)), dtype=bool)   # 2^n x n
    group = np.concatenate([swaps, ~swaps], axis=1)                            # 2^n x 2n, row 0 is the true labels
    same = group[:, :, None] == group[:, None, :]
    within = (D * (same & diff_seed)).sum((1, 2)) / (same & diff_seed).sum((1, 2))
    between = (D * (~same & diff_seed)).sum((1, 2)) / (~same & diff_seed).sum((1, 2))
    gap = between - within
    return (gap >= gap[0]).mean()

def curves(f):
    """for one distance f, the curves plotted and printed: per optimizer pair the between mean, per optimizer the
    within mean and the lr control mean, all over different-seed pairs only, plus the permutation p-values"""
    c = {'between': {}, 'within': {}, 'lr': {}, 'p': {}}
    for o1, o2 in itertools.combinations(OPTS, 2):
        pairs = [((o1, 'best', a), (o2, 'best', b)) for a in seeds for b in seeds if a != b]
        c['between'][o1, o2] = [mean_dist(f, j, pairs) for j in targets]
        c['p'][o1, o2] = [permutation_p(f, j, o1, o2) for j in targets]
    for o in OPTS:
        c['within'][o] = [mean_dist(f, j, [((o, 'best', a), (o, 'best', b)) for a, b in itertools.combinations(seeds, 2)])
                          for j in targets]
        lr_pairs = [((o, 'best', a), (o, 'second', b)) for a in SEEDS_LR2 for b in SEEDS_LR2
                    if a != b and (o, 'second', b) in runs and (o, 'best', a) in runs]
        if lr_pairs: c['lr'][o] = [mean_dist(f, j, lr_pairs) for j in targets]
    return c

x = [LOSS_GRID[j] for j in targets]
res = {'js': curves(js), 'top1': curves(top1)}

# drawn at printed size (full page width, 7 in) so text is not shrunk in the pdf
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9, 'legend.fontsize': 8})
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.9))
for k in main:
    log = runs[k][0]
    ax[0].plot(log['step'], log['train'], color=COLOR[k[0]], lw=0.6, label=NAME[k[0]] if k[2] == seeds[0] else None)
    ax[0].plot(log['step'], log['val'], color=COLOR[k[0]], ls='--', lw=0.6, alpha=0.5)
ax[0].set(xlabel='step', ylabel='loss (nats)', title='(a) loss', ylim=(0, 4.3)); ax[0].legend()
for a, (key, title) in zip(ax[1:], [('js', '(b) JS divergence (nats)'), ('top1', '(c) top guess disagreement')]):
    c = res[key]
    w = np.array(list(c['within'].values()))
    a.fill_between(x, w.min(0), w.max(0), color='gray', alpha=0.3, label='same optimizer')
    for o, y in c['lr'].items(): a.plot(x, y, color=COLOR[o], ls=':', lw=1, label=f'{NAME[o]} two rates')
    for (o1, o2), y in c['between'].items():
        line, = a.plot(x, y, lw=1.2, label=f'{NAME[o1]} vs {NAME[o2]}')
        sig = np.array(c['p'][o1, o2]) < 0.05
        a.plot(np.array(x)[sig], np.array(y)[sig], '.', color=line.get_color(), ms=3)
    a.set(xlabel='matched train loss (nats)', title=title); a.invert_xaxis()
fig.legend(*ax[1].get_legend_handles_labels(), loc='lower center', ncol=4, fontsize=7, frameon=False)
plt.tight_layout(pad=0.3, rect=(0, 0.13, 1, 1)); plt.savefig(f'{OUT}/figure.pdf'); plt.savefig(f'{OUT}/figure.png', dpi=300)

# numbers for the write-up
print('lrs', LRS)
mism = [LOSS_GRID[j] - runs[k][0]['save_train'][runs[k][0]['grid_save'][j]] for j in targets for k in runs]
print(f'largest loss mismatch at used targets {max(mism):.4f}, mean {np.mean(mism):.4f}')
for key in res:
    c = res[key]; print(key)
    for i, j in enumerate(targets):
        print(f'  loss {LOSS_GRID[j]:.2f}  within ' + ' '.join(f'{o[0]}={c["within"][o][i]:.4f}' for o in OPTS)
              + '  lr ' + ' '.join(f'{o[0]}={c["lr"][o][i]:.4f}' for o in c['lr'])
              + '  between ' + ' '.join(f'{a[0]}{b[0]}={c["between"][a, b][i]:.4f}(p={c["p"][a, b][i]:.3f})' for a, b in c['between']))
