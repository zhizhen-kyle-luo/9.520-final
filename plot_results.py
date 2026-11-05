import torch, json, sys, itertools, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else 'runs'
OPTS, SEEDS = ['sgd', 'adam', 'muon'], [0, 1]
COLOR = {'sgd': 'tab:blue', 'adam': 'tab:orange', 'muon': 'tab:green'}
BEST = json.load(open(f'{OUT}/best_lr.json'))
runs = {(o, s): (json.load(open(f'{OUT}/{o}_lr{BEST[o]}_seed{s}.json')),
                 torch.load(f'{OUT}/{o}_lr{BEST[o]}_seed{s}_logprobs.pt', weights_only=True).float())
        for o in OPTS for s in SEEDS}

def js(a, b):
    # jensen-shannon divergence in nats between next-token distributions, mean over held-out tokens
    m = torch.logsumexp(torch.stack([a, b]), 0) - np.log(2)
    return (0.5 * (a.exp() * (a - m)).sum(-1) + 0.5 * (b.exp() * (b - m)).sum(-1)).mean().item()
def top1(a, b):
    return (a.argmax(-1) != b.argmax(-1)).float().mean().item()

def matched_idx(log, target):
    # first checkpoint whose train loss reaches the target (plan, matched loss)
    return next(i for i, l in enumerate(log['train']) if l <= target)

# targets span the loss range every run reaches
lo = max(min(log['train']) for log, _ in runs.values())
hi = min(log['train'][0] for log, _ in runs.values())
targets = np.linspace(hi, lo, 25)

def dist_curve(pairs, f):
    """mean distance over the given run pairs at each matched-loss target"""
    out = []
    for t in targets:
        out.append(np.mean([f(runs[a][1][matched_idx(runs[a][0], t)], runs[b][1][matched_idx(runs[b][0], t)])
                            for a, b in pairs]))
    return np.array(out)

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
# (a) train and val loss vs step
for (o, s), (log, _) in runs.items():
    ax[0].plot(log['step'], log['train'], color=COLOR[o], ls='-' if s == 0 else ':', lw=1, label=f'{o} train' if s == 0 else None)
    ax[0].plot(log['step'], log['val'], color=COLOR[o], ls='--', lw=1, alpha=0.6, label=f'{o} val' if s == 0 else None)
ax[0].set(xlabel='step', ylabel='loss (nats)', title='(a) loss', ylim=(0, 4.3)); ax[0].legend(fontsize=7, ncol=2)
# (b) pairwise distance vs matched train loss, seed band = within-optimizer seed 0 vs seed 1
for k, (f, name) in enumerate([(js, 'JS divergence (nats)'), (top1, 'top-1 disagreement')]):
    a = ax[k + 1]
    within = np.stack([dist_curve([((o, 0), (o, 1))], f) for o in OPTS])
    a.fill_between(targets, within.min(0), within.max(0), color='gray', alpha=0.3, label='same optimizer, other seed')
    for o1, o2 in itertools.combinations(OPTS, 2):
        pairs = [((o1, s1), (o2, s2)) for s1 in SEEDS for s2 in SEEDS]
        a.plot(targets, dist_curve(pairs, f), label=f'{o1} vs {o2}')
    a.set(xlabel='matched train loss (nats)', ylabel=name, title=f'({"bc"[k]}) {name}'); a.invert_xaxis(); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f'{OUT}/checkpoint_figure.pdf'); plt.savefig(f'{OUT}/checkpoint_figure.png', dpi=150)

# numbers for the write-up
print('best lr', json.load(open(f'{OUT}/best_lr.json')))
for (o, s), (log, _) in runs.items():
    print(f'{o} s{s} final train {log["train"][-1]:.3f} min val {min(log["val"]):.3f} at step {log["step"][int(np.argmin(log["val"]))]}')
for t in targets[::4]:
    i = {r: matched_idx(runs[r][0], t) for r in runs}
    row = ' '.join(f'{o1[0]}{o2[0]}={js(runs[(o1,0)][1][i[(o1,0)]], runs[(o2,0)][1][i[(o2,0)]]):.4f}' for o1, o2 in itertools.combinations(OPTS, 2))
    seeds = ' '.join(f'{o[0]}{o[0]}={js(runs[(o,0)][1][i[(o,0)]], runs[(o,1)][1][i[(o,1)]]):.4f}' for o in OPTS)
    print(f'loss {t:.2f}  between(s0) {row}  within {seeds}')
