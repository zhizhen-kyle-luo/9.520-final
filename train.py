import torch, torch.nn as nn, torch.nn.functional as F, math, time, json, os, sys, urllib.request

# ---- setup from the plan: nanoGPT Shakespeare-char, batch 64, context 256, no dropout, no wd, no warmup, constant LR
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.backends.cuda.matmul.allow_tf32 = True
OUT = sys.argv[1] if len(sys.argv) > 1 else 'runs'
CTX, BS, N_LAYER, N_HEAD, N_EMBD = 256, 64, 6, 6, 384
SWEEP_STEPS, FULL_STEPS, EVERY, EVAL_ITERS, CHECK_ITERS = 1000, 5000, 100, 20, 8
SWEEP = {'sgd': [0.001, 0.003, 0.01, 0.03, 0.1], 'adam': [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], 'muon': [0.003, 0.01, 0.03, 0.1, 0.3]}
SEEDS, SEEDS_LR2 = range(10), range(4)   # 10 seeds at the picked rate, 4 at the second best rate (lr control)
# predictions are saved when training loss first crosses each target on this grid, so runs are compared at nearly
# the same loss; fixed step checkpoints overshot targets by up to 0.12 nats, which alone moves predictions as much
# as the optimizer effect being measured. the loss is checked often where it falls fast
LOSS_GRID = [round(4.1 - 0.02 * i, 2) for i in range(141)]   # 4.10 down to 1.30
def check_every(it): return 1 if it < 300 else 5 if it < 1500 else 20

# ---- data: same download and 90/10 split as nanoGPT data/shakespeare_char/prepare.py
URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
if not os.path.exists('input.txt'): urllib.request.urlretrieve(URL, 'input.txt')
text = open('input.txt').read()
chars = sorted(set(text)); V = len(chars)
data = torch.tensor([chars.index(c) for c in text], dtype=torch.long, device=DEVICE)
n = int(0.9 * len(data)); train_data, val_data = data[:n], data[n:]
def batch(split, bs, g):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(d) - CTX - 1, (bs,), generator=g).to(DEVICE)
    idx = ix[:, None] + torch.arange(CTX, device=DEVICE)
    return d[idx], d[idx + 1]
# fixed held-out set for prediction distances: 16 windows x 256 = 4096 val tokens
XE, YE = batch('val', 16, torch.Generator().manual_seed(123))

# ---- nanoGPT-style model (no biases, as nanoGPT) with plain attention (fused attention has no double backward, needed later for sharpness)
class Attn(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h; s.qkv = nn.Linear(d, 3*d, bias=False); s.proj = nn.Linear(d, d, bias=False)
    def forward(s, x):
        B, Tn, D = x.shape; q, k, v = s.qkv(x).split(D, -1)
        q, k, v = [t.view(B, Tn, s.h, D // s.h).transpose(1, 2) for t in (q, k, v)]
        a = (q @ k.transpose(-2, -1)) / math.sqrt(D // s.h)
        a = a.masked_fill(torch.triu(torch.ones(Tn, Tn, dtype=torch.bool, device=x.device), 1), float('-inf')).softmax(-1)
        return s.proj((a @ v).transpose(1, 2).reshape(B, Tn, D))
class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.ln1 = nn.LayerNorm(d, bias=False); s.att = Attn(d, h)
        s.ln2 = nn.LayerNorm(d, bias=False); s.mlp = nn.Sequential(nn.Linear(d, 4*d, bias=False), nn.GELU(), nn.Linear(4*d, d, bias=False))
    def forward(s, x):
        x = x + s.att(s.ln1(x))
        return x + s.mlp(s.ln2(x))
class GPT(nn.Module):
    def __init__(s, d=N_EMBD, h=N_HEAD, L=N_LAYER):
        super().__init__(); s.tok = nn.Embedding(V, d); s.pos = nn.Embedding(CTX, d)
        s.blocks = nn.ModuleList([Block(d, h) for _ in range(L)]); s.ln = nn.LayerNorm(d, bias=False); s.head = nn.Linear(d, V, bias=False)
        # nanoGPT init: weights N(0, 0.02), and the 2L layers that add into the residual stream
        # scaled by 1/sqrt(2L) so the stream does not grow with depth
        for m in s.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, std=0.02)
        for n, p in s.named_parameters():
            if n.endswith('att.proj.weight') or n.endswith('mlp.2.weight'): nn.init.normal_(p, std=0.02 / math.sqrt(2 * L))
    def forward(s, x):
        h = s.tok(x) + s.pos(torch.arange(x.size(1), device=x.device))
        for b in s.blocks: h = b(h)
        return s.head(s.ln(h))

# ---- Muon (Newton-Schulz orthogonalization of momentum for 2D hidden weights), AdamW for the rest
def ns_orth(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + 1e-7)
    tr = X.size(0) > X.size(1)
    if tr: X = X.T
    for _ in range(steps):
        A = X @ X.T; B = b * A + c * A @ A; X = a * X + B @ X
    if tr: X = X.T
    return X.float()
class Muon:
    def __init__(s, params2d, lr, mom=0.95):
        s.p = list(params2d); s.lr = lr; s.mom = mom; s.buf = [torch.zeros_like(p) for p in s.p]
    def step(s):
        for p, b in zip(s.p, s.buf):
            if p.grad is None: continue
            b.mul_(s.mom).add_(p.grad)
            o = ns_orth(b) * max(1, p.size(0) / p.size(1)) ** 0.5
            p.data.add_(o, alpha=-s.lr)
    def zero_grad(s):
        for p in s.p: p.grad = None

def make_opt(name, model, lr, adam_lr=None):
    is_hidden = lambda n, p: p.ndim == 2 and n.startswith('blocks.')
    hidden = [p for n, p in model.named_parameters() if is_hidden(n, p)]
    rest = [p for n, p in model.named_parameters() if not is_hidden(n, p)]
    if name == 'sgd': return [torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)]
    # beta2 0.99 as in nanoGPT shakespeare_char, since few tokens per step make the squared-gradient estimate noisy
    if name == 'adam': return [torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.0)]
    # plan: non-hidden params on AdamW at Adam's LR, so Adam and Muon differ only on the hidden matrices
    if name == 'muon': return [Muon(hidden, lr), torch.optim.AdamW(rest, lr=adam_lr, betas=(0.9, 0.99), weight_decay=0.0)]

@torch.no_grad()
def eval_loss(model, split, iters):
    # same batches at every call so loss values are comparable across checkpoints and runs
    g = torch.Generator().manual_seed(7)
    return sum(F.cross_entropy(model(x).view(-1, V), y.view(-1)).item()
               for x, y in (batch(split, BS, g) for _ in range(iters))) / iters

@torch.no_grad()
def weight_norms(model):
    # norm check (plan, plot d): largest entry and largest singular value of each hidden matrix
    W = [p for n, p in model.named_parameters() if p.ndim == 2 and n.startswith('blocks.')]
    return [w.abs().max().item() for w in W], [torch.linalg.matrix_norm(w.float(), ord=2).item() for w in W]

def run(name, lr, seed, steps, adam_lr=None, save_logp=True):
    """train one run and return its loss curve, plus held-out log-probs saved where training loss first crosses
    each LOSS_GRID target
    seed k fixes both init and data order, so seed k of every optimizer shares them (plan, seed control)
    grid_save[j] is the index of the save used for grid target j, or -1 if the run never reached it
    a run whose loss goes non-finite stops early and is marked diverged"""
    torch.manual_seed(seed); model = GPT().to(DEVICE); opts = make_opt(name, model, lr, adam_lr)
    g = torch.Generator().manual_seed(1000 + seed)
    log = dict(name=name, lr=lr, seed=seed, adam_lr=adam_lr, step=[], train=[], val=[], diverged=False,
               save_step=[], save_train=[], save_val=[], save_linf=[], save_spec=[], grid_save=[-1] * len(LOSS_GRID))
    logps, j = [], 0   # j is the next grid target not yet crossed
    t0 = time.time()
    for it in range(steps + 1):
        if it % EVERY == 0 or it == steps:
            model.eval(); tl, vl = eval_loss(model, 'train', EVAL_ITERS), eval_loss(model, 'val', EVAL_ITERS); model.train()
            log['step'].append(it); log['train'].append(tl); log['val'].append(vl)
            if not math.isfinite(tl): log['diverged'] = True; break
        if save_logp and j < len(LOSS_GRID) and it % check_every(it) == 0:
            model.eval(); tl = eval_loss(model, 'train', CHECK_ITERS)
            if tl <= LOSS_GRID[j]:
                with torch.no_grad(): logps.append(F.log_softmax(model(XE), -1).half().cpu())
                linf, spec = weight_norms(model)
                log['save_step'].append(it); log['save_train'].append(tl); log['save_val'].append(eval_loss(model, 'val', CHECK_ITERS))
                log['save_linf'].append(linf); log['save_spec'].append(spec)
                while j < len(LOSS_GRID) and tl <= LOSS_GRID[j]: log['grid_save'][j] = len(logps) - 1; j += 1
            model.train()
        if it == steps: break
        x, y = batch('train', BS, g)
        for o in opts: o.zero_grad()
        F.cross_entropy(model(x).view(-1, V), y.view(-1)).backward()
        for o in opts: o.step()
    print(f'{name} lr={lr} seed={seed} train {log["train"][-1]:.3f} val {log["val"][-1]:.3f} saves {len(logps)} '
          f'{"DIVERGED " if log["diverged"] else ""}time {time.time()-t0:.0f}s', flush=True)
    return log, (torch.stack(logps) if save_logp else None)

def cached_run(path, *args, **kw):
    # colab can disconnect, so each finished run is saved and skipped on restart
    if os.path.exists(path + '.json'): return json.load(open(path + '.json'))
    log, logp = run(*args, **kw)
    if logp is not None: torch.save(logp, path + '_logprobs.pt')
    json.dump(log, open(path + '.json', 'w'))
    return log

def ranked_lrs(name, adam_lr=None):
    # pick by training loss, as kunstner et al. 2023 do when the question is optimizer behavior rather
    # than generalization; mean of the last 3 checkpoints so one loss spike at step 1000 cannot flip the pick
    res = []
    for lr in SWEEP[name]:
        log = cached_run(f'{OUT}/sweep_{name}_lr{lr}', name, lr, 0, SWEEP_STEPS, adam_lr=adam_lr, save_logp=False)
        res.append((sum(log['train'][-3:]) / 3 if not log['diverged'] else float('inf'), lr))
    print('sweep', name, res, flush=True)
    return [lr for _, lr in sorted(res)]

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    print('device', DEVICE, 'vocab', V, 'params', sum(p.numel() for p in GPT().parameters()), flush=True)
    ranked = {'sgd': ranked_lrs('sgd'), 'adam': ranked_lrs('adam')}
    ranked['muon'] = ranked_lrs('muon', adam_lr=ranked['adam'][0])   # muon's AdamW part needs adam's LR first
    lrs = {name: {'best': r[0], 'second': r[1]} for name, r in ranked.items()}
    json.dump(lrs, open(f'{OUT}/lrs.json', 'w'))
    # seeds outermost so a partial run still gives every optimizer the same seeds
    for which, seeds in [('best', SEEDS), ('second', SEEDS_LR2)]:
        for seed in seeds:
            for name in ['sgd', 'adam', 'muon']:
                lr = lrs[name][which]
                cached_run(f'{OUT}/{name}_lr{lr}_seed{seed}', name, lr, seed, FULL_STEPS,
                           adam_lr=lrs['adam']['best'] if name == 'muon' else None)
