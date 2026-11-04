import torch, torch.nn as nn, torch.nn.functional as F, math, time, json, random

torch.manual_seed(0); random.seed(0)
# ---- data: synthetic "language" with structure (a second-order Markov grammar over 32 symbols)
V = 32
T = torch.rand(V, V, V); T = T ** 6; T = T / T.sum(2, keepdim=True)   # second-order Markov over 32 symbols, peaked
def gen(n):
    s = [0, 1]
    for _ in range(n - 2):
        s.append(torch.multinomial(T[s[-2], s[-1]], 1).item())
    return torch.tensor(s)
data = gen(120000)
CTX = 16
def batch(bs, g):
    ix = torch.randint(0, len(data) - CTX - 1, (bs,), generator=g)
    x = torch.stack([data[i:i+CTX] for i in ix]); y = torch.stack([data[i+1:i+CTX+1] for i in ix])
    return x, y
geval = torch.Generator().manual_seed(123)
XE, YE = batch(256, geval)   # fixed eval set, ~4000 tokens

# ---- tiny GPT
class Attn(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h; s.qkv = nn.Linear(d, 3*d); s.proj = nn.Linear(d, d)
    def forward(s, x):
        B, Tn, D = x.shape; q, k, v = s.qkv(x).split(D, -1)
        q, k, v = [t.view(B, Tn, s.h, D // s.h).transpose(1, 2) for t in (q, k, v)]
        a = (q @ k.transpose(-2, -1)) / math.sqrt(D // s.h)
        a = a.masked_fill(torch.triu(torch.ones(Tn, Tn, dtype=torch.bool), 1), float('-inf')).softmax(-1)
        return s.proj((a @ v).transpose(1, 2).reshape(B, Tn, D))
class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.ln1 = nn.LayerNorm(d); s.att = Attn(d, h)
        s.ln2 = nn.LayerNorm(d); s.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(s, x):
        x = x + s.att(s.ln1(x))
        return x + s.mlp(s.ln2(x))
class GPT(nn.Module):
    def __init__(s, d=64, h=4, L=2):
        super().__init__(); s.tok = nn.Embedding(V, d); s.pos = nn.Embedding(CTX, d)
        s.blocks = nn.ModuleList([Block(d, h) for _ in range(L)]); s.ln = nn.LayerNorm(d); s.head = nn.Linear(d, V, bias=False)
    def forward(s, x):
        h = s.tok(x) + s.pos(torch.arange(x.size(1)))
        for b in s.blocks: h = b(h)
        return s.head(s.ln(h))

# ---- Muon (Newton-Schulz orthogonalization of momentum for 2D hidden weights), AdamW for the rest
def ns_orth(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() if G.dtype == torch.float32 else G
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
            b.mul_(s.mom).add_(p.grad); g = b
            o = ns_orth(g) * max(1, p.size(0) / p.size(1)) ** 0.5
            p.data.add_(o, alpha=-s.lr)
    def zero_grad(s):
        for p in s.p: p.grad = None

def make_opt(name, model, lr):
    hidden = [p for n, p in model.named_parameters() if p.ndim == 2 and 'tok' not in n and 'pos' not in n and 'head' not in n]
    rest = [p for n, p in model.named_parameters() if not (p.ndim == 2 and 'tok' not in n and 'pos' not in n and 'head' not in n)]
    if name == 'sgd': return [torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)]
    if name == 'adam': return [torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)]
    if name == 'muon': return [Muon(hidden, lr), torch.optim.AdamW(rest, lr=3e-3, weight_decay=0.0)]

def sharpness(model, x, y, iters=10):
    params = [p for p in model.parameters()]
    loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
    g = torch.autograd.grad(loss, params, create_graph=True)
    v = [torch.randn_like(p) for p in params]; nv = math.sqrt(sum((t**2).sum() for t in v)); v = [t / nv for t in v]
    lam = 0.0
    for _ in range(iters):
        hv = torch.autograd.grad(g, params, grad_outputs=v, retain_graph=True)
        lam = sum((h * t).sum() for h, t in zip(hv, v)).item()
        n = math.sqrt(sum((h**2).sum() for h in hv)); v = [h / n for h in hv]
    return lam

def run(name, lr, seed, steps=2000, every=100, do_sharp=False):
    torch.manual_seed(seed); model = GPT(); opts = make_opt(name, model, lr)
    g = torch.Generator().manual_seed(1000)   # same data order for all runs
    log = []; t0 = time.time()
    for it in range(steps + 1):
        if it % every == 0:
            model.eval()
            with torch.no_grad():
                lp = F.log_softmax(model(XE), -1); tl = F.cross_entropy(lp.view(-1, V), YE.view(-1)).item()
            rec = dict(step=it, loss=tl, logp=lp)
            if do_sharp: rec['sharp'] = sharpness(model, *batch(64, torch.Generator().manual_seed(7)))
            log.append(rec); model.train()
        x, y = batch(64, g)
        for o in opts: o.zero_grad()
        F.cross_entropy(model(x).view(-1, V), y.view(-1)).backward()
        for o in opts: o.step()
    print(f'{name} lr={lr} seed={seed} final loss {log[-1]["loss"]:.3f} time {time.time()-t0:.0f}s', flush=True)
    return log

def kl(a, b):  # mean KL(a||b) over tokens
    return (a.exp() * (a - b)).sum(-1).mean().item()

if __name__ == '__main__':
    # quick LR sweep (3 values each), pick by final eval loss
    best = {}
    for name, lrs in [('sgd', [0.1, 0.3, 1.0]), ('adam', [1e-3, 3e-3, 1e-2]), ('muon', [0.01, 0.03, 0.1])]:
        res = [(run(name, lr, 0, steps=600, every=600)[-1]['loss'], lr) for lr in lrs]
        best[name] = min(res)[1]; print('best', name, best[name], flush=True)
    logs = {n: run(n, best[n], 0, do_sharp=(n == 'sgd')) for n in ['sgd', 'adam', 'muon']}
    logs['adam_seed1'] = run('adam', best['adam'], 1)
    out = []
    for i in range(len(logs['sgd'])):
        r = {k: logs[k][i]['loss'] for k in logs}; r['step'] = logs['sgd'][i]['step']
        r['kl_sgd_adam'] = kl(logs['sgd'][i]['logp'], logs['adam'][i]['logp'])
        r['kl_adam_muon'] = kl(logs['adam'][i]['logp'], logs['muon'][i]['logp'])
        r['kl_sgd_muon'] = kl(logs['sgd'][i]['logp'], logs['muon'][i]['logp'])
        r['kl_adam_seeds'] = kl(logs['adam'][i]['logp'], logs['adam_seed1'][i]['logp'])
        r['sharp_sgd'] = logs['sgd'][i]['sharp']; r['lr_sgd'] = best['sgd']
        out.append(r)
    json.dump(out, open('pilot_out.json', 'w'), indent=1)
    print(f"{'step':>5} {'L_sgd':>6} {'L_adam':>6} {'L_muon':>6} {'KL s-a':>7} {'KL a-m':>7} {'KL s-m':>7} {'KL seed':>7} {'sharp':>7} {'limit':>6}")
    for r in out:
        print(f"{r['step']:>5} {r['sgd']:6.3f} {r['adam']:6.3f} {r['muon']:6.3f} {r['kl_sgd_adam']:7.4f} {r['kl_adam_muon']:7.4f} {r['kl_sgd_muon']:7.4f} {r['kl_adam_seeds']:7.4f} {r['sharp_sgd']:7.2f} {3.8/best['sgd']:6.2f}")
