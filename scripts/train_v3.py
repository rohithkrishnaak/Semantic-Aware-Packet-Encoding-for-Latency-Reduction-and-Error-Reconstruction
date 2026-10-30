"""SAEP v3 admission: learned tag projection, ranking loss, policy-aware receiver, diversity, early exit.

Changes over train.py (v2), all fit on the CIFAR-100 training file only (45k train / 5k val):
  1. receiver fine-tuned on the tiles the scorer actually admits (plus random masks)
  2. context scorer trained with a listwise ranking loss (only the order inside a frame matters)
  5. tag projection 384 -> 32 learned jointly with the scorer (straight-through int8 rounding),
     initialised from the v2 PCA projection; same 32 B int8 wire format, same TagScheme
  3. diversity weights for greedy selection tuned on val
  4. early-exit confidence thresholds tuned on val so the *average* tiles/frame equals the budget

Writes artifacts/v3/ only; the v2 artifacts used by the demo are left untouched.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from saep.data import ARTIFACTS, CIFAR_DIR, load_cifar, splits  # noqa: E402
from saep.models import (ContextScorer, IntScorer, TagScheme, adaptive_exit, exit_bounds,  # noqa: E402
                         load_admission, load_receiver, order_to_mask, select_diverse)
from scripts.train import linear_probe, random_masks, tile_logits, tile_utility, to_tiles  # noqa: E402

V3 = ARTIFACTS / "v3"
BUDGETS = (1, 2, 3, 4, 5, 6, 8)


class JointTagScorer(nn.Module):
    """Learned projection -> int8 tag (straight-through rounding) -> context MLP score."""

    def __init__(self, scheme: TagScheme, ctx: ContextScorer):
        super().__init__()
        self.register_buffer("mean", torch.tensor(scheme.mean, dtype=torch.float32))
        self.proj = nn.Parameter(torch.tensor(scheme.proj, dtype=torch.float32))
        self.scale = float(scheme.scale)
        self.in_scale = ctx.in_scale
        self.l1 = nn.Linear(ctx.w1.shape[0], ctx.w1.shape[1])
        self.l2 = nn.Linear(ctx.w1.shape[1], 1)
        with torch.no_grad():  # warm start from the v2 scorer
            self.l1.weight.copy_(torch.tensor(ctx.w1.T)); self.l1.bias.copy_(torch.tensor(ctx.b1))
            self.l2.weight.copy_(torch.tensor(ctx.w2[None])); self.l2.bias.fill_(ctx.b2)

    def tags(self, tok: torch.Tensor) -> torch.Tensor:
        z = ((tok.float() - self.mean) @ self.proj / self.scale).clamp(-127, 127)
        return z + (z.round() - z).detach()

    def forward(self, tok: torch.Tensor, avail: torch.Tensor) -> torch.Tensor:
        t = self.tags(tok)
        m = avail.float().unsqueeze(-1)
        mean = (t * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)
        x = torch.cat([t, t - mean], -1) / self.in_scale
        return self.l2(F.relu(self.l1(x)))[..., 0]

    def export(self, scheme_id: int) -> tuple[TagScheme, ContextScorer]:
        scheme = TagScheme(scheme_id, self.mean.cpu().numpy(), self.proj.detach().cpu().numpy(), self.scale)
        ctx = ContextScorer(self.l1.weight.detach().cpu().numpy().T, self.l1.bias.detach().cpu().numpy(),
                            self.l2.weight.detach().cpu().numpy()[0], float(self.l2.bias.item()))
        return scheme, ctx


def random_avail(b: int, p: int, device, max_loss: float = 0.6) -> torch.Tensor:
    """Random packet loss per frame (rate ~ U(0, max_loss)), at least one tile kept."""
    avail = torch.rand(b, p, device=device) >= torch.rand(b, 1, device=device) * max_loss
    avail[torch.arange(b, device=device), torch.randint(0, p, (b,), device=device)] = True
    return avail


def train_joint(joint: JointTagScorer, tok: torch.Tensor, util: torch.Tensor, epochs: int, temp: float,
                device) -> JointTagScorer:
    """Listwise ranking loss: softmax over a frame's available tiles vs softmax(utility / temp)."""
    opt = torch.optim.Adam([{"params": [joint.proj], "lr": 2e-4},
                            {"params": [*joint.l1.parameters(), *joint.l2.parameters()], "lr": 1e-3}])
    n, p = util.shape
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            avail = random_avail(len(idx), p, device)
            s = joint(tok[idx], avail).masked_fill(~avail, -1e9)
            target = F.softmax((util[idx] / temp).masked_fill(~avail, -1e9), 1)
            loss = -(target * F.log_softmax(s, 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idx)
        if ep % 5 == 4 or ep == epochs - 1:
            print(f"  joint epoch {ep + 1}/{epochs} listwise loss {total / n:.4f}")
    return joint.eval()


def finetune_receiver(model, joint: JointTagScorer, tiles: np.ndarray, tok: torch.Tensor, labels: np.ndarray,
                      grid: int, epochs: int, device):
    """Train on the masks the receiver will really see: scorer top-k (k ~ U{1..8}) after random loss,
    mixed 50/50 with the uniformly random masks the v2 receiver was trained on."""
    p, t = grid * grid, 32 // grid
    model = model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-4)
    x_all, y_all = torch.from_numpy(tiles).to(device), torch.from_numpy(labels).to(device)
    steps = epochs * ((len(x_all) + 255) // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4, total_steps=steps, pct_start=0.15)
    for ep in range(epochs):
        perm = torch.randperm(len(x_all), device=device)
        correct = n = 0
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            b = len(idx)
            with torch.no_grad():
                avail = random_avail(b, p, device, max_loss=0.4)
                s = joint(tok[idx], avail).masked_fill(~avail, -1e9)
                k = torch.randint(1, 9, (b, 1), device=device)
                rank = s.argsort(1, descending=True).argsort(1)
                mask = (rank < k) & avail
                use_rand = torch.rand(b, device=device) < 0.5
                mask = torch.where(use_rand[:, None], random_masks(b, p, device), mask)
            x, y = x_all[idx], y_all[idx]
            flip = torch.rand(b, device=device) < 0.5  # flip image and mask together
            xf = x.view(-1, grid, grid, t, t, 3).flip(dims=(2, 4)).view_as(x)
            mf = mask.view(-1, grid, grid).flip(dims=(2,)).reshape(b, p)
            x = torch.where(flip.view(-1, 1, 1, 1, 1), xf, x)
            mask = torch.where(flip[:, None], mf, mask)
            logits = model(x, mask)
            loss = F.cross_entropy(logits, y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            correct += (logits.argmax(1) == y).sum().item(); n += b
        print(f"  receiver epoch {ep + 1}/{epochs} acc(train masks) {correct / n:.3f}")
    return model.eval()


@torch.no_grad()
def scores_np(joint: JointTagScorer, tok: torch.Tensor, avail: np.ndarray, device) -> np.ndarray:
    out = []
    for i in range(0, len(tok), 4096):
        a = torch.from_numpy(avail[i:i + 4096]).to(device)
        out.append(joint(tok[i:i + 4096], a).cpu().numpy())
    return np.concatenate(out)


def acc_of(z: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float:
    s = (z * mask[..., None]).sum(1)
    return float(((s.argmax(1) == labels) & mask.any(1)).mean())


def topk_mask(score: np.ndarray, avail: np.ndarray, k: int) -> np.ndarray:
    s = np.where(avail, score, -np.inf)
    order = np.argsort(-s, 1, kind="stable")[:, :k]
    m = np.zeros_like(avail)
    np.put_along_axis(m, order, True, 1)
    return m & avail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint-epochs", type=int, default=25)
    ap.add_argument("--receiver-epochs", type=int, default=10)
    ap.add_argument("--temp", type=float, default=1.0, help="listwise target temperature (nats)")
    ap.add_argument("--scheme-id", type=int, default=3)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    V3.mkdir(parents=True, exist_ok=True)

    feats = np.load(ARTIFACTS / "features_train.npz")
    grid = int(feats["grid"]); p = grid * grid
    n = len(feats["labels"])
    sp = splits()
    tr = np.array([i for i in sp["train"] if i < n]); va = np.array([i for i in sp["val"] if i < n])
    images, labels = load_cifar(CIFAR_DIR / "train")
    tiles = to_tiles(images[:n], grid)
    tok_all = torch.from_numpy(feats["tile_tok"]).to(device)  # fp16 on GPU
    tok_tr, tok_va = tok_all[torch.from_numpy(tr).to(device)], tok_all[torch.from_numpy(va).to(device)]
    scheme2, scorers2, _, _ = load_admission(ARTIFACTS / "admission.npz")
    recv2 = load_receiver(ARTIFACTS / "receiver.pt", grid, device)
    y_tr, y_va = labels[tr], labels[va]

    def val_report(tag: str, z_va, score_va, k=4):
        print(f"  [val] {tag}: k={k} top-k acc {acc_of(z_va, y_va, topk_mask(score_va, np.ones_like(score_va, bool), k)):.4f}")

    t0 = time.time()
    print("per-tile utilities under the v2 receiver ...")
    z_tr = tile_logits(recv2, tiles[tr], device); z_va = tile_logits(recv2, tiles[va], device)
    util_tr = torch.tensor(tile_utility(z_tr, y_tr), device=device)
    tags2_va = scheme2.tags(feats["tile_tok"][va])
    val_report("v2 PCA tags + MSE context MLP", z_va, scorers2["ctx"].score(tags2_va))

    print("stage A: learned projection + listwise scorer (v2 receiver) ...")
    joint = train_joint(JointTagScorer(scheme2, scorers2["ctx"]).to(device), tok_tr, util_tr,
                        args.joint_epochs, args.temp, device)
    all_va = np.ones((len(va), p), bool)
    val_report("stage A", z_va, scores_np(joint, tok_va, all_va, device))
    sa_scheme, sa_ctx = joint.export(args.scheme_id)
    np.savez(V3 / "admission_stageA.npz", grid=grid, scheme_id=args.scheme_id, tag_mean=sa_scheme.mean,
             tag_proj=sa_scheme.proj, tag_scale=sa_scheme.scale, ctx_w1=sa_ctx.w1, ctx_b1=sa_ctx.b1,
             ctx_w2=sa_ctx.w2, ctx_b2=sa_ctx.b2)
    del z_tr

    print("stage B: fine-tune the receiver on scorer-admitted tiles ...")
    recv3 = finetune_receiver(copy.deepcopy(recv2), joint, tiles[tr], tok_tr, y_tr, grid, args.receiver_epochs, device)
    torch.save(recv3.state_dict(), V3 / "receiver.pt")
    z_tr = tile_logits(recv3, tiles[tr], device); z_va = tile_logits(recv3, tiles[va], device)
    print(f"  [val] v3 receiver, all 16 tiles: {acc_of(z_va, y_va, all_va):.4f}"
          f"  (v2 receiver: {acc_of(tile_logits(recv2, tiles[va], device), y_va, all_va):.4f})")
    util_tr = torch.tensor(tile_utility(z_tr, y_tr), device=device)

    print("stage C: re-fit projection + scorer to the new receiver's utilities ...")
    joint = train_joint(joint.train(), tok_tr, util_tr, max(5, args.joint_epochs // 2), args.temp, device)
    score_va = scores_np(joint, tok_va, all_va, device)
    val_report("stage C", z_va, score_va)
    scheme, ctx = joint.export(args.scheme_id)
    tags = scheme.tags(feats["tile_tok"])
    # The numpy export (what the receiver runs) must pick the same tiles as the torch scorer. GPU vs CPU
    # float error flips the int8 rounding of a few tag values per million, so compare top-4 sets.
    s_np = ctx.score(tags[va])
    same = (np.sort(np.argsort(-s_np, 1)[:, :4], 1) == np.sort(np.argsort(-score_va, 1)[:, :4], 1)).all(1).mean()
    print(f"  export check: numpy and torch pick the same top-4 on {same:.2%} of val frames")
    assert same > 0.995
    score_va = s_np

    print("refit the int8 XDP-stage scorers on the learned tags ...")
    util_tr_np = util_tr.cpu().numpy()
    util_c = util_tr_np - util_tr_np.mean(1, keepdims=True)
    pos_tr = np.broadcast_to(np.arange(p), (len(tr), p))
    int_scorers = {"tag": IntScorer.fit(tags[tr], pos_tr, util_c, p, use_pos=False),
                   "tagpos": IntScorer.fit(tags[tr], pos_tr, util_c, p, use_pos=True)}

    print("stage D: tune diversity weights on val (budget 4) ...")
    best = (acc_of(z_va, y_va, topk_mask(score_va, all_va, 4)), 0.0, 0.0)
    for lam_adj in (0.0, 0.1, 0.2, 0.4, 0.8):
        for lam_sim in (0.0, 0.25, 0.5, 1.0, 2.0):
            accs = [acc_of(z_va, y_va, order_to_mask(select_diverse(score_va, tags[va], all_va, k, grid, lam_adj, lam_sim), p))
                    for k in (2, 4, 6)]
            if np.mean(accs) > best[0] + 1e-9 and (lam_adj or lam_sim):
                best = (float(np.mean(accs)), lam_adj, lam_sim)
    _, lam_adj, lam_sim = best
    print(f"  chosen lam_adj={lam_adj} lam_sim={lam_sim}")

    print("stage E: early-exit thresholds so that mean tiles/frame = budget (val) ...")
    kmax_mult = 2  # kmin = budget-1, kmax = 2*budget, margin confidence: chosen on val (see README)
    full_order = select_diverse(score_va, tags[va], all_va, p, grid, lam_adj, lam_sim)
    thetas = []
    for b in BUDGETS:
        kmin, kmax = exit_bounds(b, kmax_mult, p)
        lo, hi = 0.0, 1.0
        for _ in range(30):  # higher theta -> more tiles; find theta giving mean == b
            mid = (lo + hi) / 2
            if adaptive_exit(z_va, full_order, mid, kmin, kmax).mean() > b: hi = mid
            else: lo = mid
        used = adaptive_exit(z_va, full_order, lo, kmin, kmax)
        mask = order_to_mask(np.where(np.arange(p)[None] < used[:, None], full_order, -1), p)
        print(f"  budget {b}: theta={lo:.3f} mean tiles {used.mean():.2f}  val acc {acc_of(z_va, y_va, mask):.4f}"
              f"  (fixed k: {acc_of(z_va, y_va, order_to_mask(full_order[:, :b], p)):.4f})")
        thetas.append(lo)

    print("leakage control: linear probe on the mean of the 16 learned tags (v2 PCA tags: 71.3%) ...")
    tag_only = linear_probe(tags[tr].astype(np.float32).mean(1), y_tr, tags[va].astype(np.float32).mean(1), y_va, device)
    print(f"  tag-only probe, learned tags: {tag_only:.3f}")

    np.savez(V3 / "admission.npz", grid=grid, scheme_id=scheme.scheme_id, tag_mean=scheme.mean,
             tag_proj=scheme.proj, tag_scale=scheme.scale, pos_prior=util_c.mean(0),
             tag_only_val=tag_only, cls_only_val=0.0,
             ctx_w1=ctx.w1, ctx_b1=ctx.b1, ctx_w2=ctx.w2, ctx_b2=ctx.b2,
             div_lam_adj=lam_adj, div_lam_sim=lam_sim,
             exit_budgets=np.array(BUDGETS), exit_theta=np.array(thetas), exit_kmax_mult=kmax_mult,
             **{f"{k}_{f}": getattr(s, f) for k, s in int_scorers.items() for f in ("w", "w_pos", "b")})
    print(f"saved to {V3}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
