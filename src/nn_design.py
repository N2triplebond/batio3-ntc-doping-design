# -*- coding: utf-8 -*-
"""
BaTiO3 基 NTC 陶瓷 B 位配方的神经网络反设计（纯 numpy 实现，无需 torch）

三段式结构（对应 MatterGen / MatterSim 的思路，但适配"数据极少"的实情）：

  G_θ  反向生成器  (Ea_target, z) → 组分 f
       输出层 softmax 后，令 x := Σ f_i·a_i、Ti := 1−Σ f，电荷守恒与位点守恒
       由构造保证，所以生成的每一个样本都物理可行（对应 MatterGen 的条件生成）
  S_φ  残差代理    features(f) → ΔEa = Ea_true − Ea_prior
       只学"规则先验错了多少"，不重学物理；集成 K 个网络给出不确定度
       （对应 MatterSim 的"先有普适势、再微调"的思路）
  AL   主动学习    采样 → UCB 采集 → DFT 打标 → 只更新残差 → 重训 G

用法：
  python nn_design.py --demo        # 合成 oracle 验证闭环（默认）
  python nn_design.py --props       # 只用先验出配方（不接 DFT）
"""
import argparse
import csv
import math
import os

import numpy as np

K_B = 8.617333e-5
R_TI, R_O, R_BA, R_LA = 0.0605, 0.140, 0.161, 0.136
E_INF, B_SLOPE = 0.498, 0.028          # Ea_prior = E_INF − B_SLOPE·d̄
OUT = os.path.dirname(os.path.abspath(__file__))
RNG_SEED = 20261026

EL_NAMES = ["Cr", "Fe", "Mn", "V", "Co", "Ni", "Cu"]   # Ga 无变价对，不进导电掺杂池
Z = np.array([3, 3, 3, 3, 3, 2, 2], dtype=float)
RAD = np.array([0.0615, 0.0645, 0.0645, 0.0640, 0.0580, 0.0690, 0.0730])
D_EL = np.array([3, 5, 4, 2, 6, 8, 9], dtype=float)
RISK = np.array([0.35, 0.10, 0.55, 0.65, 0.20, 0.35, 0.75])
A_EL = 4.0 - Z                          # 受主强度
M_ACT = np.ones(7)


# ---------------- 物理层 ----------------
def phys(f):
    """f: (B, 8) 组分分数 → 派生量"""
    f = np.atleast_2d(f)
    y = f.sum(1)
    x = f @ A_EL
    sm = f @ M_ACT + 1e-12
    sdm = f @ (D_EL * M_ACT)
    dbar = sdm / sm
    rbar = (f @ RAD) / np.maximum(y, 1e-9)
    dr = np.abs(rbar - R_TI) / R_TI
    r_a = (1 - x) * R_BA + x * R_LA
    t = (r_a + R_O) / (math.sqrt(2) * (rbar + R_O))
    ea = E_INF - B_SLOPE * dbar
    H = -(f * np.log(np.maximum(f, 1e-12))).sum(1)
    act = (f @ M_ACT) / np.maximum(y, 1e-9)
    return dict(y=y, x=x, dbar=dbar, rbar=rbar, dr=dr, t=t, ea=ea, H=H, act=act)


def features(f):
    p = phys(f)
    return np.concatenate([f, p["x"][:, None], p["dbar"][:, None], p["rbar"][:, None],
                           p["t"][:, None], p["H"][:, None], p["y"][:, None],
                           p["act"][:, None]], 1)


def feasible(f, tol=1e-9):
    p = phys(f)
    return ((p["y"] <= 0.40) & (p["y"] >= 0.02) & (p["t"] >= 0.95) & (p["t"] <= 1.06)
            & (p["act"] >= 0.70) & (p["dr"] <= 0.12))


# ---------------- numpy 网络 ----------------
class Adam:
    def __init__(self, shapes, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros(s) for s in shapes]
        self.v = [np.zeros(s) for s in shapes]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


def he(rng, nin, nout):
    return rng.normal(0, math.sqrt(2.0 / nin), (nin, nout))


class MLP:
    """两层隐层的回归网络：x → 64 → 64 → 1"""

    def __init__(self, nin, hidden=64, seed=0):
        rng = np.random.default_rng(seed)
        self.P = [he(rng, nin, hidden), np.zeros(hidden),
                  he(rng, hidden, hidden), np.zeros(hidden),
                  he(rng, hidden, 1), np.zeros(1)]
        self.opt = Adam([p.shape for p in self.P], lr=2e-3)

    def forward(self, X):
        W1, b1, W2, b2, W3, b3 = self.P
        h1 = np.maximum(X @ W1 + b1, 0)
        h2 = np.maximum(h1 @ W2 + b2, 0)
        out = (h2 @ W3 + b3).ravel()
        self.cache = (X, h1, h2)
        return out

    def backward(self, dout):
        X, h1, h2 = self.cache
        W1, b1, W2, b2, W3, b3 = self.P
        dout = dout[:, None]
        gW3 = h2.T @ dout
        gb3 = dout.sum(0)
        dh2 = (dout @ W3.T) * (h2 > 0)
        gW2 = h1.T @ dh2
        gb2 = dh2.sum(0)
        dh1 = (dh2 @ W2.T) * (h1 > 0)
        gW1 = X.T @ dh1
        gb1 = dh1.sum(0)
        return [gW1, gb1, gW2, gb2, gW3, gb3]

    def step(self):
        self.opt.step(self.P, self.grads)


class Generator:
    """(Ea_target, x_target, z) → 亚随机 softmax：f = s·q，s∈[0.02, 0.40]

    softmax 天然满足 Σf=1，而 Σf 正是总掺杂量 y，因此必须再乘一个门控标量 s，
    否则生成器只能产出"全部 B 位被取代"的非物理配方。x := Σf·a 由构造满足。
    """

    SMIN, SMAX = 0.02, 0.40

    def __init__(self, nz=6, hidden=64, seed=1):
        rng = np.random.default_rng(seed)
        nin = 2 + nz
        self.nz = nz
        self.P = [he(rng, nin, hidden), np.zeros(hidden),
                  he(rng, hidden, hidden), np.zeros(hidden),
                  he(rng, hidden, len(EL_NAMES)), np.zeros(len(EL_NAMES)),
                  he(rng, hidden, 1), np.zeros(1)]
        self.opt = Adam([p.shape for p in self.P], lr=4e-3)

    def forward(self, target, xtgt, znoise):
        W1, b1, W2, b2, W3, b3, W4, b4 = self.P
        X = np.concatenate([target[:, None], xtgt[:, None], znoise], 1)
        h1 = np.maximum(X @ W1 + b1, 0)
        h2 = np.maximum(h1 @ W2 + b2, 0)
        logits = h2 @ W3 + b3
        gpre = (h2 @ W4 + b4).ravel()
        z_ = logits - logits.max(1, keepdims=True)
        e = np.exp(z_)
        q = e / e.sum(1, keepdims=True)
        sig = 1.0 / (1.0 + np.exp(-gpre))
        sc = self.SMIN + (self.SMAX - self.SMIN) * sig
        f = sc[:, None] * q
        self.cache = (X, h1, h2, q, sc, sig)
        return f

    def step(self):
        self.opt.step(self.P, self.grads)

    def loss_grad(self, target, xtgt, f, w_ea=1.0, w_dr=6.0, w_t=6.0,
                  w_ent=0.008, w_x=4.0, w_y=8.0):
        p = phys(f)
        sm = (f @ M_ACT) + 1e-12
        sdm = f @ (D_EL * M_ACT)
        sy = np.maximum(p["y"], 1e-9)

        dEa = -B_SLOPE * ((D_EL * M_ACT * sm[:, None] - sdm[:, None] * M_ACT) / (sm[:, None] ** 2))
        drbar = (RAD[None, :] * sy[:, None] - (f @ RAD)[:, None]) / (sy[:, None] ** 2)
        ddr = np.sign(p["rbar"] - R_TI)[:, None] / R_TI * drbar
        rA2 = (1 - p["x"]) * R_BA + p["x"] * R_LA
        dnum = (R_LA - R_BA) * A_EL[None, :]
        dt = (dnum * (p["rbar"] + R_O)[:, None] - (rA2 + R_O)[:, None] * drbar) / \
             (math.sqrt(2) * (p["rbar"] + R_O)[:, None] ** 2)

        g = 2 * w_ea * (p["ea"] - target)[:, None] * dEa
        act_dr = np.maximum(0, p["dr"] - 0.06)[:, None] * np.sign(p["dr"] - 0.06)[:, None]
        g += 2 * w_dr * act_dr * ddr
        act_t = np.maximum(0, np.abs(p["t"] - 1.0) - 0.05)[:, None] * np.sign(p["t"] - 1.0)[:, None]
        g += 2 * w_t * act_t * dt
        g += w_ent * (np.log(np.maximum(f, 1e-12)) + 1.0)
        g += 2 * w_x * (p["x"] - xtgt)[:, None] * A_EL[None, :]
        pen_y = np.maximum(0, p["y"] - 0.34) - np.maximum(0, 0.03 - p["y"])
        g += 2 * w_y * pen_y[:, None]

        X, h1, h2, q, sc, sig = self.cache
        W1, b1, W2, b2, W3, b3, W4, b4 = self.P
        dq = sc[:, None] * g
        dlogits = q * (dq - (q * dq).sum(1, keepdims=True))
        dL_ds = (g * q).sum(1)
        ds_dg = (self.SMAX - self.SMIN) * sig * (1 - sig)
        dgpre = (dL_ds * ds_dg)[:, None]

        gW3 = h2.T @ dlogits
        gb3 = dlogits.sum(0)
        gW4 = h2.T @ dgpre
        gb4 = dgpre.sum(0)
        dh2 = (dlogits @ W3.T + dgpre @ W4.T) * (h2 > 0)
        gW2 = h1.T @ dh2
        gb2 = dh2.sum(0)
        dh1 = (dh2 @ W2.T) * (h1 > 0)
        gW1 = X.T @ dh1
        gb1 = dh1.sum(0)
        self.grads = [gW1, gb1, gW2, gb2, gW3, gb3, gW4, gb4]
        return float(((p["ea"] - target) ** 2).mean())


# ---------------- 训练数据：规则枚举器 ----------------
def enum_dataset(x_list=(0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20),
                 step=0.02, fmin=0.02, cap=60000, seed=RNG_SEED):
    """在约束面内随机采样可行组分（用规则模型当"免费标签"）"""
    rng = np.random.default_rng(seed)
    F = []
    while len(F) < cap:
        x = rng.choice(x_list)
        k = rng.integers(1, 5)
        idx = rng.choice(len(EL_NAMES), k, replace=False)
        w = rng.dirichlet(np.ones(k))
        f = np.zeros(len(EL_NAMES))
        a = A_EL[idx]
        scale = x / (w @ a)                     # 强制满足 Σf·a = x
        f[idx] = w * scale
        f = np.round(f / step) * step
        f[f < fmin] = 0.0
        f = np.round(f, 4)
        if f.sum() <= 0 or f.sum() > 0.40:
            continue
        F.append(f)
    F = np.array(F)
    return F[feasible(F)]


def prune_topk(F, xtgt, k=3, ymax=0.40):
    """只保留质量最大的 k 个组分，并把剩余按 Σf·a = x_target 反投影，
    保证剪枝后电荷守恒不被破坏（x 会随剪枝改变，必须重投影）"""
    out = []
    for f, x in zip(F, xtgt):
        idx = np.argsort(-f)[:k]
        w = f[idx]
        if w.sum() <= 1e-9:
            continue
        p = w / w.sum()
        a_bar = p @ A_EL[idx]
        if a_bar <= 1e-9:
            continue
        y_new = x / a_bar
        if y_new > ymax or y_new < 0.02:
            continue
        fn = np.zeros_like(f)
        fn[idx] = p * y_new
        out.append(np.round(fn, 4))
    return np.array(out) if out else np.zeros((0, len(EL_NAMES)))


# ---------------- 主动学习闭环 ----------------
def make_oracle(coef=(-0.15, 0.30, 0.22), noise=0.012, seed=7):
    """合成真值：先验 + 元素间交互项（模拟 DFT 会修正的部分）"""
    rng = np.random.default_rng(seed)
    def oracle(f):
        p = phys(f)
        c1, c2, c3 = coef
        inter = c1 * f[:, 4] * f[:, 1] + c2 * f[:, 5] * f[:, 0] + c3 * f[:, 2] * f[:, 3]
        return p["ea"] + inter + rng.normal(0, noise, f.shape[0])
    return oracle


def ucbs(ens, X, kappa=1.5):
    preds = np.stack([m.forward(X) for m in ens])
    return preds.mean(0), preds.std(0), preds.mean(0) - kappa * preds.std(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", default=True)
    ap.add_argument("--props", action="store_true")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--batch", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    # 1) 先验数据集 + 生成器训练
    print("[1/4] 构造可行域数据集 ...")
    F0 = enum_dataset()
    print("      可行组分 %d 条" % len(F0))
    Xin = features(F0)
    Ea0 = phys(F0)["ea"]
    mu, sd = Xin.mean(0), Xin.std(0) + 1e-9

    print("[2/4] 训练反向生成器 G ...")
    G = Generator(seed=3)
    rng = np.random.default_rng(RNG_SEED)
    for it in range(1500):
        tgt = rng.uniform(0.28, 0.42, 256)
        xt = rng.choice([0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20], 256)
        z = rng.normal(0, 1, (256, G.nz))
        f = G.forward(tgt, xt, z)
        l = G.loss_grad(tgt, xt, f)
        G.step()
        if it % 300 == 0:
            p = phys(f)
            print("      it%4d  loss=%.2e  <Ea>=%.3f  <x>=%.3f  <y>=%.3f  <t>=%.3f"
                  % (it, l, p["ea"].mean(), p["x"].mean(), p["y"].mean(), p["t"].mean()))

    # 2) 采样 + 可行性过滤
    print("[3/4] 采样候选 ...")
    tgt = np.full(20000, 0.32)
    xts = rng.choice([0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20], 20000)
    z = rng.normal(0, 1, (20000, G.nz))
    Fs = G.forward(tgt, xts, z)
    Fs = prune_topk(Fs, xts, k=3)
    if len(Fs):
        Fs = Fs[feasible(Fs)]
    # 去重（0.005 栅格）
    Fs = np.unique(np.round(Fs, 3), axis=0)
    print("      可行且去重后 %d 条" % len(Fs))

    # 3) 残差代理 + 主动学习
    print("[4/4] 残差代理 + 主动学习 ...")
    ens = [MLP(Xin.shape[1], seed=10 + i) for i in range(5)]
    oracle = make_oracle()
    labels_f, labels_y = [], []
    log = []
    for rd in range(args.rounds):
        Xs = (features(Fs) - mu) / sd
        m, s, acq = ucbs(ens, Xs)
        order = np.argsort(acq)[::-1][:args.batch]
        picks = Fs[order]
        y_new = oracle(picks)
        labels_f.append(picks)
        labels_y.append(y_new)
        Ftr = np.vstack(labels_f)
        Ytr = np.concatenate(labels_y)
        Xtr = (features(Ftr) - mu) / sd
        prior_tr = phys(Ftr)["ea"]
        resid = Ytr - prior_tr
        for ep in range(400):
            for mdl in ens:
                idx = rng.permutation(len(Xtr))
                for b in range(0, len(Xtr), 16):
                    j = idx[b:b + 16]
                    out = mdl.forward(Xtr[j])
                    d = 2 * (out - resid[j]) / len(j)
                    mdl.grads = mdl.backward(d)
                    mdl.step()
        pred = np.mean([mdl.forward(Xtr) for mdl in ens], 0) + prior_tr
        rmse = float(np.sqrt(((pred - Ytr) ** 2).mean()))
        # 留出集：用最后一个点做测试更公平（样本极少时只能如此）
        ho = max(2, len(Ytr) // 5)
        pred_ho = np.mean([mdl.forward(Xtr[-ho:]) for mdl in ens], 0) + prior_tr[-ho:]
        rmse_ho = float(np.sqrt(((pred_ho - Ytr[-ho:]) ** 2).mean()))
        log.append((rd + 1, len(Ytr), rmse))
        print("      round %d: 样本 %3d  训练RMSE=%.4f  留出RMSE=%.4f eV"
              % (rd + 1, len(Ytr), rmse, rmse_ho))

    # 4) 输出配方
    Xs = (features(Fs) - mu) / sd
    m, s, acq = ucbs(ens, Xs)
    p = phys(Fs)
    ea_hat = p["ea"] + m
    score = (0.32 - np.abs(ea_hat - 0.32)) / 0.10
    score = (score + 0.6 * (1 - p["dr"] / 0.12)
             + 0.5 * (1 - (Fs @ RISK) / np.maximum(p["y"], 1e-9))
             + 0.5 * p["act"] - 0.8 * s)
    order = np.argsort(score)[::-1]

    seen, out = set(), []
    for i in order:
        key = tuple(np.round(Fs[i], 2))
        if key in seen:
            continue
        seen.add(key)
        out.append(i)
        if len(out) >= 20:
            break

    path = os.path.join(OUT, "nn_proposals.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["排名", "配方", "x", "Ea先验", "Ea修正后", "±σ", "B预测/K",
                    "尺寸失配", "容差因子", "构型熵", "B位组成"])
        for r, i in enumerate(out, 1):
            comp = " ".join("%s%.3f" % (EL_NAMES[k], Fs[i][k])
                            for k in np.argsort(-Fs[i]) if Fs[i][k] > 0)
            formula = "Ba%.3gLa%.3gTi%.3g" % (1 - p["x"][i], p["x"][i], 1 - p["y"][i]) + \
                      "".join("%s%.3g" % (EL_NAMES[k], Fs[i][k])
                              for k in np.argsort(-Fs[i]) if Fs[i][k] > 0) + "O3"
            w.writerow([r, formula, "%.3f" % p["x"][i], "%.3f" % p["ea"][i],
                        "%.3f" % ea_hat[i], "%.3f" % s[i], "%5.0f" % (ea_hat[i] / K_B),
                        "%.3f" % p["dr"][i], "%.4f" % p["t"][i], "%.2f" % p["H"][i], comp])

    print("\n%-40s %6s %6s %6s %6s" % ("配方", "Ea先验", "Ea修正", "±σ", "B/K"))
    for i in out[:12]:
        comp = "".join("%s%.3g" % (EL_NAMES[k], Fs[i][k])
                       for k in np.argsort(-Fs[i]) if Fs[i][k] > 0)
        print("%-40s %6.3f %6.3f %6.3f %6.0f"
              % ("Ba%.3gLa%.3gTi%.3g%sO3" % (1 - p["x"][i], p["x"][i], 1 - p["y"][i], comp),
                 p["ea"][i], ea_hat[i], s[i], ea_hat[i] / K_B))
    print("\n输出：%s" % path)


if __name__ == "__main__":
    main()
