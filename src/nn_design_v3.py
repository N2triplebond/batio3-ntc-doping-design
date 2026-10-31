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
  AL   主动学习    采样 → UCB 采集 → DFT 打标 → 只更新残差网络
       注意：G 只在第 [2/4] 段训练一次，闭环内不重训 G

v3 相对 v2 的改动（2026-10-28 18:00，DFT 数据落地后重跑）：
  A. 元素池由 DFT 收敛结果决定：仅有 Cr_Ti / Fe_Ti 完成 U=3 形成能计算
     （E_f = +0.73 / +0.78 eV，Ti-poor 端 −0.60 / −0.55 eV，二者仅差 50 meV），
     其余 Mn/V/Co/Ni/Cu 无 DFT 支撑，本轮从 B 位池中移除（POOL）
  B. 生成器 softmax 直接加 logit 掩码，保证采样与可行域都在池内
  C. 输出改为 nn_proposals_v3.csv，并对每条候选补 Cr:Fe 比

v2 相对 v1 的改动（2026-10-28，仅修 bug，不改物理与协议）：
  1. 集成每轮从零初始化，与"重训"的表述一致（v1 是热启动，权重跨轮累积）
  2. 留出集改为随机 20% 且不参与训练（v1 取训练集末尾、且在训练集内，属样本内评估）
  3. 第 1 轮集成未训练，UCB 等价随机数，显式改为随机取点并打印
  4. 训练日志返回六项总损失（v1 只返回 Ea 项，看不出几何项是否在罚）
  5. 标准化对常数维（act 恒为 1）不再用 1e-9 放大成数值噪声
  6. 掺杂量上限统一成 T_MAX，剪枝存活率、可行率都打印
  7. 删掉死变量 Ea0、废参数 tol、未使用的 log
  8. 输出改为 nn_proposals_v2.csv，不覆盖 v1 结果

用法：
  python nn_design_v2.py                    # 合成 oracle 验证闭环
  python nn_design_v2.py --rounds 4 --batch 10 --epochs 400
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
T_MAX = 0.40                            # B 位总取代量上限，可行域与生成器共用同一个数
EA_TGT = 0.32                           # 目标活化能，采样、采集、打分共用

EL_NAMES = ["Cr", "Fe", "Mn", "V", "Co", "Ni", "Cu"]   # Ga 无变价对，不进导电掺杂池
POOL = ["Cr", "Fe"]                      # DFT 已验证可掺的 B 位受主（U=3 形成能已收敛）
POOL_IDX = np.array([EL_NAMES.index(e) for e in POOL])
LOGIT_MASK = np.full(len(EL_NAMES), -1e9)
LOGIT_MASK[POOL_IDX] = 0.0               # 池外元素 softmax 权重恒为 0
Z = np.array([3, 3, 3, 3, 3, 2, 2], dtype=float)
RAD = np.array([0.0615, 0.0645, 0.0645, 0.0640, 0.0580, 0.0690, 0.0730])
D_EL = np.array([3, 5, 4, 2, 6, 8, 9], dtype=float)
RISK = np.array([0.35, 0.10, 0.55, 0.65, 0.20, 0.35, 0.75])
A_EL = 4.0 - Z                          # 受主强度
M_ACT = np.ones(7)


# ---------------- 物理层 ----------------
def phys(f):
    """f: (B, 7) 组分分数 → 派生量"""
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


def feasible(f):
    p = phys(f)
    return ((p["y"] <= T_MAX) & (p["y"] >= 0.02) & (p["t"] >= 0.95) & (p["t"] <= 1.06)
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

    SMIN, SMAX = 0.02, T_MAX

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
        logits = h2 @ W3 + b3 + LOGIT_MASK
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
        # 返回六项总损失（v1 只返回 Ea 项，看不出几何项是否在罚）
        L = ((p["ea"] - target) ** 2
             + w_dr * np.maximum(0.0, p["dr"] - 0.06) ** 2
             + w_t * np.maximum(0.0, np.abs(p["t"] - 1.0) - 0.05) ** 2
             + w_x * (p["x"] - xtgt) ** 2
             + w_y * (np.maximum(0.0, p["y"] - 0.34) ** 2 + np.maximum(0.0, 0.03 - p["y"]) ** 2)
             - w_ent * p["H"])
        return float(L.mean())


# ---------------- 训练数据：规则枚举器 ----------------
def enum_dataset(x_list=(0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20),
                 step=0.02, fmin=0.02, cap=60000, seed=RNG_SEED):
    """在约束面内随机采样可行组分（用规则模型当"免费标签"）"""
    rng = np.random.default_rng(seed)
    F = []
    while len(F) < cap:
        x = rng.choice(x_list)
        k = rng.integers(1, len(POOL_IDX) + 1)
        idx = rng.choice(POOL_IDX, k, replace=False)
        w = rng.dirichlet(np.ones(k))
        f = np.zeros(len(EL_NAMES))
        a = A_EL[idx]
        scale = x / (w @ a)                     # 强制满足 Σf·a = x
        f[idx] = w * scale
        f = np.round(f / step) * step
        f[f < fmin] = 0.0
        f = np.round(f, 4)
        if f.sum() <= 0 or f.sum() > T_MAX:
            continue
        F.append(f)
    F = np.array(F)
    ok = feasible(F)
    print("      %d 条初筛 → %d 条可行（%.1f%%）" % (len(F), int(ok.sum()), 100.0 * ok.mean()))
    return F[ok]


def prune_topk(F, xtgt, k=3, ymax=T_MAX):
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
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--acq", choices=["ucb", "target"], default="ucb",
                    help="ucb: acq = μ − 1.5σ（v1 行为）；target: acq = σ − |Êa − EA_TGT|，围绕目标采点")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    # 1) 先验数据集 + 生成器训练
    print("[1/4] 构造可行域数据集 ...")
    F0 = enum_dataset()
    print("      可行组分 %d 条" % len(F0))
    Xin = features(F0)
    mu = Xin.mean(0)
    sd = Xin.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)   # 常数维（第 14 维 act 恒为 1）不再被放大成噪声

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
    tgt = np.full(20000, EA_TGT)
    xts = rng.choice([0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20], 20000)
    z = rng.normal(0, 1, (20000, G.nz))
    Fs = G.forward(tgt, xts, z)
    n_prune = len(prune_topk(Fs, xts, k=3))
    Fs = prune_topk(Fs, xts, k=3)
    n_ok = int(feasible(Fs).sum()) if len(Fs) else 0
    if len(Fs):
        Fs = Fs[feasible(Fs)]
    # 去重（0.001 栅格）
    Fs = np.unique(np.round(Fs, 3), axis=0)
    print("      采样 20000 → 剪枝存活 %d（%.1f%%）→ 可行 %d（%.1f%%）→ 去重 %d"
          % (n_prune, 100.0 * n_prune / 20000, n_ok,
             100.0 * n_ok / max(n_prune, 1), len(Fs)))

    # 3) 残差代理 + 主动学习
    print("[4/4] 残差代理 + 主动学习 ...")
    oracle = make_oracle()
    labels_f, labels_y = [], []
    ens = None
    for rd in range(args.rounds):
        Xs = (features(Fs) - mu) / sd
        if ens is None:
            # 第 1 轮集成还没训练，UCB 只是一串随机数；这里显式改成随机取点
            order = rng.permutation(len(Fs))[:args.batch]
            print("      round 1: 集成未训练，随机取 %d 个作为初始标签" % len(order))
        else:
            m_, s_, acq_ucb = ucbs(ens, Xs)
            if args.acq == "target":
                ea_pred = phys(Fs)["ea"] + m_          # 先验 + 残差均值
                acq_use = s_ - np.abs(ea_pred - EA_TGT)
            else:
                acq_use = acq_ucb
            order = np.argsort(acq_use)[::-1][:args.batch]
        picks = Fs[order]
        labels_f.append(picks)
        labels_y.append(oracle(picks))
        Ftr = np.vstack(labels_f)
        Ytr = np.concatenate(labels_y)
        Xall = (features(Ftr) - mu) / sd
        prior_all = phys(Ftr)["ea"]
        resid_all = Ytr - prior_all

        # 真·留出：随机取 20%，且这些行完全不参与训练
        n_all = len(Xall)
        n_ho = max(2, n_all // 5)
        perm = np.random.default_rng(1000 + rd).permutation(n_all)
        ho_idx, tr_idx = perm[:n_ho], perm[n_ho:]

        # 每轮从零初始化，不做热启动
        ens = [MLP(Xin.shape[1], seed=10 + i) for i in range(5)]
        for ep in range(args.epochs):
            for mdl in ens:
                idx = rng.permutation(len(tr_idx))
                for b in range(0, len(idx), 16):
                    j = tr_idx[idx[b:b + 16]]
                    out = mdl.forward(Xall[j])
                    mdl.grads = mdl.backward(2 * (out - resid_all[j]) / len(j))
                    mdl.step()

        pred = np.mean([mdl.forward(Xall) for mdl in ens], 0) + prior_all
        d_tr = pred[tr_idx] - Ytr[tr_idx]
        d_ho = pred[ho_idx] - Ytr[ho_idx]
        rmse = float(np.sqrt((d_tr ** 2).mean()))
        rmse_ho = float(np.sqrt((d_ho ** 2).mean()))
        print("      round %d: 标签 %3d（训练 %2d / 留出 %2d）  训练RMSE=%.4f  留出RMSE=%.4f eV"
              % (rd + 1, n_all, len(tr_idx), len(ho_idx), rmse, rmse_ho))

    # 4) 输出配方
    Xs = (features(Fs) - mu) / sd
    m, s, acq = ucbs(ens, Xs)
    p = phys(Fs)
    ea_hat = p["ea"] + m
    score = (EA_TGT - np.abs(ea_hat - EA_TGT)) / 0.10
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

    path = os.path.join(OUT, "nn_proposals_v3.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["排名", "配方", "x", "Ea先验", "Ea修正后", "±σ", "B预测/K",
                    "尺寸失配", "容差因子", "构型熵", "B位组成", "Cr:Fe"])
        for r, i in enumerate(out, 1):
            comp = " ".join("%s%.3f" % (EL_NAMES[k], Fs[i][k])
                            for k in np.argsort(-Fs[i]) if Fs[i][k] > 0)
            formula = "Ba%.3gLa%.3gTi%.3g" % (1 - p["x"][i], p["x"][i], 1 - p["y"][i]) + \
                      "".join("%s%.3g" % (EL_NAMES[k], Fs[i][k])
                              for k in np.argsort(-Fs[i]) if Fs[i][k] > 0) + "O3"
            w.writerow([r, formula, "%.3f" % p["x"][i], "%.3f" % p["ea"][i],
                        "%.3f" % ea_hat[i], "%.3f" % s[i], "%5.0f" % (ea_hat[i] / K_B),
                        "%.3f" % p["dr"][i], "%.4f" % p["t"][i], "%.2f" % p["H"][i], comp,
                        "%.2f" % (Fs[i][0] / max(Fs[i][1], 1e-9))])

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
