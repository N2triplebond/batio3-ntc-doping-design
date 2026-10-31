# -*- coding: utf-8 -*-
"""
BaTiO3 基 NTC 陶瓷 B 位掺杂配方生成与筛选（约束枚举 + 多目标打分）

约束：
  电荷自补偿   x = Σ f_i · a_i ，a_i = 4 − z_i（受主强度）
  位点守恒     Ba: 1−x，Ti: 1−Σf
  固溶度       Σf ≤ 0.40，单组分 ≥ 0.015，主组分 ≥ 0.04
  活性组分     d⁰/d¹⁰ 等无变价对元素最多占 B 位 30 %（否则不供跳跃载流子）
  钙钛矿容差   t = (r_A + r_O) / (√2 (r_B + r_O)) ∈ [0.95, 1.06]

预测子（经验外推，必须由 DFT / 实验复核）
  Ea ≈ 0.498 − 0.028·d̄      d̄ 只对"有变价对"的元素取平均
  标定：Cr3+ (d3) → 0.414 eV（实测 0.414）；Co3+ (d6) → 0.330 eV（实测 0.295–0.370）
  不确定度约 ±0.05 eV，仅用于同族内排序，不可当绝对预测

用法：python recipe_search.py
"""
import csv
import math
import os
import itertools

K_B = 8.617333e-5
R_TI = 0.0605
R_O = 0.140
R_BA, R_LA = 0.161, 0.136
OUT = os.path.dirname(os.path.abspath(__file__))
STEP = 0.005
F_MIN = 0.015
F_MAJOR_MIN = 0.04
F_TOTAL_MAX = 0.40
ACTIVE_MIN_RATIO = 0.70

# 价态 z、离子半径 r(nm, CN=6)、d 电子数、异价对、工艺风险(0-1)、备注
EL = {
    "Cr": dict(z=3, r=0.0615, d=3, pair="Cr3+/Cr4+", risk=0.35, note="高温挥发 CrO/CrO2"),
    "Fe": dict(z=3, r=0.0645, d=5, pair="Fe2+/Fe3+", risk=0.10, note="—"),
    "Mn": dict(z=3, r=0.0645, d=4, pair="Mn3+/Mn4+", risk=0.55, note="Jahn-Teller 畸变强"),
    "V":  dict(z=3, r=0.0640, d=2, pair="V3+/V4+",  risk=0.65, note="易挥发，价态难控"),
    "Ga": dict(z=3, r=0.0620, d=10, pair="无",      risk=0.40, note="d10 全满，只调晶格"),
    "Co": dict(z=3, r=0.0580, d=6, pair="Co3+/Co2+", risk=0.20, note="低自旋半径偏小"),
    "Ni": dict(z=2, r=0.0690, d=8, pair="Ni2+/Ni3+", risk=0.35, note="需 2:1 配比"),
    "Cu": dict(z=2, r=0.0730, d=9, pair="Cu+/Cu2+",  risk=0.75, note="易还原挥发"),
}
for v in EL.values():
    v["a"] = 4 - v["z"]
    v["active"] = v["pair"] != "无"

X_GRID = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]


def gen_fracs(elems, x):
    """枚举满足 Σf·a = x 的分数组合"""
    n = len(elems)
    out = []

    def dfs(i, remaining, cur):
        if i == n:
            if abs(remaining) < 1e-9 and cur:
                out.append(dict(cur))
            return
        e = elems[i]
        a = EL[e]["a"]
        rest_max = sum(EL[elems[j]]["a"] for j in range(i + 1, n)) * F_TOTAL_MAX
        f = F_MIN
        while f <= F_TOTAL_MAX + 1e-9:
            rem = remaining - a * f
            if rem < -1e-9:
                break
            if rem > rest_max + 1e-9:
                f = round(f + STEP, 6)
                continue
            if sum(cur.values()) + f > F_TOTAL_MAX + 1e-9:
                break
            cur[e] = round(f, 6)
            dfs(i + 1, rem, cur)
            cur.pop(e)
            f = round(f + STEP, 6)

    dfs(0, x, {})
    return out


def evaluate(x, fracs):
    y = sum(fracs.values())
    w_act = sum(f for e, f in fracs.items() if EL[e]["active"])
    if w_act < ACTIVE_MIN_RATIO * y:
        return None
    d_bar = sum(f * EL[e]["d"] for e, f in fracs.items() if EL[e]["active"]) / w_act
    r_bar = sum(f * EL[e]["r"] for e, f in fracs.items()) / y
    dr = abs(r_bar - R_TI) / R_TI
    r_a = (1 - x) * R_BA + x * R_LA
    t = (r_a + R_O) / (math.sqrt(2) * (r_bar + R_O))
    if not (0.95 <= t <= 1.06):
        return None
    if max(fracs.values()) < F_MAJOR_MIN:
        return None

    ea = 0.498 - 0.028 * d_bar
    b = ea / K_B
    p = [f / y for f in fracs.values()]
    ent = -sum(q * math.log(q) for q in p if q > 0)
    risk = sum(f * EL[e]["risk"] for e, f in fracs.items()) / y

    f1 = max(0.0, 1 - abs(ea - 0.32) / 0.10)          # 活化能窗（B ≈ 3700 K 最优）
    f2 = max(0.0, 1 - max(0.0, dr - 0.02) / 0.07)     # 尺寸失配
    f3 = 1.0 if t < 1.045 else max(0.0, 1 - (t - 1.045) / 0.03)
    f4 = max(0.0, 1 - risk)                            # 工艺风险
    f5 = min(1.0, w_act / y)                           # 活性组分占比
    f6 = min(1.0, ent / math.log(3))                   # 熵（3 组分封顶，避免痕量凑数）
    score = 32 * f1 + 22 * f2 + 12 * f3 + 16 * f4 + 12 * f5 + 6 * f6

    order = sorted(fracs.items(), key=lambda kv: -kv[1])
    formula = "Ba%.3gLa%.3gTi%.3g" % (1 - x, x, 1 - y) + "".join(
        "%s%.3g" % (e, f) for e, f in order) + "O3"
    return dict(x=x, fracs=fracs, y=y, d_bar=d_bar, r_bar=r_bar, dr=dr, t=t,
                ea=ea, b=b, ent=ent, risk=risk, score=score, formula=formula,
                n_el=len(fracs), dominant=order[0][0],
                rho=math.exp(1.12 + 25.2 * ea))


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for x in X_GRID:
        for n in (1, 2, 3, 4):
            for combo in itertools.combinations(list(EL), n):
                for fr in gen_fracs(list(combo), x):
                    r = evaluate(x, fr)
                    if r:
                        rows.append(r)
    rows.sort(key=lambda r: -r["score"])
    print("通过全部硬约束的配方数：%d" % len(rows))

    # 按主导元素分组，每组取前 3，保证多样性
    by_dom, picked = {}, []
    for r in rows:
        k = r["dominant"]
        by_dom.setdefault(k, [])
        if len(by_dom[k]) < 3:
            by_dom[k].append(r)
            picked.append(r)
    picked.sort(key=lambda r: -r["score"])

    path = os.path.join(OUT, "top_recipes.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["排名", "配方", "x", "主导元素", "组分数", "Ea预测/eV", "B预测/K",
                    "尺寸失配", "容差因子", "构型熵", "工艺风险", "总分", "B位组成"])
        for i, r in enumerate(picked, 1):
            bstr = " ".join("%s%.3f" % (e, f) for e, f in
                            sorted(r["fracs"].items(), key=lambda kv: -kv[1]))
            w.writerow([i, r["formula"], "%.2f" % r["x"], r["dominant"], r["n_el"],
                        "%.3f" % r["ea"], "%5.0f" % r["b"], "%.3f" % r["dr"],
                        "%.4f" % r["t"], "%.2f" % r["ent"], "%.2f" % r["risk"],
                        "%.1f" % r["score"], bstr])

    print("\n%-3s %-40s %4s %6s %6s %6s %6s" % ("#", "配方", "x", "Ea", "B/K", "Δr", "得分"))
    for i, r in enumerate(picked[:16], 1):
        print("%-3d %-40s %.2f %6.3f %6.0f %6.3f %6.1f" %
              (i, r["formula"], r["x"], r["ea"], r["b"], r["dr"], r["score"]))

    print("\n各主导元素的组内最优（用于判断模型是否给出已知结论）:")
    for e, lst in sorted(by_dom.items()):
        r = lst[0]
        print("  %-3s → %-42s Ea=%.3f  B=%.0f" % (e, r["formula"], r["ea"], r["b"]))

    print("\n锚点核对（预测 vs 实验）:")
    for name, x, ea_exp, fr in [
            ("xLCF-BT [1]", 0.15, 0.414, {"Cr": 0.105, "Fe": 0.045}),
            ("xLCO-BCTS [4]", 0.15, 0.330, {"Co": 0.15})]:
        r = evaluate(x, fr)
        print("  %-14s 实验 %.3f  预测 %.3f  偏差 %+.0f%%" %
              (name, ea_exp, r["ea"], (r["ea"] / ea_exp - 1) * 100))

    print("\n输出：%s" % path)


if __name__ == "__main__":
    main()
