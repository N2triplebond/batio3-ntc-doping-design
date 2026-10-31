# -*- coding: utf-8 -*-
"""Materials Project 接入桥（零依赖：只用标准库 urllib，不装 pymatgen/mp-api）

用途分工（重要，别混）：
  MP 能给的  → 端元/有序掺杂相的 formation_energy_per_atom、energy_above_hull、
               band_gap、体积、磁构型；用来做
                 (a) 候选配方的"已知相查重"：这个化学体系 MP 里到底有没有东西
                 (b) 自有 DFT 形成能的独立交叉校验（不同泛函/U 口径，只看趋势）
                 (c) 带隙量级的旁证
  MP 不能给的 → 变温电阻、活化能 Ea、小极化子跳跃势垒、真实固溶体的无序构型。
               所以 MP 数据【不进】Ea 标签，S_φ 的残差目标仍只能来自自有 DFT/实验。

用法：
  set MP_API_KEY=xxxx                     (Windows cmd)  /  export MP_API_KEY=xxxx (bash)
  python src/mp_bridge.py pull            # 拉取并缓存到 data/mp/
  python src/mp_bridge.py pull --chemsys Ba-Ti-O,La-Cr-O
  python src/mp_bridge.py screen          # 给 results/nn_proposals_v3.csv 打 MP 覆盖度

API key 申请：https://materialsproject.org/api  （登录后在 Dashboard 生成，免费账号即可）
"""
import argparse
import csv
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.materialsproject.org/materials/summary/"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MPDIR = os.path.join(ROOT, "data", "mp")

# 与本文体系相关的化学空间：端元 + 施主/受主 + 共掺三元
CHEMSYS = [
    "Ba-Ti-O",      # BaTiO3 各相：容差因子与带隙的锚点
    "Ba-Cr-O", "Ba-Fe-O", "Ba-Mn-O",
    "La-Ti-O", "La-Cr-O", "La-Fe-O",
    "Ba-Ti-Cr-O", "Ba-Ti-Fe-O", "Ba-Ti-Mn-O", "Ba-La-Ti-O",
]

FIELDS = ["material_id", "formula_pretty", "chemsys", "nsites", "volume", "density",
          "symmetry", "formation_energy_per_atom", "energy_above_hull", "is_stable",
          "band_gap", "is_gap_direct", "is_metal", "total_magnetism", "ordering",
          "elements"]


def get_key():
    k = os.environ.get("MP_API_KEY", "").strip()
    if not k:
        sys.exit("缺少 MP_API_KEY。申请：https://materialsproject.org/api\n"
                 "  bash: export MP_API_KEY=xxx    cmd: set MP_API_KEY=xxx")
    return k


def query(chemsys, key, limit=1000, max_total=6000, fields=True):
    """按化学体系分页拉取；fields=True 时用精简字段，400 则退回全字段"""
    out, skip, t0 = [], 0, time.time()
    while skip < max_total:
        params = {"chemsys": chemsys, "_limit": limit, "_skip": skip,
                  "_sort_fields": "material_id"}
        if fields:
            params["_fields"] = ",".join(FIELDS)
        url = API + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url)
        req.add_header("X-API-KEY", key)
        req.add_header("Accept", "application/json")
        # Cloudflare 会拦 urllib 默认 UA（Error 1010），必须伪装成普通客户端
        req.add_header("User-Agent", UA)
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                doc = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:300]
            if e.code == 400 and fields:
                print("   字段被拒(400)，退回全字段重试: %s" % body)
                return query(chemsys, key, limit, max_total, fields=False)
            if e.code in (401, 403):
                sys.exit("认证失败(%d)：%s\n  → 检查 MP_API_KEY 是否正确/是否已激活" % (e.code, body))
            if e.code == 429:
                print("   限流，等 5 s 重试")
                time.sleep(5)
                continue
            sys.exit("请求失败(%d) %s: %s" % (e.code, url, body))
        data = doc.get("data", [])
        out.extend(data)
        if len(data) < limit:
            break
        skip += limit
        time.sleep(0.3)
    print("   %-14s %4d 条  %.1f s" % (chemsys, len(out), time.time() - t0))
    return out


def flat(rec):
    sym = rec.get("symmetry") or {}
    if not isinstance(sym, dict):
        sym = {}
    return {
        "material_id": rec.get("material_id"),
        "formula": rec.get("formula_pretty"),
        "chemsys": rec.get("chemsys"),
        "nsites": rec.get("nsites"),
        "crystal_system": sym.get("crystal_system"),
        "spacegroup": sym.get("symbol"),
        "sg_number": sym.get("number"),
        "Ef_per_atom": rec.get("formation_energy_per_atom"),
        "E_hull": rec.get("energy_above_hull"),
        "is_stable": rec.get("is_stable"),
        "band_gap": rec.get("band_gap"),
        "gap_direct": rec.get("is_gap_direct"),
        "is_metal": rec.get("is_metal"),
        "mag": rec.get("total_magnetism"),
        "ordering": rec.get("ordering"),
        "volume": rec.get("volume"),
        "density": rec.get("density"),
    }


def cmd_pull(args):
    key = get_key()
    os.makedirs(MPDIR, exist_ok=True)
    csl = [c.strip() for c in (args.chemsys or ",".join(CHEMSYS)).split(",") if c.strip()]
    rows, ok = [], 0
    for cs in csl:
        try:
            recs = query(cs, key, max_total=args.max)
            ok += 1
        except SystemExit:
            raise
        except Exception as e:                     # 单个体系失败不拖垮整体
            print("   %-14s 失败: %s" % (cs, e))
            continue
        with open(os.path.join(MPDIR, "raw_%s.json" % cs), "w", encoding="utf-8") as fh:
            json.dump(recs, fh, ensure_ascii=False, indent=1)
        for r in recs:
            rows.append(flat(r))
    if not rows:
        sys.exit("没有拿到任何数据（%d/%d 个体系成功）" % (ok, len(csl)))
    p = os.path.join(MPDIR, "summary.csv")
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n合计 %d 条 → %s" % (len(rows), p))


SYM = re.compile(r"([A-Z][a-z]?)([0-9.]*)")


def chemsys_of(formula):
    return "-".join(sorted({m.group(1) for m in SYM.finditer(formula)}))


def parse_counts(formula):
    """'Ba0.8La0.2Ti0.8Fe0.195Cr0.005O3' → {'Ba':0.8,'La':0.2,...}"""
    out = {}
    for m in SYM.finditer(formula):
        el, n = m.group(1), m.group(2)
        try:
            v = float(n) if n else 1.0
        except ValueError:
            v = 1.0
        out[el] = out.get(el, 0.0) + v
    return out


# 单掺端元锚点：受主/施主单独进 BaTiO3 时的"最接近凸包的有序相"所在体系
ANCHOR = {"Cr": "Ba-Cr-O-Ti", "Fe": "Ba-Fe-O-Ti", "Mn": "Ba-Mn-O-Ti", "La": "Ba-La-O-Ti"}


def _num(x, default=None):
    try:
        if x in ("", None):
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def load_index():
    p = os.path.join(MPDIR, "summary.csv")
    if not os.path.exists(p):
        sys.exit("还没有缓存，先跑 python src/mp_bridge.py pull")
    by = {}
    with open(p, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            by.setdefault(r["chemsys"], []).append(r)
    return by


def featurize(formula, by):
    """单条配方的 MP 派生量。注意：这些量描述的是热力学/电子结构，
    与 Ea 没有物理对应，只能当弱先验，别当标签。"""
    cs = chemsys_of(formula)
    target = set(cs.split("-"))
    pool = by.get(cs, [])
    # 元素集最接近的体系（Jaccard 最大），用于"没有同体系相时退而求其次"
    best_j, best_cs = 0.0, ""
    for k in by:
        s = set(k.split("-"))
        j = len(s & target) / max(1, len(s | target))
        if j > best_j:
            best_j, best_cs = j, k
    near = by.get(best_cs, []) if best_cs else []
    ref = pool if pool else near

    e_hull = [_num(x["E_hull"]) for x in ref]
    e_hull = [v for v in e_hull if v is not None]
    ef = [_num(x["Ef_per_atom"]) for x in ref]
    ef = [v for v in ef if v is not None]
    gaps = [_num(x["band_gap"]) for x in ref if str(x.get("is_metal")) != "True"]
    gaps = [v for v in gaps if v is not None]
    ns = [_num(x["nsites"]) for x in ref]
    ns = [int(v) for v in ns if v]
    best = min(ref, key=lambda x: _num(x["E_hull"], 9e9)) if ref else None

    d = {
        "化学体系": cs,
        "MP同体系相数": len(pool),
        "最近邻体系": best_cs if not pool else cs,
        "体系Jaccard": round(best_j, 3) if not pool else 1.0,
        "最低E_hull_eV": round(min(e_hull), 4) if e_hull else "",
        "最低Ef_eV_per_atom": round(min(ef), 4) if ef else "",
        "最小带隙_eV": round(min(gaps), 3) if gaps else "",
        "最大带隙_eV": round(max(gaps), 3) if gaps else "",
        "稳定相数": sum(1 for x in ref if str(x.get("is_stable")) == "True"),
        "最可算超胞原子数": min(ns) if ns else "",
        "对标相": best["formula"] if best else "",
    }
    # 端元锚点：每个掺杂元素对应"单掺 BaTiO3"体系的最稳相，按组成加权
    cnt = parse_counts(formula)
    num = den = 0.0
    gaps_anchor = []
    for el, csel in ANCHOR.items():
        if cnt.get(el, 0.0) <= 0:
            continue
        ap = by.get(csel, [])
        ae = [_num(x["E_hull"]) for x in ap]
        ae = [v for v in ae if v is not None]
        ag = [_num(x["band_gap"]) for x in ap]
        ag = [v for v in ag if v is not None]
        eh = min(ae) if ae else None
        d["端元%s_E_hull" % el] = round(eh, 4) if eh is not None else ""
        d["端元%s_最小带隙" % el] = round(min(ag), 3) if ag else ""
        d["端元%s_相数" % el] = len(ap)
        if eh is not None:
            num += cnt[el] * math.exp(-max(0.0, eh) / 0.10)
            den += cnt[el]
        if ag:
            gaps_anchor.append(min(ag))
    d["S_端元"] = round(num / den, 3) if den > 0 else 0.0
    if gaps_anchor:
        gm = min(gaps_anchor)
        d["S_带隙"] = 1.0 if 0.2 <= gm <= 3.2 else (0.5 if gm > 3.2 else 0.0)
    else:
        d["S_带隙"] = 0.0
    d["S_覆盖"] = 1.0 if len(pool) >= 1 else 0.0
    d["S_MP"] = round(0.55 * d["S_端元"] + 0.25 * d["S_带隙"] + 0.20 * d["S_覆盖"], 3)
    return d


def cmd_screen(args):
    by = load_index()
    src = args.input or os.path.join(ROOT, "results", "nn_proposals_v3.csv")
    with open(src, encoding="utf-8-sig") as fh:
        props = list(csv.DictReader(fh))
    print("候选 %d 条，来源 %s" % (len(props), os.path.basename(src)))

    out = []
    for i, r in enumerate(props[: args.top]):
        f = r.get("配方") or r.get("formula") or ""
        d = featurize(f, by)
        row = {"排名": r.get("排名", i + 1), "配方": f}
        row.update(d)
        out.append(row)

    cov = sum(1 for r in out if r["MP同体系相数"])
    print("MP 有同化学体系的: %d / %d" % (cov, len(out)))
    q = os.path.join(MPDIR, "coverage_%s" % os.path.basename(src))
    with open(q, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print("→ %s" % q)
    for r in out[:args.show]:
        print("  #%-3s %-40s %-18s 相数%-4s E_hull=%-8s S_MP=%s"
              % (r["排名"], r["配方"], r["化学体系"], r["MP同体系相数"],
                 r["最低E_hull_eV"] or "-", r["S_MP"]))


# 本文最需要拿来当 DFT 参考起点的几个相
REF_IDS = {
    "mp-aaaaaiof": "BaTiO3 Amm2（稳定相，带隙 2.29 eV）",
    "mp-aaaaaeli": "BaTiO3 Pm-3m（立方顺电，建超胞常用）",
    "mp-aaaaaiwg": "BaTiO3 P4mm（四方铁电）",
    "mp-aaacrwxr": "Ba3TiFe2O9（Fe 掺体系最稳定相，E_hull=0）",
    "mp-aaacrxil": "Ba2Ti4Cr2O13（Cr 掺体系唯一相，E_hull=+7 meV）",
    "mp-aaacrxco": "Ba2TiMnO6（Mn 掺体系）",
    "mp-aaacqicx": "BaLa2Ti3O10（La 体系层状相）",
}


def fetch_structure(mid, key):
    # 注意：路径式 /materials/summary/{id} 已被废弃（403 deprecated），
    # 参数名是 material_ids（复数）；material_id 单数会 400。
    url = API + "?" + urllib.parse.urlencode(
        {"material_ids": mid, "_fields": "material_id,formula_pretty,symmetry,structure"})
    req = urllib.request.Request(url)
    req.add_header("X-API-KEY", key)
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", UA)
    with urllib.request.urlopen(req, timeout=90) as r:
        doc = json.loads(r.read().decode())
    d = doc.get("data") or []
    if not d:
        raise RuntimeError("查不到 %s" % mid)
    return d[0]


def to_poscar(rec):
    """MP structure → VASP POSCAR 文本（直接坐标，无 pymatgen）"""
    st = rec["structure"]
    M = st["lattice"]["matrix"]
    sites = st["sites"]
    order, groups = [], {}
    for s in sites:
        el = s["species"][0]["element"]
        occu = float(s["species"][0].get("occu", 1.0))
        if occu < 0.999:
            raise RuntimeError("含部分占据(%.2f)，不适合直接出 POSCAR" % occu)
        if el not in groups:
            order.append(el)
            groups[el] = []
        groups[el].append(s["abc"])
    lines = ["%s  %s  MP %s" % (rec.get("formula_pretty"), REC_TAG, rec.get("material_id")),
             "1.0"]
    for v in M:
        lines.append("  %14.10f %14.10f %14.10f" % tuple(v))
    lines.append("  " + " ".join(order))
    lines.append("  " + " ".join(str(len(groups[e])) for e in order))
    lines.append("Direct")
    for e in order:
        for abc in groups[e]:
            lines.append("  %14.10f %14.10f %14.10f" % tuple(abc))
    return "\n".join(lines) + "\n", order, groups


REC_TAG = "ref"


def cmd_poscar(args):
    key = get_key()
    ids = [x.strip() for x in (args.ids or ",".join(REF_IDS)).split(",") if x.strip()]
    outdir = os.path.join(MPDIR, "poscar")
    os.makedirs(outdir, exist_ok=True)
    for mid in ids:
        try:
            rec = fetch_structure(mid, key)
            txt, order, groups = to_poscar(rec)
        except SystemExit:
            raise
        except Exception as e:
            print("  %-14s 跳过: %s" % (mid, e))
            continue
        sym = (rec.get("symmetry") or {})
        name = "%s_%s.vasp" % (mid, rec.get("formula_pretty", "x").replace(" ", ""))
        p = os.path.join(outdir, name)
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(txt)
        print("  %-14s %-16s sg=%-8s %d 原子 → %s"
              % (mid, rec.get("formula_pretty"), sym.get("symbol"),
                 sum(len(groups[e]) for e in order), name))
    print("→ %s" % outdir)


def cmd_prior(args):
    """把 MP 特征与 S_MP 先验分合并进候选表，输出 results/mp_prior_v3.csv"""
    by = load_index()
    src = args.input or os.path.join(ROOT, "results", "nn_proposals_v3.csv")
    with open(src, encoding="utf-8-sig") as fh:
        rd = csv.DictReader(fh)
        props = list(rd)
        cols = list(rd.fieldnames or [])
    rows = []
    for i, r in enumerate(props):
        f = r.get("配方") or r.get("formula") or ""
        d = featurize(f, by)
        row = dict(r)
        row.update(d)
        rows.append(row)
    extra = [c for c in rows[0].keys() if c not in cols]
    q = os.path.join(ROOT, "results", "mp_prior_%s" % os.path.basename(src))
    with open(q, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols + extra)
        w.writeheader()
        w.writerows(rows)
    vals = [r["S_MP"] for r in rows]
    print("S_MP: 均值 %.3f  最高 %.3f  为 0 的 %d / %d"
          % (sum(vals) / len(vals), max(vals), sum(1 for v in vals if v == 0), len(vals)))
    print("→ %s" % q)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("pull", help="拉取并缓存 MP 数据")
    p1.add_argument("--chemsys", help="逗号分隔，默认内置 11 个体系")
    p1.add_argument("--max", type=int, default=6000, help="单体系上限")
    p1.set_defaults(fn=cmd_pull)
    p2 = sub.add_parser("screen", help="给候选配方打 MP 覆盖度")
    p2.add_argument("--input")
    p2.add_argument("--top", type=int, default=50)
    p2.add_argument("--show", type=int, default=12)
    p2.set_defaults(fn=cmd_screen)
    p3 = sub.add_parser("prior", help="合并 MP 特征+先验分到候选表")
    p3.add_argument("--input")
    p3.set_defaults(fn=cmd_prior)
    p4 = sub.add_parser("poscar", help="导出参考相的 POSCAR（DFT 初始构型）")
    p4.add_argument("--ids", help="逗号分隔 material_id，默认 7 个参考相")
    p4.set_defaults(fn=cmd_poscar)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
