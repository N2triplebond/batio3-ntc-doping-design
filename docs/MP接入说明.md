# Materials Project 接入说明

> 目的：把 Materials Project（下称 MP）作为 **DFT 数据参考**接进来 —— 参考的是端元与掺杂相的结构、形成能、带隙，不是模型的输入输出。
> 结论先放这里：**MP 能当 DFT 的参照基准、能支撑三条论证，但不能给这批候选排序。** 下面是数据与依据。

---

## 1. 接入方式

MP 的 REST API 没有匿名访问（无 key 一律 401），需要免费账号的 API key：

1. 注册/登录 <https://materialsproject.org>，在 Dashboard → API 生成 key；
2. 只通过环境变量传入，不写进任何文件：

```bash
# bash
export MP_API_KEY=your_key_here
# Windows cmd
set MP_API_KEY=your_key_here
```

3. 运行（零第三方依赖，只用标准库，不装 pymatgen / mp-api）：

```bash
python src/mp_bridge.py pull                 # 拉 11 个相关化学体系 → data/mp/
python src/mp_bridge.py pull --chemsys Ba-Cr-O-Ti,La-Fe-O
python src/mp_bridge.py screen               # 候选 × MP 覆盖度 → data/mp/coverage_*.csv
python src/mp_bridge.py prior                # 合并 MP 特征+先验分 → results/mp_prior_*.csv
```

**坑**：`api.materialsproject.org` 前置 Cloudflare，会按 UA 拦截 urllib 默认头（`Error 1010: Access denied`），必须带浏览器 UA。脚本已处理。

---

## 2. 拉到了什么

11 个化学体系，共 **223 条** materials（`data/mp/summary.csv`，原始 JSON 在 `data/mp/raw_*.json`）：

| 化学体系 | 相数 | 与本文的关系 |
|---|---|---|
| Ba-O-Ti | 53 | BaTiO₃ 各相，带隙/容差因子锚点 |
| Ba-Mn-O | 41 | Mn 端元 |
| Ba-Fe-O | 32 | Fe 端元 |
| La-O-Ti | 30 | La 端元 |
| Ba-Cr-O | 28 | Cr 端元 |
| Fe-La-O | 14 | LaFeO₃ 等 |
| Cr-La-O | 8 | LaCrO₃ 等 |
| **Ba-Fe-O-Ti** | **7** | Fe 掺 BaTiO₃ 有序相 |
| **Ba-La-O-Ti** | **7** | La 掺 BaTiO₃（层状相） |
| Ba-Mn-O-Ti | 2 | Mn 掺 BaTiO₃ |
| **Ba-Cr-O-Ti** | **1** | Cr 掺 BaTiO₃（几乎空白） |

---

## 3. 负结果：先验打分不成立

生成栈 v3 的 20 条候选，**全部**落在同一个化学体系 `Ba-Cr-Fe-La-O-Ti`，而 MP 在这个体系里 **0 条记录**。

退而求其次做了两块替代打分（端元锚点 + 带隙），实测：

| 指标 | 值 |
|---|---|
| MP 同体系覆盖率 | **0 / 20** |
| S_MP 极差 | **0.005**（0.519 ~ 0.524） |
| S_MP 不同取值 | 6 种（20 条里） |

**0.005 的极差等于没有区分度**——排第一和排第二十的 MP 支持度一样。原因不是算法写坏了，而是 MP 里根本没有"La+Cr+Fe 共掺 BaTiO₃"的数据，任何由它派生的分数都只能是常数。

由此两条结论：
- ❌ **不进总评分**：加进去只会给排序制造虚假精度；
- ❌ **不进 `features()`**：60k 枚举样本里绝大多数查询不到同体系数据，这一维近乎常量，对残差网络是纯噪声。

---

## 4. 真正的价值：三条可写进论文/答辩的论据

### 4.1 端元可溶性——Cr、Fe 单掺在热力学上站得住

| 相 | material_id | E_hull (eV/atom) | 带隙 (eV) | 空间群 | 原子数 |
|---|---|---|---|---|---|
| Ba₂Ti₄Cr₂O₁₃ | mp-aaacrxil | **+0.0073** | 1.84 | C2/m | 42 |
| Ba₃TiFe₂O₉ | mp-aaacrwxr | **0.000** | 0.25 | P6₃mc | 30 |
| Ba₂TiFeO₆ | mp-aaacrxjk | +0.0151 | 0.00 | P3m1 | 30 |
| Ba₂TiMnO₆ | mp-aaacrxco | +0.0194 | 1.29 | R-3m | 40 |

Cr、Fe 单掺 BaTiO₃ 都存在紧贴凸包（E_hull ≤ 7 meV）的有序相 → 两种受主都能进晶格，这是我们做共掺的前置条件，而不是区分候选的依据。

**顺带一个硬事实**：`Ba-Cr-O-Ti` 全库只有 **1 条**，`Ba-Cr-Fe-La-O-Ti` **0 条**。Cr 共掺 BaTiO₃ 的有序结构在公开数据库里近乎空白 → 本文的 DFT 形成能（+0.73/+0.78 eV Ti-rich、−0.60/−0.55 eV Ti-poor）属于增量数据，不是重复劳动。这句话可以直接用在引言或答辩。

### 4.2 带隙塌缩——Fe 比 Cr 更容易把体系压成窄带

- BaTiO₃（Amm2，mp-aaaaaiof，稳定）：带隙 **2.29 eV**；其它多形体 1.67–2.55 eV
- Ba₃TiFe₂O₉：带隙 **0.25 eV** ← Fe 掺入后几乎金属化
- Ba₂Ti₄Cr₂O₁₃：带隙 **1.84 eV** ← Cr 掺入后只轻微压窄

与"Fe 引入深能级、漏导上升"的实验直觉一致，可作为 Fe:Cr 配比讨论的旁证（注意：这是有序相的计算值，不是固溶体，只能看趋势）。

### 4.3 La 的上限有独立旁证

`Ba-La-O-Ti` 的 7 条记录**全是层状 RP 相**，没有一条是 La 在 A 位的简单钙钛矿固溶体：

| 相 | E_hull (eV/atom) | 带隙 (eV) |
|---|---|---|
| BaLa₂Ti₃O₁₀（mp-aaacqicx） | +0.0100 | 1.99 |
| BaLa₄Ti₄O₁₅ | +0.0305 | 2.89 |
| Ba₂La₄Ti₅O₁₈ | +0.0801 | 2.39 |

说明高 La 量在热力学上倾向形成层状第二相而非无限固溶 → 支持模型里 **x ≤ 0.20** 的上限不是拍出来的。

### 4.4 结构文件：可直接当 DFT 初始构型

`data/mp/poscar/` 下已导出 7 个参考相的标准 POSCAR（Direct 坐标，VASP 直接可用）：

| 文件 | 相 | 空间群 | 原子数 |
|---|---|---|---|
| `mp-aaaaaiof_BaTiO3.vasp` | BaTiO₃（稳定相） | Amm2 | 5 |
| `mp-aaaaaeli_BaTiO3.vasp` | BaTiO₃（立方顺电） | Pm-3m | 10 |
| `mp-aaaaaiwg_BaTiO3.vasp` | BaTiO₃（四方铁电） | P4mm | 5 |
| `mp-aaacrwxr_Ba3TiFe2O9.vasp` | Fe 掺体系最稳定相 | P6₃mc | 30 |
| `mp-aaacrxil_Ba2Ti4Cr2O13.vasp` | Cr 掺体系唯一相 | C2/m | 42 |
| `mp-aaacrxco_Ba2TiMnO6.vasp` | Mn 掺体系 | R-3m | 40 |
| `mp-aaacqicx_BaLa2Ti3O10.vasp` | La 体系层状相 | Ama2 | 32 |

用途：建超胞时不必从文献手抄晶格常数；Cr/Fe 有序相可作为"取代位是否合理"的对照构型（Ba₂Ti₄Cr₂O₁₃ 里 Cr 明确占 Ti 位）。

```bash
python src/mp_bridge.py poscar                    # 导出 7 个默认参考相
python src/mp_bridge.py poscar --ids mp-xxxx,yyyy # 指定 material_id
```

（导出前会检查占据数，含部分占据的条目直接跳过——无序固溶相不能直接当 POSCAR 用。）


---

## 5. 与本文 DFT 的对照关系（能对什么、不能对什么）

| 量 | 能不能对 | 说明 |
|---|---|---|
| BaTiO₃ 晶格常数 / 体积 | ✅ | MP Amm2 相 a=3.983 Å、b≈4.057 Å、V=65.57 Å³；与自有 U=3 静态结果比，偏差 >2% 说明参数有问题 |
| 带隙量级 | ⚠️ 只看趋势 | MP 2.29 eV vs 实验 3.2 eV，GGA 系统性低估约 30% |
| 掺杂相"是否存在/是否近凸包" | ✅ | 判断某元素能不能进晶格（Ba₂Ti₄Cr₂O₁₃ E_hull = +7 meV → 能） |
| 缺陷形成能绝对值与符号 | ❌ | MP 给的是"相"的总能；本文 +0.73/−0.60 eV 是"缺陷"形成能，化学势与参考态口径完全不同，不能相减 |
| 活化能 Eₐ / 跳跃势垒 | ❌ | MP 无输运数据 |

**一条可执行的自检**：拿 `mp-aaaaaiof_BaTiO3.vasp` 用本文 INCAR（U=3、ENCUT 500）跑一次静态，比较晶格常数与总能量。晶格常数偏差在 1% 内、总能差在每原子 0.05 eV 内，说明两套计算同一量级，后续交叉引用才站得住；否则只能各自表述。

## 6. 使用限制（别越界）

1. **口径不同，绝对值不可比**：MP 是混合泛函/U 的数据库，与本文 U_Ti=3.0 / U_Cr=3.7 的形成能不可直接相减，只能比趋势。
2. **无输运量**：MP 没有变温电阻、没有活化能 Ea、没有小极化子跳跃势垒 → S_φ 的残差目标仍只能来自自有 DFT/实验。
3. **数据质量要过滤**：MP 里 BaTiO₃ 有多条 `band_gap = 0.0` 的条目（低精度或金属化误判），统计带隙时必须剔掉 `is_metal` 与非最低精度条目。
4. **有序相 ≠ 固溶体**：所有对照都是有序超胞，代表不了真实固溶体的无序构型与局域畸变。

---

## 7. 复现产物

| 文件 | 内容 |
|---|---|
| `data/mp/summary.csv` | 223 条材料的扁平化字段（Excel 友好，utf-8-sig） |
| `data/mp/raw_<chemsys>.json` | 原始 API 返回，便于离线复查 |
| `data/mp/poscar/*.vasp` | 7 个参考相的标准 POSCAR（DFT 初始构型） |
| `data/mp/coverage_nn_proposals_v3.csv` | 20 条候选的 MP 覆盖度与端元锚点 |
| `results/mp_prior_nn_proposals_v3.csv` | 候选表 + MP 特征 + S_MP（含各分项） |
