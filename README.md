# ntc-formula-screening

BaTiO₃ 基宽温 NTC 热敏电阻 **B 位多元素掺杂配方的机器学习反设计**——生成式反演 + 残差校正闭环的模型代码与结果。

> 本仓库**只包含模型代码、算法说明与模型输出**（CSV/Markdown）。
> **不包含任何 DFT 计算数据**：没有 INCAR / POSCAR / POTCAR / KPOINTS / OUTCAR / OSZICAR / DOSCAR / CHGCAR / WAVECAR，也没有结构文件与电荷密度。DFT 部分另行管理。

---

## 1 要解决的问题

目标是在 `xLaCr₀.₇Fe₀.₃O₃-(1−x)BaTiO₃` 体系基础上，为 B 位寻找 **La 施主 + Cr/Fe/Mn 受主** 的多元掺杂配方，使其

- 活化能 Eₐ 落在 0.28–0.42 eV（对应 B 值 3200–4900 K），目标值 **0.315 eV / B ≈ 3654 K**；
- 同时满足容差因子、尺寸失配、掺杂量、电荷与位点守恒等硬约束。

难点在于**没有数据**：可用的实测点只有 4 个组分加 2 个文献体系，共 6 个。用 6 个样本训练"组分 → 性能"的预测器，得到的只是插值噪声。

## 2 方法：不把网络当预测器

因此网络被放在两个更能发挥长处的位置（详见 [`docs/算法说明.md`](docs/算法说明.md)）：

1. **反向生成（G_θ）**——给定目标 Eₐ 与掺杂量，直接生成满足全部物理约束的组分，而不是先预测再筛选；
2. **残差校正（S_φ）**——只学"规则先验错了多少"，不重学物理。

```
                    ┌──────────────── G_θ ────────────────┐
 (Ea_t, x_t, z) ──► │ MLP(8→64→64) ──► logits(7), gate      │
                    │      softmax q          s=0.02+0.38σ(g)│
                    │            └──── f = s·q ────┘         │
                    │   x := Σ fᵢaᵢ ；Ti := 1 − Σ fᵢ          │
                    └───────────────────┬───────────────────┘
                                        │ 候选池
                    ┌───────────────────▼───────────────────┐
                    │ S_φ：14 维特征 → MLP → ΔEₐ（K=5 集成） │
                    │ Eₐ = Eₐ,prior(f) + mean_k ΔEₐ,k        │
                    └───────────────────────────────────────┘
```

两个关键设计：

- **亚随机参数化**：不用纯 softmax（`Σf ≡ 1` 会把配方退化为"B 位全取代"），改为 `f = s·q`，`s ∈ [0.02, 0.40]` 作可学习门控；
- **守恒闭式消去**：电荷与位点守恒不写成罚项，而由 `x = Σfᵢaᵢ`、`Ti = 1 − Σfᵢ` 直接给出，构造上严格成立。

## 3 目录

```
src/
  nn_design_v3.py    当前版本：生成器 + 残差校正闭环（元素池收窄至 Cr/Fe）
  nn_design_v2.py    上一版：元素池含 Cu/Ni/Mn/V
  nn_design.py       初版
  recipe_search.py   暴力枚举基线（用于对照与生成先验表）
  mp_bridge.py       Materials Project 参考库桥接（零依赖，见 §6）
data/
  top_recipes.csv    暴力枚举输出的候选表
  mp/                MP 副本（summary.csv + raw_*.json，可离线复查）
results/
  nn_proposals_v3.csv   v3 输出（含 Eₐ 先验/修正、±σ、B、失配、容差因子、构型熵、Cr:Fe）
  nn_proposals_v2.csv   v2 输出
  nn_proposals_v1.csv   v1 输出
  mp_prior_*.csv        候选表 + MP 特征 + S_MP（§6）
docs/
  算法说明.md         推导、参数、损失与训练细节
  配方汇总_v3.md      v3 结果整理与解读
  配方汇总_v2.md      v2 结果整理
  MP接入说明.md       MP 数据统计、负结果与三条可用论据
```

## 4 运行

```bash
pip install -r requirements.txt

# 当前版本（默认 --demo，闭环 4 轮、每轮 10 条、集成 5 个、400 epochs）
python src/nn_design_v3.py

# 可选参数
python src/nn_design_v3.py --rounds 4 --batch 10 --epochs 400 --acq ucb

# 暴力枚举基线
python src/recipe_search.py
```

输出写到**脚本所在目录**（`OUT = os.path.dirname(os.path.abspath(__file__))`），不依赖任何本机绝对路径。

## 5 结果摘要（v3）

Top-1：`Ba₀.₈La₀.₂Ti₀.₈Fe₀.₁₉₅Cr₀.₀₀₅O₃`，Eₐ 修正后 **0.319 eV**（先验 0.359），B ≈ **3702 K**，±σ = 0.037。

需要如实说明的两点：

- v3 把元素池收窄到 **{Cr, Fe}** 后，模型偏 Fe（Cr:Fe 中位数 ≈ 0.03），与实验锚定的 **7:3 反向**。原因是先验只含 d 电子计数、不含具体元素化学，比例必须由实验锚定；
- 训练目标由**合成 oracle**（先验 + 三项交叉项 + 0.012 eV 噪声）给出，闭环验证的是"通路能跑通、约束能守住"，不是真实性能提升。

## 6 Materials Project 参考库（可选）

需要 MP API key（`export MP_API_KEY=...`，不写入文件）：

```bash
python src/mp_bridge.py pull     # 11 个相关化学体系 → data/mp/（223 条）
python src/mp_bridge.py screen   # 候选 × MP 覆盖度
python src/mp_bridge.py prior    # 候选表 + MP 特征 + S_MP
python src/mp_bridge.py poscar   # 7 个参考相 → POSCAR（DFT 初始构型）
```

定位是 **DFT 数据参考**：参考端元与掺杂相的结构、形成能、带隙，不作为模型输入。

**结论是负结果，先看这里再决定要不要用**：20 条候选全在 `Ba-Cr-Fe-La-O-Ti`，MP 该体系 **0 条记录**，由它派生的 S_MP 极差仅 **0.005**，无区分度 —— 因此 MP 分数**不进总评分、不进 `features()`**，避免制造虚假精度。

它的实际用途是三条论据（详见 `docs/MP接入说明.md`）：

1. **端元可溶性**：Ba₂Ti₄Cr₂O₁₃（E_hull = +7 meV）、Ba₃TiFe₂O₉（E_hull = 0）→ Cr/Fe 单掺热力学站得住；同时 `Ba-Cr-O-Ti` 全库仅 **1 条**，说明 Cr 共掺 BaTiO₃ 有序结构近乎空白，本文 DFT 属增量数据；
2. **带隙塌缩**：BaTiO₃ 2.29 eV → Ba₃TiFe₂O₉ 0.25 eV、Ba₂Ti₄Cr₂O₁₃ 1.84 eV，Fe 压窄带隙远强于 Cr；
3. **La 上限旁证**：`Ba-La-O-Ti` 7 条记录全是层状 RP 相（BaLa₂Ti₃O₁₀ 等），无 A 位固溶相 → 支持 x ≤ 0.20。

MP 无输运量（无变温电阻、无 Eₐ、无跳跃势垒），且与本文 U=3 口径不同，只能比趋势、不能比绝对值。

## 7 环境

Python 3.9+，仅依赖 `numpy`。无 GPU、无深度学习框架（反向传播与解析梯度手写，见 `src/nn_design_v3.py` 中的 `backward`）。MP 桥接只用标准库。

## 8 版本演进

| 版本 | 变化 |
|---|---|
| v1 | 生成器 + 残差校正雏形；元素池 7 元 |
| v2 | 采样加入剪枝与电荷重投影；输出不覆盖 v1 |
| v3 | 元素池收窄至 {Cr, Fe}；对每条候选补 Cr:Fe 比，暴露比例偏差 |

## 9 引用与致谢

基于已发表体系 `xLaCr₀.₇Fe₀.₃O₃-(1−x)BaTiO₃`（J. Alloys Compd. 1037 (2025) 182661）扩展。方法上参考 MatterGen（反向生成）与 MatterSim（小样本微调）的思路。

---

*本仓库为在研工作，尚未发表。未经许可请勿转载或用于衍生发表。*
