# RecSim 系统流程全景

> 8 个步骤，从真实数据到 10 亿用户模拟的完整闭环

---

## 总览

```
① 数据加载 ──→ ② 用户聚类 ──→ ③ 分布提取
                                    │
④ Persona生成 ←─────────────────────┘
    │
    ▼
⑤ 模拟运行 ──→ ⑥ 保真度计算
    ▲               │
    │               ▼
    └── ⑦ 校准环 ←──┘  F不达标 → 调参回⑤
                        F达标 ↓
                     ⑧ 外推 → 10亿
```

---

## ① 数据加载 — 把真实数据读进来

```
输入: KuaiRec 的 CSV 文件
输出: 统一的 DataFrame

small_matrix.csv (470万条):
  user_id=1, video_id=100, watch_ratio=0.73, play_duration=12.5s

item_categories.csv (1万条):
  video_id=100, feat=[8, 27]    ← 品类8和品类27
```

**原理:** 不同数据集格式不同，统一成 (user_id, item_id, watch_ratio, duration_ms) 的 schema，后续模块不需要关心数据源。

| 项目 | 值 |
|------|-----|
| 代码 | `baseline/loader.py` |
| 用到 LLM | 否 |
| 状态 | 已完成 |

---

## ② 用户聚类 — 把 1411 个用户分成 50 个"典型人群"

```
输入: 每个用户的行为特征
输出: 50 个 archetype（用户原型）

特征提取:
  user_1 → [avg_wr=0.68, n_items=2500, wr_std=0.25, 
            avg_dur=8500, 品类8占比=0.35, 品类27占比=0.12, ...]
  (5维基础 + 10维品类 = 15维特征向量)

StandardScaler 标准化 → KMeans(k=50)

结果:
  archetype_0: 142个用户, 特征中心=[0.72, 3000, ...]  "重度美食用户"
  archetype_1: 89个用户,  特征中心=[0.35, 500, ...]   "轻度随机用户"
  ...
```

**原理:** 1411 个用户太多不好直接建模，聚成 50 类后每类有统计意义（平均 28 人/类），可以拟合参数分布。

| 项目 | 值 |
|------|-----|
| 代码 | `baseline/clustering.py` |
| 用到 LLM | 否 |
| 状态 | 已完成 |

---

## ③ 分布提取 — 每个 archetype 的行为用数学公式描述

```
输入: 每个 archetype 里所有用户的行为数据
输出: 参数化分布 + 兴趣向量

archetype_0 的 142 个用户:
  完播率数据 → 拟合 Beta(a=2.3, b=1.8)     ← 偏右分布，多数人看完
  观看时长   → 拟合 LogNormal(μ=9.1, σ=0.6) ← 正偏态，多数短视频
  兴趣向量   → [品类8: 0.35, 品类27: 0.12, ...] ← watch_ratio加权平均

同时提取:
  真实品类消费分布 → {品类8: 23%, 品类27: 15%, ...}    ← 长尾
  真实条件分布     → P(完播|品类8)=0.72, P(完播|品类27)=0.45
```

**为什么用 Beta 拟合完播率?** 完播率天然在 [0,1] 区间，Beta 分布是 [0,1] 上最灵活的参数分布，两个参数 a、b 能描述各种形状（U 型、J 型、钟型）。

**为什么用 LogNormal 拟合时长?** 观看时长是正偏态分布（多数人看短视频，少数人看长视频），LogNormal 天然适合这种形状。

**兴趣向量怎么算?** 从用户的观看历史，用 watch_ratio 加权每个品类的曝光量，归一化为概率向量。比如 user_1 看了很多品类 8 的视频且完播率高，那他的兴趣向量里品类 8 的权重就大。

| 项目 | 值 |
|------|-----|
| 代码 | `baseline/distribution.py` + `baseline/interest.py` |
| 用到 LLM | 否 |
| 状态 | 已完成 |

---

## ④ Persona 生成 — 在高维空间最优放置 1000 个点

```
输入: 50 个 archetype 的分布描述
输出: 1000 个 persona 骨架（每个都是独特的虚拟用户）

Step 1: 按 archetype 比例分配名额
  archetype_0 占 15% → 分配 150 个 agent
  archetype_1 占 3%  → 分配 30 个 agent
  (最大余额法保证总数 = 1000)

Step 2: Latin Hypercube Sampling 生成参数
  在 (stall_tolerance, quality_sensitivity, fatigue_rate, patience) 
  4 维空间做 LHS，保证每个维度被均匀覆盖

Step 3: 从 archetype 的分布中采样个体参数
  agent_42: archetype_0
    watch_ratio_baseline = 0.68  (从 Beta(2.3,1.8) 采样)
    stall_tolerance = 1200ms     (从 LHS 采样)
    quality_sensitivity = 0.6
    interest_vector = archetype_0 的平均兴趣向量
```

**LHS 比随机采样好在哪?** 随机采样在高维空间会"扎堆"，LHS 强制在每个维度上均匀切分，用 N 个点就能保证每个维度被 N 等份覆盖。

**运筹学优化布局（设计文档规划，部分待实现）：**
- Phase A: 正交设计/LHS — 200 个点（已实现）
- Phase B: Support Points — 500 个（待实现，用 Wasserstein 距离优化）
- Phase C: 列生成 — 300 个（待实现，迭代补盲）

**可选的 LLM 步骤 — Persona 叙事（待实现）：**

```
agent_42 的骨架参数
    │
    ▼ Claude API
"你是小美，22岁，住洛阳，大专毕业在奶茶店上班。
 每天下班后刷2-3小时短视频，最爱看美食探店和旅行vlog。
 对画质不太敏感但非常讨厌卡顿，
 网络不好时会直接退出。"
```

这一步是可选的，骨架参数已经足够驱动 Layer 0 和 Layer 1 的参数化决策。LLM 叙事主要服务于 Layer 2 的 prompt 构建。

| 项目 | 值 |
|------|-----|
| 代码 | `persona/skeleton.py` |
| 用到 LLM | 可选（叙事填充，待实现） |
| 状态 | 骨架已完成，LLM 叙事待实现 |

---

## ⑤ 模拟运行 — 每个 agent 看视频做决策

```
输入: 1000 个 persona + 视频池 + 基础设施配置
输出: 行为日志（约 30,000 条记录）
```

对每个 agent 创建一个会话，对每个视频经过三层决策：

### Layer 0: 体验决策（最重要，要最准）

```
"用户因为体验问题离开，跟内容无关"

输入: 基础设施状态 + 用户容忍度
输出: watch_pct_factor (0~1 的乘数) + 是否强制退出/跳过

三个惩罚项叠加:
  stall_penalty    = 1 - exp(-0.7 * 卡顿时长 / 容忍阈值)   ← 指数衰减
  quality_penalty  = 画质敏感度 * (1 - 画质分) * 0.4
  first_frame_pen  = min(超时比例 * 0.3, 0.5)

首刷时容忍阈值减半 → 同样卡顿惩罚加倍

force_exit: 首刷 + 总惩罚 > 0.7 → 50%概率直接退出app
force_skip: 卡顿严重 → 概率性划走，不看内容
```

### Layer 1: 内容决策（对齐大盘分布）

```
"体验没问题的前提下，用户对内容的反应"

输入: 用户兴趣向量 + 视频品类向量 + L0 factor + 疲劳度
输出: watch_pct + liked/commented/shared

核心公式:
  interest = cosine(用户兴趣向量, 视频品类向量)   ← 余弦相似度
  raw_pct  = baseline + (interest-0.5)*0.4 - fatigue*0.3 + noise
  watch_pct = clip(raw_pct * L0_factor, 0, 1)

L0 和 L1 是乘法关系:
  L0_factor=0.7, L1_base=0.65 → 最终 0.455
  可拆解: "内容本身值 0.65，画质拖了 30%"
```

### Layer 2: LLM 涌现决策（待实现）

```
"多因素交织时的不确定性决策"

触发条件（只有 ~20% 的决策走这层）:
  · 首刷前 5 个视频（留存关键路径）
  · 新品类首次遭遇（兴趣探索）
  · 多因素冲突（内容好但画质差）
  · 随机 10% 抽检校验 Layer 1

调用方式:
  prompt = "你是{persona叙事}。现在推荐给你一个{品类}视频，
            画质{quality}，卡顿了{stall}次。你会怎么做？"
  → Claude API → {"watch_pct": 0.85, "reason": "虽然画质差但内容太感兴趣"}
```

### 三层联动示例

```
推荐系统选出视频 V
    │
    ▼
Layer 0: 首帧 800ms + 4G + 卡顿1次
  → stall_penalty=0.28, quality_penalty=0.05
  → watch_pct_factor = 0.72
  → 不退出，不跳过，继续
    │
    ▼
Layer 1: interest = cosine(用户, 视频) = 0.85
  → base_wr = 0.68 + 0.14 - 0.03 = 0.79
  → final_wr = 0.79 * 0.72 = 0.57
  → 看了 57%，没点赞
    │
    ▼
日志: {watch_pct=0.57, l0_factor=0.72, l1_base=0.79,
       decision_layer=1, fidelity_tag="parametric"}

可拆解:
  "如果画质正常(l0=1.0)，完播率是 79%"
  "画质问题导致完播率从 79% 降到 57%"
  "体验损失 = 28%"
```

### 成本估算

```
1000 agents × 30 videos = 30,000 decisions/day

Layer 0 规则短路:   ~10% =  3,000 次 → $0     (纯数学)
Layer 1 参数化:     ~70% = 21,000 次 → $0     (纯数学)
Layer 2 LLM:        ~20% =  6,000 次 → ~$15-45/day (Claude API)
```

| 项目 | 值 |
|------|-----|
| 代码 | `interaction/layer0.py` + `layer1.py` + `engine.py` + `runner.py` |
| 用到 LLM | Layer 2 用（待实现） |
| 状态 | L0+L1 已完成, L2 待实现 |

---

## ⑥ 保真度计算 — 模拟结果和真实数据对比

```
输入: 模拟行为日志 + 真实数据分布
输出: 多维保真度 F
```

### 5 个维度同时衡量

| 维度 | 指标 | 含义 | 最大可接受偏差 |
|------|------|------|---------------|
| 完播率分布 | JS 散度 | 分布形状是否一致 | 0.3 |
| 品类消费分布 | JS 散度 | 品类偏好是否匹配 | 0.3 |
| 条件分布 P(完播\|品类) | 平均偏差 | 各品类的完播率差异 | 0.2 |
| 活跃度分布 | Wasserstein | 活跃度分布搬土距离 | 50.0 |
| 相关性矩阵 | Frobenius 范数 | 维度间联合特征 | 2.0 |

### 各指标的原理

**KL 散度:** 信息论度量，"用分布 q 近似分布 p 需要多少额外 bit"。不对称。

**JS 散度:** KL 的对称化版本，`JS(p,q) = 0.5*KL(p,m) + 0.5*KL(q,m)`，其中 `m=(p+q)/2`。有界 [0, ln2]。

**Wasserstein 距离:** "搬土距离"——把一个分布的概率质量搬到另一个分布需要的最小代价。直觉上比 KL 更稳定。

**Frobenius 范数:** 矩阵的"欧几里得长度"，用来衡量两个相关性矩阵差了多少。

### 综合打分

```
score_i = 1 - min(raw_i / max_acceptable_i, 1.0)
F_multidim = 加权平均(所有 score_i)

F > 0.9  ✅ 高保真，可用于算法评估
F > 0.7  ⚠️ 中保真，趋势可信
F < 0.7  ❌ 需要校准
```

### 输出

每次测试自动生成 `reports/latest_report.json`，包含所有指标。Dashboard (`dashboard.html`) 自动加载渲染为交互式图表。

| 项目 | 值 |
|------|-----|
| 代码 | `fidelity/metrics.py` + `fidelity/multidim.py` + `report.py` + `dashboard.html` |
| 用到 LLM | 否 |
| 状态 | 已完成 |

---

## ⑦ 校准环 — F 不达标就自动调参重跑

```
输入: ⑥ 的保真度结果
输出: 调整后的参数 → 回到 ⑤ 重跑

三层嵌套循环:
```

### 外循环（代价最高，很少触发）

```
触发条件: 边缘分布偏差 > 15%
动作: 回到 ④ 重新布局 1000 个点（运筹学重新优化）
频率: 理想状态下只在初始化阶段跑 2-3 轮
```

### 中循环（每轮检查）

```
触发条件: 条件分布偏差 > 10%
动作: 微调 persona 骨架参数
  · 兴趣权重向量微调
  · 行为基线率校正
  · Layer 0 体验响应曲线参数修正
频率: 每轮模拟后检查一次
```

### 内循环（持续运行）

```
触发条件: 持续运行
动作: Layer 2 LLM 抽检 Layer 1 的参数化决策
  从 Layer 1 的决策中随机抽样
  问 LLM "这个用户在这个场景下真的会这样做吗？"
  如果 LLM 和参数化模型分歧大 → 调整 L1 参数
频率: 每 N 条决策抽检一次
```

### 防止过拟合

1. **留出验证集:** KuaiRec 按时间 70/30 切分，校准集 vs 验证集的 F 差异监控 overfit
2. **交叉数据集验证:** 在 KuaiRec 上校准的参数拿到 MovieLens 上检验泛化性
3. **参数正则化:** `|θ_calibrated - θ_prior| < λ * σ_prior`，防止偏离 prior 太远

### 自动修正逻辑

```
找到偏差
  → 归因（属于哪层/哪个参数）
  → 定位参数 → MLE 重拟合或梯度小步调整
  → 验证（目标改善 + 其他指标不被破坏）
```

| 项目 | 值 |
|------|-----|
| 代码 | `calibration/` (待建) |
| 用到 LLM | 内循环的 LLM 抽检 |
| 状态 | 待实现 |

---

## ⑧ 外推 — 1000 人 → 10 亿人

```
输入: 校准后的 1000 agent 行为数据
输出: 10 亿规模的虚拟流量生成器
```

### Stage 1: 拟合生成模型

```
1000 个 agent 的行为特征向量
  → 拟合 Vine Copula
    边缘分布: 非参数 KDE（不强加分布假设）
    相关性: Vine 结构建模（灵活且可解释）
    1000 样本足够拟合 vine 结构
```

**为什么选 Vine Copula?**

| 候选模型 | 优点 | 风险 |
|----------|------|------|
| 多维 GMM | 简单可解释 | 高维下组件数爆炸 |
| Vine Copula | 边缘和相关性分离 | 尾部依赖建模需经验 |
| Normalizing Flow | 任意分布都能拟合 | 1000 样本不够训练 |
| VAE | 有隐空间可插值 | 同上 |

### Stage 2: 带权采样

```
N = 10,000:
  直接采样，检查是否在训练数据覆盖范围内
  范围外的标注 fidelity=estimated

N = 10 亿:
  不需要真的生成 10 亿个独立个体
  生成 10,000-50,000 个代表性个体
  每个附带 OT 权重 wi，Σwi = 10亿
  行为由 Layer 1 参数化模型驱动（不调 LLM）
```

### Stage 3: 外推质量验证

```
1. 自洽性: 从 10 亿中随机抽 1000 个，分布 ≈ 原始 1000 agent
2. 放大稳定性: 1K/10K/100K/1M/1B 指标趋于稳定
3. 联合分布保持: ||Corr(1K) - Corr(1B)||_F < ε
```

### 输出格式

```
TrafficGenerator:
  输入:  infra_config + rec_algorithm
  输出:  {
    total_dau: 3.2亿,
    per_archetype_metrics: { user_count, avg_completion, churn_rate },
    infra_impact: { "720p→480p": { completion_delta, cost_saving } },
    fidelity: { F_overall, F_marginal, F_conditional, warning }
  }
```

| 项目 | 值 |
|------|-----|
| 代码 | `extrapolation/` (待建) |
| 用到 LLM | 否 |
| 状态 | 待实现 |

---

## LLM 在系统中的位置

整个系统 LLM 出现在 3 个地方，全部是可选的：

| 位置 | 步骤 | 作用 | 调用量 | 成本 |
|------|------|------|--------|------|
| Persona 叙事 | ④ | 把数字参数变成人物描述 | 1000 次（一次性） | ~$2 |
| Layer 2 决策 | ⑤ | 处理参数化模型拿不准的场景 | ~6000 次/天 | ~$15-45/天 |
| 校准环抽检 | ⑦ | 验证 Layer 1 决策是否合理 | ~500 次/轮 | ~$1-3/轮 |

**设计原则:** 先保证统计基座（①-⑥）是对的（纯数学，不花钱），再用 LLM 做精细化。如果统计基座本身就偏了，LLM 的判断也没有好的 baseline 来对比。

---

## 当前实测结果

### P0 阶段（修复前）

```
F_overall:  0.983  ← 只看均值，虚高
F_multidim: 0.669
  watch_ratio_js:       0.678 (分数)  raw=0.097
  category_js:          0.000 (分数)  raw=0.693  ← 品类完全对不上
  activity_wasserstein: 1.000 (分数)  raw=0.0    ← 占位值
  correlation_distance: 1.000 (分数)  raw=0.0    ← 占位值
```

### P1 阶段（修复中）

修复内容:
- runner 从真实品类分布采样（替代均匀随机）
- 用余弦相似度做真实兴趣匹配（替代 Beta(2,2) 随机数）
- 接入真实活跃度和相关性数据（替代占位值 0.0）

预期: category_js 大幅下降，F_multidim 显著提升

---

## 进度总表

| 步骤 | LLM? | 状态 | 代码 |
|------|------|------|------|
| ① 数据加载 | 否 | ✅ 完成 | `baseline/loader.py` |
| ② 用户聚类 | 否 | ✅ 完成 | `baseline/clustering.py` |
| ③ 分布提取 | 否 | ✅ 完成 | `baseline/distribution.py` + `interest.py` |
| ④ Persona 骨架 | 否 | ✅ 完成 | `persona/skeleton.py` |
| ④ Persona 叙事 | 是 | ⬜ 待实现 | `persona/narrative.py` |
| ⑤ L0 体验决策 | 否 | ✅ 完成 | `interaction/layer0.py` |
| ⑤ L1 内容决策 | 否 | ✅ 完成 | `interaction/layer1.py` |
| ⑤ L2 LLM 决策 | 是 | ⬜ 待实现 | `interaction/layer2.py` |
| ⑥ 保真度计算 | 否 | ✅ 完成 | `fidelity/` + `report.py` |
| ⑦ 校准环 | 可选 | ⬜ 待实现 | `calibration/` |
| ⑧ Vine Copula 外推 | 否 | ⬜ 待实现 | `extrapolation/` |
