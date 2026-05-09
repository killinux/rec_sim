# F_multidim 五维保真度指标详解

> 更新时间: 2026-05-08
> 当前最佳: F_multidim = 0.630 (R8, 校准后)

---

## 总览

F_multidim 由 5 个维度的加权平均构成，每个维度衡量模拟和真实数据在某个方面的相似程度。

```
              衡量什么          raw    分数   权重   max_acceptable
watch_ratio_js   完播率分布形状     0.192  0.36  1.0    0.3
category_js      品类消费比例       0.048  0.84  1.0    0.3
activity_wass    用户活跃度差异     0.128  0.57  1.0    0.3
correlation      特征关联结构       0.560  0.72  1.0    2.0
conditional_rank 品类偏好排序       0.569  0.62  0.5    1.5
```

### 计分公式

每个维度的分数:

```
score = 1.0 - min(raw / max_acceptable, 1.0)
```

- raw = 0 → 满分 1.0（完美匹配）
- raw >= max_acceptable → 0 分（完全不像）

F_multidim = 所有维度分数的加权平均。

---

## 维度 1: watch_ratio_js — 完播率分布形状

### 衡量什么

模拟用户的完播率（watch_pct）分布和真实用户长得像不像。不是比平均值，是比整个分布的形状。

### 为什么需要

两个平均完播率都是 0.6 的系统可能完全不同:
- 系统 A: 双峰分布（大量 0.1 和 0.9，平均 0.6）→ 用户要么看完要么秒划
- 系统 B: 钟型分布（全堆在 0.5-0.7，平均 0.6）→ 所有人都看一半

对推荐系统来说，"有多少人看完"和"有多少人秒划"的比例直接影响留存和 DAU 预测。只看均值会丢掉这些信息。

### 怎么算

```
代码: src/rec_sim/report.py:29-40

1. 收集模拟侧数据
   sim_watch_pcts = [l["watch_pct"] for l in watch_logs]
   → 所有 action=="watch" 的 log 里的 watch_pct（每次看视频的完播率, 0-1）

2. 做直方图
   sim_wr_hist = np.histogram(sim_watch_pcts, bins=20, range=(0,1), density=True)
   → 把 0-1 切成 20 个 bin（每个宽 0.05），统计每个 bin 的概率密度

3. 真实侧: 从每个 archetype 的 Beta 分布采样
   real_samples = concat([d.sample_watch_ratios(200) for d in distributions])
   → 20 个 archetype × 200 = 4000 个样本，同样做 20-bin 直方图

4. 两个直方图算 JS 散度
   js = js_divergence(real_wr_hist, sim_wr_hist)
   → JS=0 完全一样，越大越不像
```

### watch_pct 的产生过程

```
代码: src/rec_sim/interaction/layer1.py

# 速决机制（双峰的来源）
if interest > 0.65 and random < 0.3:
    raw_pct = uniform(0.85, 1.0)       # 30% 概率 "hooked"
elif interest < 0.25 and random < 0.2:
    raw_pct = uniform(0.0, 0.12)       # 20% 概率 "秒划"
else:
    # 参数化路径（钟型的来源）
    raw_pct = base + interest_boost - fatigue + noise

watch_pct = clip(raw_pct × l0_factor, 0, 1)
```

### 当前状态

raw=0.192, 分数=0.36。5 个维度里最差的。模拟偏钟型，真实偏双峰。

---

## 维度 2: category_js — 品类消费分布

### 衡量什么

模拟用户看各品类视频的比例和真实用户是否匹配。

### 为什么需要

如果真实用户 30% 看美食、5% 看科技，但模拟出来各品类平均分配，那:
- A/B test 里算某品类的完播率提升时，品类流量占比就不对
- 外推到 10 亿用户时分品类的 DAU 预测会有偏

### 怎么算

```
代码: src/rec_sim/fidelity/multidim.py:7-25

1. 统计真实侧每个品类的观看次数占比
   real_dist = {美食: 0.30, 科技: 0.15, 搞笑: 0.25, ...}

2. 统计模拟侧每个品类的观看次数占比
   sim_dist = {美食: 0.28, 科技: 0.14, 搞笑: 0.27, ...}

3. 两个占比向量算 JS 散度
```

### 当前状态

raw=0.048, 分数=0.84。很好。因为模拟直接用了 KuaiRec 的品类分布 `category_distribution` 做采样权重，所以天然接近。

---

## 维度 3: activity_wasserstein — 用户活跃度差异

### 衡量什么

模拟用户之间的活跃度差异（有人是重度用户、有人是轻度用户）和真实用户是否匹配。

### 为什么需要

真实用户有重度/轻度之分（有人平均看完 90%，有人平均只看 20%）。如果模拟出来所有人都差不多，外推到 10 亿用户时的分群就不准:
- Heavy (WR>70%): 高价值用户，贡献大部分时长
- Light (WR<30%): 流失风险高
- 分群比例影响 LTV 计算和运营策略评估

### 怎么算

```
代码: src/rec_sim/fidelity/multidim.py:77-94
      src/rec_sim/report.py:159-183

1. 模拟侧: 每个 agent 所有视频的平均 watch_pct
   agent_wrs = {agent_0: [0.8, 0.3, 0.7, ...], agent_1: [...], ...}
   sim_user_avg_wr = [mean(agent_0), mean(agent_1), ...]  → 100 个值

2. 真实侧: 从每个 archetype 的 Beta 分布采样 per-user 平均完播率
   real_user_avg_wr = concat([d.sample_watch_ratios(n_users) for d in dists])

3. 均值中心化（去掉 session 长度导致的水平差异）
   real_centered = real - real.mean()
   sim_centered = sim - sim.mean()

4. 算 Wasserstein 距离（地球搬运距离）
   → 衡量"要搬多少土才能把一个分布变成另一个"
```

### 为什么要均值中心化

KuaiRec 是全观测数据集（用户看了所有视频），平均 WR≈0.7。模拟里用户只看 50 个视频且有疲劳衰减，平均 WR≈0.5。直接比 Wasserstein 会因为均值差 0.2 而距离很大，但这不是模型的问题——是数据集性质不同。中心化后只比分布形状（方差、偏度）。

### 当前状态

raw=0.128, 分数=0.57。R5 时因均值偏移崩到 0，加了均值中心化后修好。

---

## 维度 4: correlation_distance — 特征相关性结构

### 衡量什么

用户行为特征之间的关联关系是否被模拟保留。

### 为什么需要

真实用户的行为特征之间有结构性关联:
- "看得多的人完播率也高" → 正相关
- "完播率方差大的人平均完播率低" → 负相关（口味挑剔的人）
- "看得多的人完播率方差大" → 正相关（看得多样）

外推时 GMM 建模的是联合分布，如果相关性结构错了，采样出来的 10 亿"虚拟用户"的特征组合就不真实（比如出现"看很多但每个都只看 5%"的不存在的用户类型）。

### 怎么算

```
代码: src/rec_sim/fidelity/multidim.py:97-112

1. 真实侧: 取 KuaiRec 用户的特征矩阵（3 列: avg_wr, wr_std, interaction_count）
   real_corr = np.corrcoef(real_features.T)  → 3×3 相关系数矩阵

2. 模拟侧: 每个 agent 的同样 3 个指标
   sim_corr = np.corrcoef(sim_features.T)    → 3×3 相关系数矩阵

3. 两个矩阵做差，算 Frobenius 范数
   dist = ||real_corr - sim_corr||_F
   → 矩阵里每个元素差值的平方和开根号

4. 除以维度数归一化
   normalized_distance = dist / n_features
```

### 当前状态

raw=0.56, 分数=0.72。还行但有提升空间。

---

## 维度 5: conditional_rank_dist — 品类偏好排序

### 衡量什么

各品类的平均完播率排序是否一致。不比绝对值（因为量级不同），只比"哪个品类完播率更高"的相对排序。

### 为什么需要

KuaiRec 全观测数据里各品类完播率≈0.85-0.95（用户被迫看了所有视频）。模拟里≈0.4-0.6（用户可以跳过）。绝对值不可比，但排序应该一致:
- 如果真实数据里用户最爱看美食、最不爱看新闻
- 模拟里也应该是美食完播率最高、新闻最低

这对"推荐某品类的视频更多后效果如何"的实验至关重要。

### 怎么算

```
代码: src/rec_sim/fidelity/multidim.py:28-74

1. 真实侧: 每个品类的平均 watch_ratio
   real_means = {美食: 0.92, 科技: 0.85, 搞笑: 0.95, 新闻: 0.78, ...}

2. 模拟侧: 每个品类的平均 watch_pct
   sim_means = {美食: 0.58, 科技: 0.52, 搞笑: 0.61, 新闻: 0.45, ...}

3. Spearman 秩相关
   rho = spearmanr(real_means_排序, sim_means_排序)
   → rho=1 完全同序, rho=0 无关, rho=-1 完全反序

4. 转成距离
   raw = 1 - rho  → 0=完美排序, 2=完全反向
```

### 为什么用 Spearman 不用 Pearson

Spearman 只看排名不看数值，不受量级差异影响。两个数据集的完播率量级差 0.3，但只要"哪个品类完播率更高"的排序对了就给高分。

### 为什么权重是 0.5

全观测数据集的 conditional WR pattern 和模拟 app 行为有本质区别（用户看所有视频 vs 可以跳过）。品类排序不可能完全匹配，所以降权避免这个结构性限制过度拉低总分。

### 当前状态

raw=0.57 (rho≈0.43), 分数=0.62。中等，品类排序部分正确。

---

## F_multidim 计算

```
代码: src/rec_sim/fidelity/multidim.py:115-167

F_multidim = weighted_average(scores, weights)

其中:
  scores = [0.36, 0.84, 0.57, 0.72, 0.62]
  weights = [1.0,  1.0,  1.0,  1.0,  0.5]

F_multidim = (0.36×1 + 0.84×1 + 0.57×1 + 0.72×1 + 0.62×0.5) / (1+1+1+1+0.5)
           = 2.80 / 4.5
           = 0.622  (校准前)
           → 校准后 0.630
```

---

## 优化历程

```
基线  0.497 ──→ R8  0.630  (+27%)

最大贡献:
  activity:    0.000 → 0.574  (+0.574)  ← 均值中心化修复
  conditional: 0.000 → 0.621  (+0.621)  ← Spearman + DeepSeek LLM
  category:    0.835 → 0.840  (稳定)    ← 已经很好

当前瓶颈:
  watch_ratio_js: 0.360 → 0.192 (反而变差)
  → 50 videos 让分布形状差异更明显
  → 需要从产生机制上解决（KDE 采样或更多 LLM）
```

---

## 代码文件索引

| 文件 | 作用 |
|------|------|
| `src/rec_sim/fidelity/multidim.py` | 5 个维度的计算函数 + F_multidim 汇总 |
| `src/rec_sim/fidelity/metrics.py` | 基础统计工具: KL, JS, Wasserstein, composite |
| `src/rec_sim/report.py` | 从模拟结果生成报告，调用 multidim 计算 |
| `src/rec_sim/interaction/layer1.py` | watch_pct 的产生（影响 watch_ratio_js） |
| `src/rec_sim/runner.py` | 模拟主循环，产生 logs |
| `src/rec_sim/calibration/loop.py` | 校准环，迭代调参提升 F_multidim |
