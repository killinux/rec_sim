# RecSim 优化路线图

> 基于 MVP Pipeline 实测数据（2026-05-07）的问题分析和优化方向

---

## 一、当前已知问题

### 问题清单

| 问题 | 严重度 | 影响 | 修复难度 |
|------|--------|------|---------|
| 校准环 F 下降 0.489→0.427 | 高 | 越调越差 | 中 |
| conditional_delta=0.278 | 高 | 品类完播率偏差28% | 中 |
| watch_ratio_js=0.184 | 中 | 分布形状不够像 | 中 |
| videos_per_session=10 | 中 | 活跃度量级差330倍 | 低（改配置） |
| MIND数据集未下载 | 低 | 少一个数据源 | 低 |

### 根因分析

#### 校准环 F 下降

问题不只是 learning_rate 太大，有三个结构性原因：

**1. 调参方向是全局的**

当前实现对所有 archetype 的 Beta 参数朝同一方向调。应该按 archetype 分别调——品类8偏低就只调品类8相关的 archetype，而不是所有人一起动。

**2. 缺少梯度信号**

当前只知道"偏了多少"（delta），不知道"哪个参数导致的偏差"。应该对每个参数做灵敏度分析：改一点参数看 F 变多少，沿梯度方向调。

**3. 没有回滚机制**

当前调完就用，即使 F 下降也继续下一轮。应该每步保存参数快照，F 下降就回滚到上一步并缩小 lr。

#### conditional_delta=0.278 的语义问题

```
真实数据: P(完播|品类8)=0.92, P(完播|品类11)=0.94 ...
模拟数据: P(完播|品类8)=0.56, P(完播|品类11)=0.62 ...
```

差距约 0.3，但原因不是模型不准：

- KuaiRec 是**全观测数据集**，用户看了"所有"视频包括不感兴趣的 → 真实 watch_ratio 反映的是"被迫看完"的比例
- 模拟里用户可以跳过不感兴趣的 → 更像真实 app 行为
- 两者度量的东西**不完全一样** → 需要在 report 层做语义对齐，或者用 KuaiRec big_matrix（稀疏版本，更接近真实 app）

---

## 二、优化方向

### 方向 1: 数据层 — 更准的 ground truth

**当前状态：** 只用 KuaiRec small_matrix (1411 用户 × 3327 视频，全观测)

**问题：** 全观测数据集的行为模式和真实 app 不同（用户看了所有视频，而不是只看推荐的）

**优化项：**

| 项目 | 说明 | 价值 |
|------|------|------|
| KuaiRec big_matrix | 7176 用户，稀疏矩阵，更接近真实 app | 高 |
| KuaiRand 数据集 | 同团队出的随机曝光数据集，无偏反馈 | 高 |
| MIND 数据集 | 新闻推荐，补充时序行为 | 中 |
| 自有 app 数据 | 最好的校准源，如果有的话 | 最高 |

**KuaiRand 特别值得关注：** 它是随机曝光的数据（用户看到的是随机视频而不是算法推荐的），所以用户反馈是无偏的。用它做 ground truth 比 KuaiRec 的全观测矩阵更合理。

### 方向 2: 模型层 — 更准的行为模拟

**当前状态：** Beta + LogNormal 参数化 + L0×L1 乘法关系

#### 2.1 Layer 0 响应函数精细化

```
当前: 通用 sigmoid 曲线，所有人群共享
应该: 按 (网络类型 × 设备档次 × 时段) 分别拟合

例如:
  wifi + 高端机 + 晚间: 卡顿容忍极低，1秒就划走
  4G + 中端机 + 通勤: 卡顿容忍中等，3秒才划走
  3G + 低端机: 习惯了，5秒才划走

数据来源: 行业 QoE 报告 (Akamai/Conviva/ITU-T P.1203)
```

#### 2.2 Layer 1 兴趣模型增强

```
当前: 余弦相似度，一维匹配分
问题: 只看品类重叠，不看更细粒度的偏好

优化:
  - 协同过滤信号: "类似用户喜欢的视频，你也可能喜欢"
  - Embedding-based 匹配: 视频和用户都映射到低维空间
  - 时间衰减: 最近看的品类权重更高，早期的衰减
```

#### 2.3 兴趣漂移建模

```
当前: 兴趣向量是静态的，整个会话不变
问题: 连续看3个美食视频后，用户会审美疲劳

应该:
  每看一个视频后更新兴趣向量:
  interest_vector[food] *= decay_factor  (刚看过，暂时不想再看)
  interest_vector[tech] *= boost_factor  (好久没看，好奇心上升)
```

#### 2.4 社交效应

```
当前: 完全忽略社交
数据: KuaiRec 有 social_network.csv（用户好友关系）

可以建模:
  - 朋友分享的视频 → 完播率加成 (社交信任)
  - 热门视频(多人看过) → 从众效应
  - 社交触发的会话 (朋友分享链接进入) vs 自然打开
```

#### 2.5 首刷专项建模

```
当前: 首刷只是 stall_tolerance × 0.5
问题: 太简化。首刷的前5个视频决定留存，需要精细分阶段

应该:
  第1个视频: 容忍度最低，期望最高
  第2-3个: 如果第1个体验好，容忍度稍微回升
  第4-5个: 开始建立使用习惯
  → 阶梯式容忍度函数，而不是一刀切 × 0.5
```

### 方向 3: 校准层 — 更聪明的调参

#### 3.1 Per-archetype 校准

```
当前: 所有 archetype 统一调
应该: 每个 archetype 独立调参

流程:
  for each archetype:
    archetype_agents = [a for a in agents if a.archetype_id == arch_id]
    archetype_logs = [l for l in logs if l.agent_id in archetype_agents]
    archetype_delta = real_wr[arch_id] - sim_wr[arch_id]
    adjust(archetype.beta_a, archetype.beta_b, archetype_delta)
```

#### 3.2 有限差分梯度估计

```
对每个参数 θ_i:
  F_plus = simulate(θ_i + ε)
  F_minus = simulate(θ_i - ε)
  gradient_i = (F_plus - F_minus) / (2 * ε)

然后沿梯度方向更新:
  θ_new = θ_old + lr * gradient

好处: 知道每个参数对 F 的贡献，不会盲调
坏处: 每个参数要跑 2 次模拟，参数多时很慢
折中: 只对最重要的 5-10 个参数做梯度估计
```

#### 3.3 回滚 + Learning Rate Scheduling

```
best_params = current_params.copy()
best_f = current_f

for iteration in range(max_iter):
    new_params = adjust(current_params)
    new_f = evaluate(new_params)
    
    if new_f > best_f:
        best_params = new_params
        best_f = new_f
    else:
        current_params = best_params  # 回滚
        lr *= 0.5                     # 缩小步长
        
    if lr < min_lr:
        break  # 收敛
```

#### 3.4 LLM 内循环（校准抽检）

```
从 Layer 1 的参数化决策中随机抽 100 条
对每条问 DeepSeek:
  "这个用户(profile)在这个场景(context)下看了45%，合理吗？"
  → LLM 回答 "不合理，这个用户很喜欢美食，应该看80%+"

如果 LLM 和 L1 分歧大（|LLM_wr - L1_wr| > 0.2）:
  → 说明 L1 的参数在这个区域不准
  → 调整对应 archetype 的参数

费用: 100 次 DeepSeek 调用 ≈ ¥0.05/轮
```

#### 3.5 多目标优化

```
当前: F_multidim 是简单加权平均
问题: 一个维度改善但另一个恶化，平均分不变 → 看不出来

应该: Pareto 前沿
  同时优化多个维度，找到不可再改善的点集
  用户选择自己关心的 tradeoff 点
```

### 方向 4: 外推层 — 更好的分布建模

#### 4.1 Vine Copula 替代 GMM

```
当前: GMM (每个 component 假设高斯)
问题: 真实行为分布不是高斯的（比如完播率是 U 型或 J 型）

Vine Copula:
  边缘分布: 非参数 KDE（不假设任何形状）
  相关性: vine 结构（灵活建模尾部依赖）
  
  好处: 边缘和相关性分离建模，各自用最合适的方法
  需要: pip install pyvinecopulib
```

#### 4.2 条件采样（反事实推断）

```
当前: 从 GMM 独立采样，生成"随机"的 10 亿用户
应该: 支持条件采样

例如: "如果把所有用户升级到 WiFi，行为会怎样？"
  → 固定 network=wifi，采样其他维度
  → 对比 current vs counterfactual 的聚合指标
  → 这是因果推断的方向
```

#### 4.3 外推稳定性验证

```
当前: 只做了 1 次 self-consistency check
应该:
  1. Bootstrap 100 次，报告置信区间
  2. 1000 → 1万 → 10万 → 100万 → 10亿 递进验证
     每个量级的分布指标应该趋于稳定
  3. 画出 "样本量 vs F" 曲线，找到收益递减的拐点
```

#### 4.4 最优传输权重

```
当前: 按 GMM component 比例分配权重
应该: 用 Optimal Transport 求解最优权重

min W2(P_target, Σ wi * δ(xi))
s.t. Σ wi = N_target, wi ≥ 0

使得加权样本的分布最接近真实分布
需要: pip install POT (Python Optimal Transport)
```

### 方向 5: 评估层 — 更丰富的实验能力

#### 5.1 推荐算法插拔

```
定义标准接口:
class RecSys:
    def recommend(self, user_profile, history, n=10) -> list[VideoItem]:
        ...

内置 3 个 baseline:
  - RandomRecSys: 随机推荐
  - PopularRecSys: 热门推荐（按历史播放量排序）
  - CollaborativeRecSys: 协同过滤

用户可以实现自己的算法插入做对比
```

#### 5.2 基础设施干预实验

```
支持的干预类型:
  - "把 30% 用户从 1080p 降到 720p"
  - "H.264 全量切到 H.265"
  - "CDN 预算砍 20%"

实现: 在 runner 里支持 per-agent 的 infra override
  infra_overrides = {
      "quality": "720p",           # 固定画质
      "stall_rate_multiplier": 1.5, # 卡顿率 ×1.5
  }
```

#### 5.3 多臂 A/B/N 测试

```
当前: 只支持 A/B 两组
应该: 支持 N 组，带 Bonferroni 修正

run_multitest(configs=[
    ("algo_v1", config_1),
    ("algo_v2", config_2),
    ("algo_v3_with_720p", config_3),
])
→ 两两比较 + 多重检验修正
```

#### 5.4 时序实验（多天留存）

```
当前: 单次会话模拟
应该: 模拟多天

Day 1: 推荐算法 A，首刷漏斗
Day 2: 留存回访？(基于 Day 1 体验)
Day 3: 切换到算法 B
Day 7: 7日留存率

关键: 需要跨会话状态（用户的兴趣漂移、习惯养成、流失风险）
```

#### 5.5 ROI 计算器

```
自动计算:
  input:
    infra_cost_per_user_per_day = ¥0.01 (1080p) vs ¥0.006 (720p)
    user_ltv = ¥50
    churn_rate_delta = +2.1%
  
  output:
    daily_cost_saving = (0.01 - 0.006) × 3.2亿 = ¥128万/天
    daily_user_loss = 3.2亿 × 2.1% = 672万 users
    daily_ltv_loss = 672万 × ¥50 = ¥3.36亿
    ROI = 128万 / 33600万 = 0.004 → 不值得降画质
```

### 方向 6: 工程层 — 更快更稳

#### 6.1 并行模拟

```
当前: 单进程顺序跑 1000 agents，~258s
应该: multiprocessing Pool

from multiprocessing import Pool

def simulate_agent(args):
    skeleton, videos, config = args
    return engine.run_session(skeleton, videos, config)

with Pool(10) as pool:
    results = pool.map(simulate_agent, agent_configs)

预期: 速度 ×5-8 (受 GIL 和 DeepSeek API 延迟限制)
```

#### 6.2 LLM 调用优化

```
batch 调用:
  当前: 每次 1 个 HTTP 请求
  应该: 收集 10-20 个决策，一次 batch 调用

缓存:
  key = (archetype_id, category_bucket, infra_bucket, session_type)
  相同 key 的决策复用之前的 LLM 结果 + 个体扰动
  → LLM 调用从 16% 降到 3-5%

Teacher-Student:
  L2 LLM 做 teacher
  每 N 轮收集 L2 的决策数据
  用这些数据重新拟合 L1 的参数
  → 迭代后 LLM 比例越来越低
```

#### 6.3 Dashboard 增强

```
当前: 单次报告展示
应该:
  - 历史对比: 加载多个 report JSON，叠加曲线
  - 校准过程: 每轮迭代的参数变化动画
  - A/B test: 控制组 vs 实验组的对比视图
  - 外推层: 10亿用户分层饼图 + 细分维度
  - 实时进度: WebSocket 推送模拟进度
```

#### 6.4 CI/CD

```
GitHub Actions:
  push → 跑 unit tests (85 tests)
  daily → 跑 full pipeline → 保存报告
  weekly → 跑校准环 → 追踪 F 趋势

报告存档:
  reports/YYYY-MM-DD_report.json
  → 自动生成 F 趋势曲线
```

### 方向 7: 研究方向 — 方法论突破

#### 7.1 LLM Agent 行为一致性

```
实验设计:
  同一个 persona，同一个场景，跑 10 次
  → 10 次决策一样吗？方差多大？

  同一个场景，换 LLM (DeepSeek vs GPT-4o vs Claude)
  → 不同模型的行为差异？谁更"像人"？

  temperature=0 vs 0.7 vs 1.0
  → 温度对行为多样性的影响？

论文价值: 高。这是 "LLM agent 模拟真实用户" 方法论的核心问题。
```

#### 7.2 真实用户验证闭环

```
如果有自有 app 数据:
  1. 用历史数据校准模拟系统
  2. 用模拟系统预测"如果切换算法 B，留存会怎样"
  3. 真正上线算法 B
  4. 对比预测 vs 实际
  
  如果预测误差 < 10% → 模拟系统有实际产品价值
  如果预测误差 > 30% → 模型需要改进

这个闭环是证明整个系统价值的唯一方式。
```

#### 7.3 运筹学布局验证

```
对比实验:
  方法 A: 随机采样 1000 个 persona
  方法 B: LHS 采样 1000 个
  方法 C: Support Points 1000 个
  方法 D: 列生成 1000 个

  固定其他条件，只换 persona 布局方法
  比较: F_multidim、外推稳定性、覆盖率

  如果 C/D 明显优于 A → 运筹学方法有价值
  如果差异不大 → 1000 个样本下优化布局收益有限

论文价值: 中。是对设计文档里"运筹学优化布局"假设的验证。
```

#### 7.4 涌现行为分析

```
Layer 2 LLM 的 "reason" 字段是宝藏:
  {"watch_pct": 0.85, "reason": "虽然画质差但内容太感兴趣了"}
  {"watch_pct": 0.10, "reason": "第一次用app就卡成这样，再见"}

分析:
  收集所有 L2 决策
  找出 LLM 和 L1 参数化模型分歧最大的场景
  → 这些场景就是参数化模型的盲区
  → 反过来指导 L1 模型设计

  例如发现: "首刷 + 高兴趣 + 卡顿" 场景下 LLM 比 L1 宽容
  → 说明 L1 的首刷惩罚过重
  → 调整首刷 tolerance 系数
```

---

## 三、建议优先级

### 短期（1-2天）— 立竿见影

| 序号 | 任务 | 预期效果 | 难度 |
|------|------|---------|------|
| 1 | 校准环加回滚 + per-archetype 调参 | F_multidim 不再下降 | 中 |
| 2 | videos_per_session=50 重跑 | 活跃度保真度更真实 | 低 |
| 3 | 兴趣漂移（每视频后更新兴趣向量） | 会话内行为更真实 | 低 |

### 中期（1周）— 能力扩展

| 序号 | 任务 | 预期效果 | 难度 |
|------|------|---------|------|
| 4 | Vine Copula 替代 GMM | 外推质量提升 | 中 |
| 5 | 推荐算法接口 + 3 个 baseline | 评估层真正可用 | 中 |
| 6 | 并行模拟 + LLM 缓存 | 性能 ×5 | 中 |
| 7 | KuaiRand 数据接入 | ground truth 更准 | 低 |

### 长期（研究方向）— 论文和产品价值

| 序号 | 任务 | 预期效果 | 难度 |
|------|------|---------|------|
| 8 | LLM 行为一致性实验 | 方法论论文 | 中 |
| 9 | 真实数据验证闭环 | 产品价值证明 | 高 |
| 10 | 运筹学布局对比实验 | 设计假设验证 | 中 |
| 11 | 时序多天留存模拟 | 留存分析能力 | 高 |
| 12 | ROI 计算器 | 商业决策支持 | 低 |

---

## 四、MVP 实测基线（对比基准）

后续所有优化都应该和这个基线对比：

```
MVP Pipeline (2026-05-07, DeepSeek, 100 agents × 10 videos):

F_overall:   0.468
F_multidim:  0.497
  watch_ratio_js:       0.457
  category_js:          0.835
  activity_wasserstein: 0.389
  correlation_distance: 0.805
  conditional_avg_delta:0.000

外推 10 亿:
  Heavy (WR>70%):   73.9%
  Medium (30-70%):  24.1%
  Light (WR<30%):    2.0%
  质量 JS=0.040

A/B (10 vs 30 videos):
  avg_watch_pct: -10.2%, p<0.0001, d=-0.45

成本: ~¥0.10 / 次完整 pipeline run
```
