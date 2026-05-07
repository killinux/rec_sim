# 测试原理说明

## 单元测试（不依赖真实数据）

### test_distribution.py — 分布拟合

验证 `ArchetypeDistribution` 的数学正确性：
- 给一组 watch_ratio 数据，用 **Beta 分布拟合**（MLE），检查拟合后的 a、b 参数能否正确采样
- 给一组 duration 数据，用 **LogNormal 分布拟合**，检查 mu、sigma
- 序列化/反序列化一致性（to_dict → JSON → from_dict）

**为什么用 Beta 拟合 watch_ratio？** 完播率天然在 [0,1] 区间，Beta 分布是 [0,1] 上最灵活的参数分布，两个参数 a、b 能描述各种形状（U 型、J 型、钟型）。

### test_metrics.py — 保真度指标

验证四种距离度量和综合保真度系数：
- KL 散度：同分布 → 0，不同分布 → 正数
- JS 散度：对称性（JS(p,q) = JS(q,p)）、有界性 [0,1]
- Wasserstein 距离：平移分布的距离 = 平移量
- 综合保真度 F：满分 1.0，各维度加权计算

**原理：**
- KL 衡量"用 q 近似 p 需要多少额外信息量"，不对称
- JS 是 KL 的对称化版本，JS(p,q) = 0.5*KL(p,m) + 0.5*KL(q,m)，其中 m=(p+q)/2
- Wasserstein 是"搬土距离"——把一个分布的概率质量搬到另一个分布需要的最小代价

### test_skeleton.py — Persona 骨架生成

验证 LHS（Latin Hypercube Sampling）生成的 persona 骨架：
- 1000 个 agent 按 archetype 比例分配（70%/30% → 检查实际比例在 65%-75%）
- LHS 保证每个维度均匀覆盖
- 骨架参数有方差（不是所有人都一样）

**LHS 比随机采样好在哪？** 随机采样在高维空间会"扎堆"，LHS 强制在每个维度上均匀切分，用 N 个点就能保证每个维度被 N 等份覆盖。这是运筹学优化布局 Phase A 的实现。

### test_infra.py — 基础设施和上下文模型

验证基础设施状态采样和会话上下文生成：
- 4G 网络下画质分布合理（不会出现 4K 占主导）
- 疲劳度随 step_index 按指数曲线增长：`fatigue = 1 - exp(-0.05 * step)`

### test_layer0.py — Layer 0 体验决策

验证 Layer 0（体验决策层，整个系统最重要的一层）：
- 无卡顿 → watch_pct_factor >= 0.9（几乎不惩罚）
- 重度卡顿 → force_skip 或 factor < 0.5
- 首刷比普通更敏感（同样卡顿，首刷用户容忍度减半）
- 低画质降低 factor

**核心公式：** `experience_decision` 用三个惩罚项叠加：

```
stall_penalty = 1 - exp(-0.7 * 卡顿时长 / 容忍阈值)     ← 指数衰减曲线
quality_penalty = 画质敏感度 * (1 - 画质分) * 0.4
first_frame_penalty = min(超时比例 * 0.3, 0.5)
total_penalty = min(stall + quality + first_frame, 1.0)
watch_pct_factor = clip(1.0 - total_penalty + noise, 0.05, 1.0)
```

首刷时 `tolerance *= 0.5`，相当于把容忍阈值砍半，同样的卡顿在首刷场景下惩罚更重。

### test_layer1.py — Layer 1 内容决策

验证 Layer 1（内容决策层，对齐大盘分布）：
- 高兴趣匹配 → 高完播率
- Layer 0 的 factor 乘进去（画质差 → 即使内容好也看不完）
- 疲劳降低完播率

**核心公式：**

```
raw_pct = baseline + (interest_match - 0.5) * 0.4 - fatigue * 0.3 + noise
watch_pct = clip(raw_pct * l0_factor, 0.0, 1.0)
```

Layer 0 和 Layer 1 是**乘法关系**，不是加法——这保证了"体验损失 X%"可以从最终 watch_pct 中精确拆解出来。例如 l0_factor=0.7, l1_base=0.65 → 最终 0.455，可拆解为"内容本身值 0.65，画质拖了 30%"。

### test_engine.py — 决策引擎编排

验证 Layer 0 → Layer 1 的编排逻辑：
- L0 判定 force_exit → action="exit_app"，不再调 L1
- L0 判定 force_skip → action="skip"，不再调 L1
- 正常情况走完 L0+L1，输出带 `fidelity_tag: "parametric"`

**决策流：**

```
视频到达
    ↓
Layer 0: 体验能接受吗？
├── force_exit → "exit_app" (fidelity_tag="rule", decision_layer=0)
├── force_skip → "skip" (fidelity_tag="rule", decision_layer=0)
└── 通过 → 计算 watch_pct_factor，传给 Layer 1
    ↓
Layer 1: 内容感兴趣吗？
└── watch_pct = 内容分 * L0_factor → "watch"/"skip" (fidelity_tag="parametric", decision_layer=1)
```

### test_runner.py — 仿真循环

验证 N agents × M videos 的完整模拟循环：
- 日志包含所有必要字段（session_id, agent_id, action, fidelity_tag 等）
- 输出包含保真度 F
- exit_app 后该 agent 的会话提前结束（所以总步数 <= N*M）

---

## E2E 集成测试（依赖 KuaiRec 真实数据）

### test_e2e.py — 端到端闭环

加载真实的 `small_matrix.csv`（1411 用户 × 3327 视频 ≈ 470 万条交互），执行完整流程：

```
KuaiRec 真实数据
    → 提取用户特征（5维基础 + 10维品类占比）
    → StandardScaler 标准化
    → KMeans 聚成 20 个 archetype
    → 每个 archetype 拟合 Beta + LogNormal 分布
    → 按比例分配 100 个 persona 骨架（LHS）
    → 每个 persona 模拟看 10 个视频
    → 每个视频经过 Layer 0（体验）→ Layer 1（内容）
    → 汇总行为日志
    → 计算 F_overall = 1 - |真实avg完播率 - 模拟avg完播率| / 真实avg完播率
    → 打印保真度报告
```

**这个测试验证的核心问题：** 从真实数据提取的分布，经过骨架生成 → 参数化决策后，模拟出来的行为分布能多大程度对齐真实分布。F_overall 是第一个端到端的保真度信号。

---

## 各测试文件与设计文档模块的对应关系

| 测试文件 | 设计文档模块 | 验证内容 |
|----------|-------------|---------|
| test_distribution.py | 3. 基座层 | 分布拟合的数学正确性 |
| test_metrics.py | 4. 保真度度量体系 | 距离度量和综合 F 的计算 |
| test_skeleton.py | 5. Persona 生成层 | LHS 布局和比例分配 |
| test_infra.py | 6.1 交互层三维输入 | 基础设施和上下文采样 |
| test_layer0.py | 6.3 Layer 0 体验决策层 | 卡顿/画质/首帧的响应函数 |
| test_layer1.py | 6.4 Layer 1 内容决策层 | 兴趣匹配和大盘对齐 |
| test_engine.py | 6.6 三层联动 | L0→L1 编排和拆解能力 |
| test_runner.py | 完整仿真循环 | N×M 循环和日志格式 |
| test_e2e.py | 全链路 | 真实数据端到端验证 |
