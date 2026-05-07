# RecSim 项目进展记录

> 更新时间: 2026-05-07

---

## 项目概述

大规模用户模拟系统，用 1000 个 AI agent 模拟短视频推荐场景下的用户行为，通过统计外推覆盖 10 亿用户规模。

**仓库:** [github.com/killinux/rec_sim](https://github.com/killinux/rec_sim)

**开发环境:**
- 代码编写: Windows `C:\Users\Administrator\Desktop\macwork\rec_sim\`
- 运行测试: Mac `/Users/bytedance/Desktop/hehe/research/rec_sim/`
- Mac 远程控制: HTTP 轮询 `localhost:8900`

**数据资产 (Mac):**
| 数据集 | 状态 | 路径 | 大小 |
|--------|------|------|------|
| KuaiRec 2.0 | 已下载 (Zenodo) | `/Users/bytedance/Desktop/hehe/datasets/KuaiRec/data/` | ~1.5GB |
| MovieLens-25M | 已下载 | `/Users/bytedance/Desktop/hehe/datasets/ml-25m/` | ~250MB |
| MIND-small | 下载失败 (Azure 链接不通) | - | 待重试 |

---

## Phase 1: 端到端最小闭环 — 已完成

### 目标
从 KuaiRec 真实数据提取分布 → 生成 persona → 参数化模拟 → 输出保真度报告

### 完成的 Tasks

| Task | 内容 | 文件 | Commit |
|------|------|------|--------|
| 0 | 项目脚手架 + git init | pyproject.toml, config.py, sync.sh | `173313e` |
| 1 | KuaiRec 数据加载器 | baseline/loader.py | `f20246d` |
| 2 | 用户聚类 (KMeans) | baseline/clustering.py | `f4f8a6d` |
| 3 | Archetype 分布提取 (Beta + LogNormal) | baseline/distribution.py | `fa2af79` |
| 4 | 保真度指标 (KL/JS/Wasserstein/F) | fidelity/metrics.py | `6ad7cd5` |
| 5 | Persona 骨架 (LHS) | persona/skeleton.py | `a4898e3` |
| 6 | 基础设施 + 上下文模型 | interaction/infra.py, context.py | `24a0da7` |
| 7 | Layer 0 体验决策 | interaction/layer0.py | `6b4e136` |
| 8 | Layer 1 内容决策 | interaction/layer1.py | `9128a29` |
| 9 | 决策引擎 (L0+L1 编排) | interaction/engine.py | `dd2df14` |
| 10 | 仿真 Runner | runner.py | `3e931c9` |
| 11 | E2E 集成测试 | tests/test_e2e.py | `051988f` |

### 修复记录
| 问题 | 原因 | 修复 |
|------|------|------|
| Python 3.9 类型注解报错 | `dict[str, float] \| None` 需要 3.10+ | 加 `from __future__ import annotations` |
| test_runner 断言失败 | exit_app 导致总步数 < N*M | 改为 `assert len(logs) <= N*M` |
| KuaiRec 数据找不到 | Zenodo 解压多了一层 `KuaiRec 2.0/data/` | 拷贝 CSV 到 `data/` 根目录 |

### Phase 1 测试结果
```
41 passed, 0 failed, 16.66s
```

### Phase 1 E2E 保真度 (单维)
```
Agents: 100 | Videos/session: 10 | Archetypes: 20

F_overall:  0.983
Target WR:  0.692 (KuaiRec 真实平均完播率)
Actual WR:  0.680 (模拟平均完播率)
Exit Rate:  0.4%
Skip Rate:  14.2%
L0 Factor:  0.872
```

---

## P0: 真实兴趣匹配 + 多维保真度 — 已完成

### P0a: 真实兴趣匹配

**问题:** interest_match 是随机 Beta(2,2)，跟用户/视频无关

**解法:** 用 KuaiRec item_categories 的品类标签构建兴趣向量

```
1. 每个视频有品类标签: video_0 → [8], video_1 → [27, 9]
2. 从用户的观看历史 + watch_ratio 加权，构建用户兴趣向量
   user_1 = {品类8: 0.75, 品类9: 0.125, 品类27: 0.125}
3. 视频也有品类向量 (one-hot 归一化)
4. interest_match = cosine(用户向量, 视频向量)
```

**文件:** `src/rec_sim/baseline/interest.py`
**函数:**
- `build_category_map(items)` → `{video_id: [cat_ids]}`
- `build_user_interest_vectors(interactions, cat_map, all_cats)` → `{user_id: np.array}`
- `build_archetype_interest_vectors(user_vectors, user_ids, labels)` → `{arch_id: np.array}`
- `compute_interest_match(user_vec, item_vec)` → cosine similarity

### P0b: 多维保真度

**问题:** F 只看平均完播率的相对误差，单一维度无法反映分布结构

**解法:** 5 个维度同时衡量

| 维度 | 指标 | 含义 | max_acceptable |
|------|------|------|---------------|
| 完播率分布 | JS 散度 | 分布形状差异 | 0.3 |
| 品类消费分布 | JS 散度 | 品类偏好差异 | 0.3 |
| 条件分布 P(WR\|品类) | 平均偏差 | 各品类完播率差异 | 0.2 |
| 活跃度分布 | Wasserstein | 活跃度分布搬土距离 | 50.0 |
| 相关性矩阵 | Frobenius 范数 | 维度间相关性差异 | 2.0 |

**综合打分:**
```
score_i = 1 - min(raw_i / max_acceptable_i, 1)
F_multidim = 加权平均(所有 score_i)
```

**文件:** `src/rec_sim/fidelity/multidim.py`

### P0c: 报告 + Dashboard 增强

**报告:** `src/rec_sim/report.py`
- 新增 `fidelity_multidim` 字段，包含各维度分数和原始指标
- 输出到 `reports/latest_report.json`

**Dashboard:** `src/rec_sim/dashboard.html`
- 新增 "Multi-dim F" KPI 卡片
- 新增 "Multi-dimensional Fidelity" 详情卡片，显示各维度得分和进度条
- 页面加载时自动 fetch 同目录的 `latest_report.json`
- 访问地址: `http://localhost/research/rec_sim/reports/dashboard.html`

### P0 测试结果
```
53 passed, 0 failed, 20.56s (新增 12 个测试)
```

### P0 E2E 保真度 (多维)
```
=== 单维 ===
F_overall:       0.602

=== 多维 ===
F_multidim:      0.669

各维度拆解:
  watch_ratio_js:       0.678 (分数)    raw=0.097    完播率分布形状差异适中
  category_js:          0.000 (分数)    raw=0.693    品类分布完全对不上 ← 最大问题
  activity_wasserstein: 1.000 (分数)    raw=0.0      暂未接入真实数据
  correlation_distance: 1.000 (分数)    raw=0.0      暂未接入真实数据
```

### P0 关键发现

1. **F 从 0.983 降到 0.602** — Phase 1 的 0.983 是假象，只看了均值。多维度量暴露了真实保真度
2. **品类分布 JS=0.693 (≈ln2)** — 模拟品类和真实品类几乎完全不同。原因: runner 里品类是 `rng.choice(categories)` 随机选的，没用真实品类分布
3. **完播率分布 JS=0.097** — 形状有差距但可接受
4. **活跃度和相关性暂为占位值** — 需要在 runner 里接入真实数据

---

## 代码结构

```
rec_sim/
├── docs/
│   ├── 2026-05-06-user-simulation-system-design.md   # 设计文档
│   ├── test-principles.md                             # 测试原理说明
│   └── progress-log.md                                # 本文档
├── reports/
│   ├── latest_report.json                             # 最新仿真报告
│   └── dashboard.html                                 # H5 可视化面板
├── src/rec_sim/
│   ├── config.py                                      # 路径和常量
│   ├── runner.py                                      # 仿真主循环 (accepts RealDataContext)
│   ├── report.py                                      # 报告生成器 + 多维保真度
│   ├── dashboard.html                                 # H5 可视化面板
│   ├── baseline/
│   │   ├── loader.py                                  # KuaiRec 数据加载
│   │   ├── clustering.py                              # 用户聚类
│   │   ├── distribution.py                            # 分布提取 (Beta/LogNormal)
│   │   └── interest.py                                # 兴趣向量 + 余弦匹配
│   ├── fidelity/
│   │   ├── metrics.py                                 # 基础指标 (KL/JS/Wass/F)
│   │   └── multidim.py                                # 多维保真度 (5维)
│   ├── persona/
│   │   └── skeleton.py                                # LHS 骨架生成
│   ├── interaction/
│   │   ├── infra.py                                   # 基础设施状态模型
│   │   ├── context.py                                 # 会话上下文模型
│   │   ├── layer0.py                                  # Layer 0 体验决策
│   │   ├── layer1.py                                  # Layer 1 内容决策
│   │   ├── layer2.py                                  # Layer 2 LLM 涌现决策
│   │   └── engine.py                                  # 决策引擎 (L0+L1+L2 编排)
│   ├── llm/
│   │   └── provider.py                                # LLM 抽象接口 (Mock/DeepSeek/OpenAI/Ollama)
│   └── calibration/
│       └── loop.py                                    # 校准环 (外循环+中循环)
├── tests/
│   ├── test_loader.py          (3 tests)
│   ├── test_clustering.py      (3 tests)
│   ├── test_distribution.py    (3 tests)
│   ├── test_metrics.py         (10 tests)
│   ├── test_skeleton.py        (4 tests)
│   ├── test_infra.py           (4 tests)
│   ├── test_layer0.py          (4 tests)
│   ├── test_layer1.py          (4 tests)
│   ├── test_layer2.py          (12 tests)
│   ├── test_engine.py          (3 tests)
│   ├── test_runner.py          (3 tests)
│   ├── test_interest.py        (6 tests)
│   ├── test_multidim.py        (6 tests)
│   ├── test_provider.py        (5 tests)
│   ├── test_calibration.py     (4 tests)
│   └── test_e2e.py             (1 test, 需要 KuaiRec 真实数据)
└── pyproject.toml
```

**总计:** 22 个 Python 模块 + 16 个测试文件 (73 test cases)

---

## 校准环 — 已完成

### 原理

模拟跑完后，保真度 F 不达标（比如各品类完播率偏差 28%）。偏差不是代码 bug，而是参数经过 LHS 采样 + L0 打折 + L1 噪声后偏移了。校准环自动找偏差 → 调参 → 重跑 → 检查，迭代收敛。

### 两层循环

- **外循环**（最多 3 轮）：检查整体 F_multidim，偏差太大则标记需重建 persona
- **中循环**（最多 10 轮）：找最差维度 → 调 Beta 分布参数 → 重跑模拟 → 检查改善
- 内循环（LLM 抽检）：留给 Layer 2 实现

### 调参方式

```
目标: 品类8的完播率偏低 0.17
→ 对相关 archetype 的 Beta(a, b) 做调整:
  Beta 均值 = a/(a+b)
  new_mean = old_mean + lr * delta
  正则化: max_shift = old_mean / regularization_factor
  防止单次调太远导致震荡
```

### 测试

4 个测试全通过：基本运行、多轮迭代稳定性、历史记录结构、收敛条件。

---

## Layer 2 LLM 集成 — 已完成

### 架构

抽象 provider 接口，支持多种 LLM 后端：

```
LLMProvider (抽象基类)
├── MockProvider         — 测试用，关键词匹配，不调 API
├── OpenAICompatibleProvider — 统一接口
│   ├── DeepSeek        — base_url=api.deepseek.com (已测通)
│   ├── OpenAI          — base_url=api.openai.com
│   └── Ollama          — base_url=localhost:11434 (本地)
```

通过工厂函数切换：`create_provider("deepseek", api_key=...)` 或 `create_provider("mock")`

API key 通过环境变量 `LLM_API_KEY` 传入，不硬编码。

### Layer 2 触发条件

不是每个决策都走 LLM（太贵），只有 ~20% 的复杂场景触发：

| 条件 | 触发原因 |
|------|---------|
| 首刷前 N 个视频 | 留存关键路径 |
| L0 factor < 0.5 | 严重基础设施降级 |
| 兴趣 > 0.7 且 L0 < 0.7 | 内容好但体验差的冲突 |
| 随机 10% | 校准抽检 |

### Prompt 结构

```
USER PROFILE: 完播倾向、卡顿容忍、画质敏感度
CURRENT SITUATION: 首刷/普通、第几个视频、时段、网络、疲劳
VIDEO: 品类、兴趣匹配度、画质、首帧延迟、卡顿
→ 输出: {"watch_pct", "liked", "commented", "shared", "reason"}
```

### 成本估算

| 场景 | LLM 调用 | DeepSeek 费用 |
|------|---------|--------------|
| 单次 E2E (100 agents × 10 videos × 20%) | ~200 次 | ~¥0.1 |
| 校准环 (5 轮 × 200) | ~1000 次 | ~¥0.5 |
| 正式跑 (1000 × 30 × 20%) | ~6000 次 | ~¥3 |

### JSON 解析容错

LLM 返回可能不规范（markdown code block、不完整 JSON 等），provider 内置了多层 fallback：
1. 尝试直接 json.loads
2. 从 markdown ``` 块中提取
3. 失败返回默认值 + parse_error 标记

### 测试

16 个测试全部通过：provider 行为、触发条件、prompt 内容、Mock 决策、JSON 解析容错。

---

## 各模块原理速查

### 基座层 (baseline/)

**loader.py** — 加载 KuaiRec CSV，统一为 (user_id, item_id, watch_ratio, duration_ms, video_duration_ms) schema

**clustering.py** — 提取用户特征 (5 维基础 + 10 维品类占比) → StandardScaler → KMeans 聚成 N 个 archetype

**distribution.py** — 每个 archetype 拟合参数化分布:
- 完播率 → Beta(a, b)，因为 watch_ratio 在 [0,1] 区间，Beta 是最灵活的
- 观看时长 → LogNormal(mu, sigma)，因为时长是正偏态分布

**interest.py** — 从真实观看数据建兴趣向量:
- 用 watch_ratio 加权每个品类的曝光量
- 归一化为概率分布向量
- 匹配时用余弦相似度（归一化长度，只看偏好方向）

### 保真度 (fidelity/)

**metrics.py** — 四种距离度量:
- KL: 信息论度量，不对称，"用 q 近似 p 需要多少额外 bit"
- JS: KL 的对称化，有界 [0, ln2]
- Wasserstein: 搬土距离，"把分布 A 搬成分布 B 的最小代价"
- Composite F: 加权归一化综合分

**multidim.py** — 五维度综合保真度:
- 完播率分布形状 (JS)
- 品类消费分布 (JS)
- 条件分布 P(完播|品类) (均值偏差)
- 活跃度分布 (Wasserstein)
- 相关性矩阵 (Frobenius 范数)

### Persona (persona/)

**skeleton.py** — 两阶段生成:
1. 按 archetype 比例分配名额 (最大余额法)
2. Latin Hypercube Sampling 生成参数，保证各维度均匀覆盖
3. 从 archetype 的 Beta/LogNormal 分布采样个体参数

### 交互层 (interaction/)

**infra.py** — 网络环境决定画质分布和卡顿率:
- 2G: 60% 360p, 30% 卡顿率, 3s 首帧
- WiFi: 60% 1080p, 2% 卡顿率, 0.4s 首帧

**context.py** — 会话上下文:
- 时段分布: 晚间 35% 占主导
- 疲劳曲线: `fatigue = 1 - exp(-0.05 * step_index)`

**layer0.py** — 体验决策 (最重要的一层):
```
stall_penalty = 1 - exp(-0.7 * 卡顿时长 / 容忍阈值)
quality_penalty = 画质敏感度 * (1 - 画质分) * 0.4
first_frame_penalty = min(超时比例 * 0.3, 0.5)
watch_pct_factor = clip(1 - total_penalty + noise, 0.05, 1.0)
```
首刷时容忍阈值减半。

**layer1.py** — 内容决策:
```
watch_pct = (baseline + 兴趣加成 - 疲劳惩罚 + noise) * L0_factor
```
L0 和 L1 是乘法关系，可拆解"体验损失 X%"。

**engine.py** — 编排逻辑:
```
L0 force_exit → "exit_app" (不调 L1)
L0 force_skip → "skip" (不调 L1)
L0 通过 → L1 计算 → "watch"/"skip"
```

---

## 保真度演进记录

| 阶段 | F_overall | F_multidim | 关键变化 |
|------|-----------|-----------|---------|
| Phase 1 | 0.983 | - | 只看均值，虚高 |
| P0 多维 | 0.602 | 0.669 | 加了多维度量，品类 JS=0.693 暴露 |
| P1 品类修复 | 0.440 | 0.406 | 品类 JS 0.693→0.050，但活跃度暴露 |
| P1 归一化 | 0.440 | 0.489 | 活跃度 0→0.415，category 0.835 |

### Known Issues (MVP 后回来改)
- conditional_delta = 0.278 (各品类完播率偏差 28%) → 校准环可修
- watch_ratio_js = 0.184 (完播率分布形状) → 校准环可修
- videos_per_session=10 → 可增大到 100+ 重测

## 下一步计划

| 状态 | 任务 | 说明 |
|------|------|------|
| ✅ | 品类分布修复 | runner 用真实品类分布 |
| ✅ | 活跃度归一化 | 比较 per-user avg_wr 而非原始交互数 |
| ✅ | 校准环 | 外循环+中循环，自动调 Beta 参数 |
| ✅ | Layer 2 LLM 集成 | Provider 抽象 + DeepSeek/Mock |
| ⬜ | Vine Copula 外推 | 1000 → 10 亿统计放大 |
| ⬜ | 评估层 | A/B test 框架，算法对比 |

---

## 运行指南

### 在 Mac 上跑全套测试
```bash
cd /Users/bytedance/Desktop/hehe/research/rec_sim
git pull origin main
export PYTHONPATH="$PWD/src"
python3 -m pytest tests/ -v -s
```

### 生成报告 + 查看 Dashboard
```bash
# 跑 E2E 测试会自动生成 reports/latest_report.json
python3 -m pytest tests/test_e2e.py -v -s

# 复制 dashboard 到 reports 目录
cp src/rec_sim/dashboard.html reports/

# 浏览器访问
http://localhost/research/rec_sim/reports/dashboard.html
```

### 推送代码
```bash
# Windows 端
cd C:\Users\Administrator\Desktop\macwork\rec_sim
git add -A && git commit -m "..." && git push origin main

# Mac 端拉取
cd /Users/bytedance/Desktop/hehe/research/rec_sim && git pull origin main
```
