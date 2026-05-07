# RecSim Project Context — Read This First

## What Is This

大规模用户模拟系统，用 1000 个 AI agent 模拟短视频推荐场景下的用户行为，通过统计外推覆盖 10 亿用户。

## Repo

- GitHub: `git@github.com:killinux/rec_sim.git`
- Windows 代码目录: `C:\Users\Administrator\Desktop\macwork\rec_sim\`
- Mac 运行目录: `/Users/bytedance/Desktop/hehe/research/rec_sim/`
- Mac 数据目录: `/Users/bytedance/Desktop/hehe/datasets/` (KuaiRec + MovieLens, 不在 git 里)

## Mac Remote Control

Mac 通过 HTTP 轮询远程控制，server 在 Windows 本机 `localhost:8900`:
- 发命令: `POST http://localhost:8900/cmd` body `{"cmd": "..."}`
- 取结果: `GET http://localhost:8900/results` 返回 JSON 数组
- 单槽: 一次只能有一条待执行命令，Mac 每 3 秒轮询拿走
- server 代码在 `C:\Users\Administrator\Desktop\macwork\mac-r\server.py`

Mac 上跑测试:
```bash
cd /Users/bytedance/Desktop/hehe/research/rec_sim
git pull origin main
export PYTHONPATH="$PWD/src"
python3 -m pytest tests/ -v -s
```

## Current Progress (2026-05-07)

### Completed
- Phase 1: 12 tasks, end-to-end simulation loop (loader → clustering → distribution → persona → L0+L1 decision → runner → report)
- P0: Real interest matching (cosine similarity from KuaiRec item_categories) + multi-dimensional fidelity (5 metrics)
- P1: Connected real KuaiRec category distributions, activity, correlation data
- 53/53 tests passing on Mac (132s)
- H5 Dashboard at `http://localhost/research/rec_sim/reports/dashboard.html` (Mac)
- All docs written and pushed

### Latest Fidelity (P1 E2E)
```
F_overall:   0.440
F_multidim:  0.406
  category_js:          0.835  ✅ fixed
  correlation:          0.811  ✅ ok
  watch_ratio_js:       0.385  ⚠️ needs improvement
  activity_wasserstein: 0.000  ❌ config issue (10 vs 3327 videos/user)
  conditional_delta:    0.000  ❌ 28% avg deviation per category
```

### Next Steps (priority order)
1. Fix activity normalization (videos_per_session=10 vs KuaiRec's 3327 — normalize or increase)
2. Calibration loop (3 nested loops: outer=persona rebuild, mid=param tune, inner=LLM audit)
3. Layer 2 LLM integration (Claude API for complex decisions: first-visit, new-category, conflicts)
4. Vine Copula extrapolation (1000 → 1 billion)

## Key Documentation

- `docs/system-pipeline.md` — 8-step pipeline overview with principles
- `docs/2026-05-06-user-simulation-system-design.md` — full design spec
- `docs/progress-log.md` — detailed progress with metrics history
- `docs/test-principles.md` — test principles and math foundations
- `docs/superpowers/plans/2026-05-07-rec-sim-phase1.md` — Phase 1 implementation plan

## Code Structure

```
src/rec_sim/
├── config.py              # paths, constants
├── runner.py              # simulation main loop (accepts RealDataContext)
├── report.py              # JSON report generator + multidim fidelity
├── dashboard.html         # H5 visualization (Chart.js)
├── baseline/
│   ├── loader.py          # KuaiRec data loading
│   ├── clustering.py      # KMeans user clustering
│   ├── distribution.py    # Beta/LogNormal fitting per archetype
│   └── interest.py        # interest vectors + cosine matching
├── fidelity/
│   ├── metrics.py         # KL, JS, Wasserstein, composite F
│   └── multidim.py        # 5-dim fidelity (category, conditional, activity, correlation)
├── persona/
│   └── skeleton.py        # LHS-based persona generation
└── interaction/
    ├── infra.py           # network/quality/stall state model
    ├── context.py         # session context (time, fatigue)
    ├── layer0.py          # experience decision (stall/quality → skip/exit)
    ├── layer1.py          # content decision (interest match → watch_pct)
    └── engine.py          # L0+L1 orchestration
```

## Tech

- Python 3.9+ (Mac has 3.9.6, use `from __future__ import annotations`)
- pandas, numpy, scipy, scikit-learn, pytest
- No LLM dependencies yet (Phase 1 is pure math)
