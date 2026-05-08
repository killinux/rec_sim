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

### Latest Fidelity (R4: DeepSeek + 10 videos)
```
F_overall:   0.420
F_multidim:  0.562
  category_js:          0.835  ✅ stable
  correlation:          0.799  ✅ ok
  conditional_rank_dist:0.529  ✅ improved (was 0.000)
  watch_ratio_js:       0.404  ⚠️ needs bimodal distribution
  activity_wasserstein: 0.228  ⚠️ needs more videos/session
```

### Optimization History (R1-R4, see docs/optimization-log.md)
| Round | F_multidim | Key Change |
|-------|-----------|------------|
| Base  | 0.497 | - |
| R1    | 0.406 | Spearman rank (stricter metrics) |
| R2    | 0.543 | Fix thresholds |
| R3    | 0.532 | Interest differentiation |
| R4    | 0.562 | DeepSeek LLM (conditional improved) |

### Post-MVP Improvements (2026-05-08)
Done:
- ~~Increase videos_per_session~~ → 10→50 (R5, awaiting Mac result)
- ~~Interest drift~~ → decay watched categories, boost unseen
- ~~Per-archetype calibration~~ → replace global tuning
- ~~Snap decision mechanism~~ → bimodal WR distribution (hooked/skip)

Remaining:
- Vine Copula (replace GMM) for better joint distribution modeling
- Support Points + Column Generation for persona optimization
- LLM inner loop in calibration (audit Layer 1 decisions)
- Parallel simulation + LLM caching for performance
- Dashboard historical comparison (multiple reports over time)

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
├── runner.py              # simulation main loop (RealDataContext + LLMProvider)
├── report.py              # JSON report generator + multidim fidelity
├── dashboard.html         # H5 visualization (Chart.js, auto-loads JSON)
├── baseline/
│   ├── loader.py          # KuaiRec data loading
│   ├── clustering.py      # KMeans user clustering
│   ├── distribution.py    # Beta/LogNormal fitting per archetype
│   └── interest.py        # interest vectors + cosine matching
├── fidelity/
│   ├── metrics.py         # KL, JS, Wasserstein, composite F
│   └── multidim.py        # 5-dim fidelity
├── persona/
│   └── skeleton.py        # LHS-based persona generation
├── interaction/
│   ├── infra.py           # network/quality/stall state model
│   ├── context.py         # session context (time, fatigue)
│   ├── layer0.py          # experience decision
│   ├── layer1.py          # content decision
│   ├── layer2.py          # LLM emergent decision (DeepSeek/OpenAI/Mock)
│   └── engine.py          # L0+L1+L2 orchestration
├── llm/
│   └── provider.py        # LLM abstraction (Mock/DeepSeek/OpenAI/Ollama)
├── calibration/
│   └── loop.py            # calibration loop (outer + mid iterations)
├── extrapolation/
│   └── scaler.py          # GMM-based 1000→1B scaling
└── evaluation/
    └── abtest.py          # A/B test with statistical significance

scripts/
└── run_full_pipeline.py   # runs all 7 steps end-to-end
```

## Tech

- Python 3.9+ (Mac has 3.9.6, use `from __future__ import annotations`)
- pandas, numpy, scipy, scikit-learn, pytest
- LLM: DeepSeek API (OpenAI-compatible), key via `LLM_API_KEY` env var
- 85 tests, 16 test files, 25 Python modules

## Running the Full Pipeline

```bash
cd /Users/bytedance/Desktop/hehe/research/rec_sim
export PYTHONPATH="$PWD/src"
export LLM_API_KEY="your-deepseek-key"
python3 scripts/run_full_pipeline.py
```

Outputs 3 reports to `reports/`:
- latest_report.json (simulation + multidim fidelity)
- traffic_report_1B.json (1B user extrapolation)
- abtest_result.json (statistical A/B comparison)
