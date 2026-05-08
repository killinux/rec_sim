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

### Latest Fidelity (R8: DeepSeek + 50 videos, post-calibration)
```
F_multidim:  0.630   ← best so far (was 0.497 baseline)
  category_js:          0.840  ✅ stable
  correlation:          0.720  ✅ ok
  conditional_rank_dist:0.621  ✅ good
  activity_wasserstein: 0.574  ✅ fixed (was 0.000 in R5)
  watch_ratio_js:       0.192  ⚠️ main bottleneck — distribution shape
```

### Optimization History (R1-R8, see docs/optimization-log.md)
| Round | F_multidim | Key Change |
|-------|-----------|------------|
| Base  | 0.497 | Initial 5-dim metrics |
| R4    | 0.562 | DeepSeek LLM |
| R5    | 0.467 | 50 videos (exposed activity bug) |
| R7    | 0.603 | Activity mean-center fix, calibration converges |
| R8    | **0.630** | Snap-skip tuned, calibration converges in 2 iterations |

### Post-MVP Improvements (2026-05-08)
Done:
- ~~videos_per_session 10→50~~ → reduces per-user stat noise
- ~~Interest drift~~ → decay watched categories, boost unseen
- ~~Per-archetype calibration~~ → replace global tuning (calibration now converges!)
- ~~Snap decision mechanism~~ → bimodal WR distribution (tuned)
- ~~Activity mean-center~~ → robust to session length differences

Next (to reach F=0.8):
- Fix watch_ratio_js (0.192) — KDE-based WR sampling or higher LLM rate
- Vine Copula (replace GMM) for better joint distribution modeling
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
