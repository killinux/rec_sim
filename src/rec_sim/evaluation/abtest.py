"""A/B test framework for comparing simulation variants."""
from __future__ import annotations
import copy
import numpy as np
from dataclasses import dataclass, field
from scipy import stats as sp_stats
from rec_sim.runner import run_simulation, SimulationConfig, RealDataContext, SimulationResult
from rec_sim.baseline.distribution import ArchetypeDistribution
from rec_sim.report import generate_report
from rec_sim.llm.provider import LLMProvider


@dataclass
class ABTestConfig:
    """Configuration for an A/B test."""
    name: str
    control_label: str = "control"
    treatment_label: str = "treatment"
    # Treatment modifies these in RealDataContext or SimulationConfig
    treatment_overrides: dict = field(default_factory=dict)
    significance_level: float = 0.05


@dataclass
class ABTestResult:
    """Results of an A/B test comparison."""
    name: str
    control_label: str
    treatment_label: str
    control_metrics: dict
    treatment_metrics: dict
    deltas: dict
    statistical_tests: dict
    winner: str
    summary: str


def run_abtest(
    ab_config: ABTestConfig,
    distributions: list[ArchetypeDistribution],
    sim_config: SimulationConfig,
    real_data: RealDataContext | None = None,
    llm_provider: LLMProvider | None = None,
    report_dir: str | None = None,
) -> ABTestResult:
    """Run an A/B test comparing control vs treatment.

    The treatment_overrides dict can modify:
    - sim_config fields: n_agents, videos_per_session, seed
    - infra defaults (applied via modified seed to get different infra samples)
    """
    # Run control
    control_result = run_simulation(sim_config, distributions, real_data, llm_provider)
    control_report = generate_report(
        control_result, distributions, sim_config, real_data=real_data,
    )

    # Build treatment config
    treatment_sim_config = copy.deepcopy(sim_config)
    treatment_sim_config.seed = sim_config.seed + 10000  # different randomness

    for key, val in ab_config.treatment_overrides.items():
        if hasattr(treatment_sim_config, key):
            setattr(treatment_sim_config, key, val)

    # Run treatment
    treatment_result = run_simulation(treatment_sim_config, distributions, real_data, llm_provider)
    treatment_report = generate_report(
        treatment_result, distributions, treatment_sim_config, real_data=real_data,
    )

    # Extract key metrics
    control_metrics = _extract_metrics(control_result, control_report)
    treatment_metrics = _extract_metrics(treatment_result, treatment_report)

    # Compute deltas
    deltas = {}
    for key in control_metrics:
        if key in treatment_metrics:
            c_val = control_metrics[key]
            t_val = treatment_metrics[key]
            deltas[key] = {
                "control": c_val,
                "treatment": t_val,
                "absolute_delta": t_val - c_val,
                "relative_delta": (t_val - c_val) / max(abs(c_val), 1e-10),
            }

    # Statistical tests
    stat_tests = _run_statistical_tests(
        control_result, treatment_result, ab_config.significance_level
    )

    # Determine winner
    winner = _determine_winner(deltas, stat_tests, ab_config)

    # Summary text
    summary_lines = [
        f"A/B Test: {ab_config.name}",
        f"Control ({ab_config.control_label}) vs Treatment ({ab_config.treatment_label})",
        "",
    ]
    for key, d in deltas.items():
        direction = "+" if d["absolute_delta"] > 0 else ""
        summary_lines.append(
            f"  {key}: {d['control']:.4f} -> {d['treatment']:.4f} "
            f"({direction}{d['relative_delta']:.1%})"
        )
    summary_lines.append(f"\nWinner: {winner}")

    return ABTestResult(
        name=ab_config.name,
        control_label=ab_config.control_label,
        treatment_label=ab_config.treatment_label,
        control_metrics=control_metrics,
        treatment_metrics=treatment_metrics,
        deltas=deltas,
        statistical_tests=stat_tests,
        winner=winner,
        summary="\n".join(summary_lines),
    )


def _extract_metrics(result: SimulationResult, report: dict) -> dict:
    """Extract comparable metrics from simulation result."""
    logs = result.logs
    watch_logs = [l for l in logs if l["action"] == "watch"]
    watch_pcts = [l["watch_pct"] for l in watch_logs]

    return {
        "avg_watch_pct": float(np.mean(watch_pcts)) if watch_pcts else 0,
        "median_watch_pct": float(np.median(watch_pcts)) if watch_pcts else 0,
        "watch_rate": len(watch_logs) / max(len(logs), 1),
        "skip_rate": sum(1 for l in logs if l["action"] == "skip") / max(len(logs), 1),
        "exit_rate": sum(1 for l in logs if l["action"] == "exit_app") / max(len(logs), 1),
        "like_rate": sum(1 for l in logs if l.get("liked")) / max(len(logs), 1),
        "f_multidim": report.get("fidelity_multidim", {}).get("F_multidim", 0),
    }


def _run_statistical_tests(
    control: SimulationResult,
    treatment: SimulationResult,
    alpha: float,
) -> dict:
    """Run statistical significance tests on key metrics."""
    c_wp = [l["watch_pct"] for l in control.logs if l["action"] == "watch"]
    t_wp = [l["watch_pct"] for l in treatment.logs if l["action"] == "watch"]

    results = {}

    if len(c_wp) > 1 and len(t_wp) > 1:
        # Mann-Whitney U test (non-parametric, no normality assumption)
        stat, p_value = sp_stats.mannwhitneyu(c_wp, t_wp, alternative="two-sided")
        results["watch_pct_mannwhitney"] = {
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": p_value < alpha,
        }

        # Welch's t-test
        stat, p_value = sp_stats.ttest_ind(c_wp, t_wp, equal_var=False)
        results["watch_pct_welch_t"] = {
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": p_value < alpha,
        }

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(c_wp)**2 + np.std(t_wp)**2) / 2)
        if pooled_std > 0:
            cohens_d = (np.mean(t_wp) - np.mean(c_wp)) / pooled_std
        else:
            cohens_d = 0.0
        results["effect_size_cohens_d"] = float(cohens_d)

    return results


def _determine_winner(deltas: dict, stat_tests: dict, config: ABTestConfig) -> str:
    """Determine which variant won based on primary metric and significance."""
    if "avg_watch_pct" not in deltas:
        return "inconclusive"

    d = deltas["avg_watch_pct"]
    sig_test = stat_tests.get("watch_pct_mannwhitney", {})
    is_significant = sig_test.get("significant", False)

    if not is_significant:
        return "no_significant_difference"

    if d["absolute_delta"] > 0:
        return config.treatment_label
    elif d["absolute_delta"] < 0:
        return config.control_label
    else:
        return "tie"
