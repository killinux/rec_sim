"""Calibration loop: iteratively adjust persona parameters to align distributions."""
from __future__ import annotations
import copy
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from rec_sim.baseline.distribution import ArchetypeDistribution
from rec_sim.runner import run_simulation, SimulationConfig, RealDataContext, SimulationResult
from rec_sim.report import generate_report


@dataclass
class CalibrationConfig:
    max_outer_iterations: int = 3
    max_mid_iterations: int = 10
    f_multidim_target: float = 0.7
    per_dim_targets: dict = field(default_factory=lambda: {
        "watch_ratio_js": 0.5,
        "category_js": 0.7,
        "activity_wasserstein": 0.5,
        "correlation_distance": 0.5,
        "conditional_avg_delta": 0.5,
    })
    learning_rate: float = 0.1
    regularization: float = 2.0


@dataclass
class CalibrationResult:
    converged: bool
    iterations_used: int
    initial_f: float
    final_f: float
    history: list[dict] = field(default_factory=list)
    final_distributions: list[ArchetypeDistribution] = field(default_factory=list)
    final_report: dict = field(default_factory=dict)


def calibrate(
    distributions: list[ArchetypeDistribution],
    sim_config: SimulationConfig,
    real_data: RealDataContext,
    cal_config: CalibrationConfig | None = None,
    report_dir: str | Path | None = None,
) -> CalibrationResult:
    """Run the calibration loop.

    Outer loop: check if F_multidim meets target
    Mid loop: adjust per-archetype parameters based on per-dimension gaps

    Returns CalibrationResult with convergence status and history.
    """
    if cal_config is None:
        cal_config = CalibrationConfig()

    current_dists = [copy.deepcopy(d) for d in distributions]
    history = []
    initial_f = 0.0
    total_iterations = 0

    for outer in range(cal_config.max_outer_iterations):
        for mid in range(cal_config.max_mid_iterations):
            total_iterations += 1

            # Run simulation with current distributions
            result = run_simulation(sim_config, current_dists, real_data=real_data)

            # Generate report to get fidelity metrics
            report_path = None
            if report_dir:
                report_path = Path(report_dir) / f"calibration_iter_{total_iterations}.json"
            report = generate_report(
                result, current_dists, sim_config,
                output_path=report_path, real_data=real_data,
            )

            fm = report.get("fidelity_multidim", {})
            f_multidim = fm.get("F_multidim", 0)
            per_dim = fm.get("per_dimension", {})
            raw_metrics = fm.get("raw_metrics", {})

            if total_iterations == 1:
                initial_f = f_multidim

            step_record = {
                "iteration": total_iterations,
                "outer": outer,
                "mid": mid,
                "F_multidim": f_multidim,
                "per_dimension": dict(per_dim),
                "raw_metrics": dict(raw_metrics),
                "adjustments": [],
            }

            # Check convergence
            if f_multidim >= cal_config.f_multidim_target:
                step_record["status"] = "converged"
                history.append(step_record)
                # Save final report
                if report_dir:
                    final_path = Path(report_dir) / "latest_report.json"
                    generate_report(result, current_dists, sim_config,
                                    output_path=final_path, real_data=real_data)
                return CalibrationResult(
                    converged=True, iterations_used=total_iterations,
                    initial_f=initial_f, final_f=f_multidim,
                    history=history, final_distributions=current_dists,
                    final_report=report,
                )

            # Find worst dimensions and adjust
            adjustments = _compute_adjustments(
                current_dists, per_dim, raw_metrics, report,
                cal_config, real_data,
            )
            step_record["adjustments"] = adjustments
            step_record["status"] = "adjusting"
            history.append(step_record)

            # Apply adjustments
            if not adjustments:
                break  # nothing to adjust, exit mid loop

            current_dists = _apply_adjustments(current_dists, adjustments, cal_config)

        # End of mid loop — if we get here without converging,
        # outer loop could rebuild personas (not implemented in MVP)

    # Did not converge
    result = run_simulation(sim_config, current_dists, real_data=real_data)
    final_report_path = Path(report_dir) / "latest_report.json" if report_dir else None
    report = generate_report(result, current_dists, sim_config,
                             output_path=final_report_path, real_data=real_data)
    fm = report.get("fidelity_multidim", {})

    return CalibrationResult(
        converged=False, iterations_used=total_iterations,
        initial_f=initial_f, final_f=fm.get("F_multidim", 0),
        history=history, final_distributions=current_dists,
        final_report=report,
    )


def _compute_adjustments(
    distributions, per_dim, raw_metrics, report, cal_config, real_data,
) -> list[dict]:
    """Determine what parameters to adjust based on fidelity gaps."""
    adjustments = []

    # 1. Watch ratio distribution gap → adjust Beta parameters
    wr_score = per_dim.get("watch_ratio_js", 1.0)
    if wr_score < cal_config.per_dim_targets.get("watch_ratio_js", 0.5):
        # Compare target vs actual avg watch ratio
        fidelity = report.get("fidelity", {})
        target_wr = fidelity.get("target_avg_wr", 0.5)
        actual_wr = fidelity.get("actual_avg_wr", 0.5)
        delta = target_wr - actual_wr

        if abs(delta) > 0.01:
            adjustments.append({
                "type": "watch_ratio_shift",
                "delta": delta,
                "reason": f"avg WR gap: target={target_wr:.3f} actual={actual_wr:.3f}",
            })

    # 2. Conditional distribution gap → adjust per-category behavior
    cond_score = per_dim.get("conditional_avg_delta", 1.0)
    if cond_score < cal_config.per_dim_targets.get("conditional_avg_delta", 0.5):
        cond_data = report.get("fidelity_multidim", {}).get("conditional_wr_by_category", {})
        for cat, vals in cond_data.items():
            cat_delta = vals.get("real_mean", 0.5) - vals.get("sim_mean", 0.5)
            if abs(cat_delta) > 0.05:
                adjustments.append({
                    "type": "conditional_wr_shift",
                    "category": cat,
                    "delta": cat_delta,
                    "reason": f"cat {cat}: real={vals.get('real_mean',0):.3f} sim={vals.get('sim_mean',0):.3f}",
                })

    return adjustments


def _apply_adjustments(
    distributions: list[ArchetypeDistribution],
    adjustments: list[dict],
    cal_config: CalibrationConfig,
) -> list[ArchetypeDistribution]:
    """Apply parameter adjustments to distributions with regularization."""
    new_dists = [copy.deepcopy(d) for d in distributions]
    lr = cal_config.learning_rate
    reg = cal_config.regularization

    for adj in adjustments:
        if adj["type"] == "watch_ratio_shift":
            delta = adj["delta"]
            for d in new_dists:
                # Shift Beta distribution mean by adjusting a parameter
                # Beta mean = a / (a + b), increase a to increase mean
                original_mean = d.watch_ratio_beta_a / (d.watch_ratio_beta_a + d.watch_ratio_beta_b)
                target_shift = lr * delta

                # Regularization: don't shift too far from original
                max_shift = original_mean / reg
                clamped_shift = max(min(target_shift, max_shift), -max_shift)

                # Adjust a to shift the mean
                total = d.watch_ratio_beta_a + d.watch_ratio_beta_b
                new_mean = np.clip(original_mean + clamped_shift, 0.05, 0.95)
                d.watch_ratio_beta_a = float(new_mean * total)
                d.watch_ratio_beta_b = float((1 - new_mean) * total)
                d.watch_ratio_mean = float(new_mean)

        elif adj["type"] == "conditional_wr_shift":
            # Adjust distributions that are most associated with this category
            # For MVP: apply a smaller shift to all distributions
            delta = adj["delta"]
            for d in new_dists:
                shift = lr * delta * 0.3  # smaller effect for conditional
                original_mean = d.watch_ratio_beta_a / (d.watch_ratio_beta_a + d.watch_ratio_beta_b)
                max_shift = original_mean / reg
                clamped_shift = max(min(shift, max_shift), -max_shift)
                total = d.watch_ratio_beta_a + d.watch_ratio_beta_b
                new_mean = np.clip(original_mean + clamped_shift, 0.05, 0.95)
                d.watch_ratio_beta_a = float(new_mean * total)
                d.watch_ratio_beta_b = float((1 - new_mean) * total)
                d.watch_ratio_mean = float(new_mean)

    return new_dists
