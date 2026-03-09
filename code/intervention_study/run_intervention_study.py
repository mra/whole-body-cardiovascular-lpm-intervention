#!/usr/bin/env python3
"""Hypothesis-driven intervention study on top of the whole-body cardiovascular model.

Study hypothesis:
In hypotension, vasopressor-like vasoconstriction restores MAP but reduces renal and
hepatic perfusion more than cerebral perfusion.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import diffrax
import jax
import jax.numpy as jnp

# Keep matplotlib cache writable inside sandboxed environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "codex_mplconfig"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jax import jit

# Allow importing core model modules from ../
SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data_structures import Params, State
from dynamics import cardiovascular_model


jax.config.update("jax_enable_x64", True)


HYPOTENSION_MULTIPLIERS: tuple[tuple[tuple[str, ...], float], ...] = (
    (("LV", "EA"), 0.80),
    (("RV", "EA"), 0.85),
    (("Compliance", "C_SVC"), 1.20),
    (("Compliance", "C_IVC"), 1.20),
)

PRESSOR_RESISTANCE_KEYS: tuple[tuple[str, ...], ...] = (
    ("Resistance", "R_CerT"),
    ("Resistance", "R_RenT"),
    ("Resistance", "R_MesT"),
    ("Resistance", "R_HepT"),
    ("Resistance", "R_SplT"),
    ("Resistance", "R_ULimbT"),
    ("Resistance", "R_LLimbT"),
)

UNCERTAINTY_KEYS: tuple[tuple[str, ...], ...] = (
    ("BPM",),
    ("LV", "EA"),
    ("Resistance", "R_SVC"),
    ("Resistance", "R_RenT"),
    ("Resistance", "R_HepT"),
)


@dataclass(frozen=True)
class Scenario:
    name: str
    short: str
    apply_hypotension: bool
    pressor_scale: float | None


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(name="S0_Baseline", short="S0", apply_hypotension=False, pressor_scale=None),
    Scenario(name="S1_Hypotension", short="S1", apply_hypotension=True, pressor_scale=None),
    Scenario(name="S2_LowPressor", short="S2", apply_hypotension=True, pressor_scale=1.10),
    Scenario(name="S3_MediumPressor", short="S3", apply_hypotension=True, pressor_scale=1.25),
    Scenario(name="S4_HighPressor", short="S4", apply_hypotension=True, pressor_scale=1.40),
)


@jit
def _solve_model(init_state: State, params: Params, t1: float, save_times: jnp.ndarray) -> State:
    sol = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(cardiovascular_model),
        solver=diffrax.Tsit5(),
        t0=0.0,
        t1=t1,
        dt0=1e-3,
        y0=init_state,
        args=params,
        saveat=diffrax.SaveAt(ts=save_times),
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
        max_steps=int(1e9),
    )
    return sol.ys


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _multiply_at_path(data: dict[str, Any], key_path: tuple[str, ...], factor: float) -> None:
    ref: Any = data
    for key in key_path[:-1]:
        if key not in ref or not isinstance(ref[key], dict):
            raise KeyError(f"Missing nested key path component: {key_path}")
        ref = ref[key]
    leaf = key_path[-1]
    if leaf not in ref:
        raise KeyError(f"Missing key: {key_path}")
    ref[leaf] = float(ref[leaf]) * float(factor)


def build_scenario_params(base_params_dict: dict[str, Any], scenario: Scenario) -> dict[str, Any]:
    d = copy.deepcopy(base_params_dict)

    if scenario.apply_hypotension:
        for key_path, factor in HYPOTENSION_MULTIPLIERS:
            _multiply_at_path(d, key_path, factor)

    if scenario.pressor_scale is not None:
        for key_path in PRESSOR_RESISTANCE_KEYS:
            _multiply_at_path(d, key_path, scenario.pressor_scale)

    return d


def run_single_simulation(
    init_state: State,
    params_dict: dict[str, Any],
    n_cycles: int,
    tspan: int,
) -> dict[str, float]:
    params = Params.from_dict(params_dict)
    cycle_period = 60.0 / float(params.BPM)
    t0 = (n_cycles - 1) * cycle_period
    t1 = n_cycles * cycle_period
    save_times = jnp.linspace(t0, t1, tspan)

    output = _solve_model(init_state, params, float(t1), save_times)

    p_aa = np.asarray(output.P_AA)
    q_aa = np.asarray(output.Q_AA)
    q_cer = np.asarray(output.Q_Cer)
    q_ren = np.asarray(output.Q_Ren2)
    q_hep_in = np.asarray(output.Q_Hep1 + output.Q_Ven_Por)

    return {
        "MAP_proxy_mmHg": float(np.mean(p_aa)),
        "CO_ml_s": float(np.mean(q_aa)),
        "Q_Cer_ml_s": float(np.mean(q_cer)),
        "Q_Ren_ml_s": float(np.mean(q_ren)),
        "Q_HepIn_ml_s": float(np.mean(q_hep_in)),
    }


def _compute_uncertainty_samples(
    init_state: State,
    scenario_params: dict[str, Any],
    n_cycles: int,
    tspan: int,
    n_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []

    for i in range(n_samples):
        if (i + 1) % 10 == 0 or i == 0 or i + 1 == n_samples:
            print(f"[INFO]     sample {i + 1}/{n_samples}", flush=True)
        p = copy.deepcopy(scenario_params)
        for key_path in UNCERTAINTY_KEYS:
            factor = float(rng.uniform(0.9, 1.1))
            _multiply_at_path(p, key_path, factor)
        metrics = run_single_simulation(init_state, p, n_cycles=n_cycles, tspan=tspan)
        metrics["sample_index"] = i
        rows.append(metrics)

    return pd.DataFrame(rows)


def _ci_summary(df: pd.DataFrame, metric: str) -> tuple[float, float, float]:
    values = df[metric].to_numpy(dtype=float)
    return (
        float(np.median(values)),
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def _save_effect_size_table(det_df: pd.DataFrame, path: Path) -> None:
    baseline = det_df.loc[det_df["scenario_short"] == "S0"].iloc[0]

    out_rows: list[dict[str, Any]] = []
    for _, row in det_df.iterrows():
        out_rows.append(
            {
                "scenario": row["scenario"],
                "scenario_short": row["scenario_short"],
                "MAP_proxy_mmHg": row["MAP_proxy_mmHg"],
                "CO_ml_s": row["CO_ml_s"],
                "Q_Cer_ml_s": row["Q_Cer_ml_s"],
                "Q_Ren_ml_s": row["Q_Ren_ml_s"],
                "Q_HepIn_ml_s": row["Q_HepIn_ml_s"],
                "MAP_change_pct_vs_baseline": 100.0 * (row["MAP_proxy_mmHg"] - baseline["MAP_proxy_mmHg"]) / baseline["MAP_proxy_mmHg"],
                "CO_change_pct_vs_baseline": 100.0 * (row["CO_ml_s"] - baseline["CO_ml_s"]) / baseline["CO_ml_s"],
                "Q_Cer_change_pct_vs_baseline": 100.0 * (row["Q_Cer_ml_s"] - baseline["Q_Cer_ml_s"]) / baseline["Q_Cer_ml_s"],
                "Q_Ren_change_pct_vs_baseline": 100.0 * (row["Q_Ren_ml_s"] - baseline["Q_Ren_ml_s"]) / baseline["Q_Ren_ml_s"],
                "Q_HepIn_change_pct_vs_baseline": 100.0 * (row["Q_HepIn_ml_s"] - baseline["Q_HepIn_ml_s"]) / baseline["Q_HepIn_ml_s"],
                "MAP_target_met_ge_65": bool(row["MAP_proxy_mmHg"] >= 65.0),
            }
        )

    out_df = pd.DataFrame(out_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(path, index=False)


def _plot_dose_response(
    det_df: pd.DataFrame,
    uncertainty: dict[str, pd.DataFrame],
    figure_path: Path,
) -> None:
    baseline = det_df.loc[det_df["scenario_short"] == "S0"].iloc[0]

    dose_order = ["S1", "S2", "S3", "S4"]
    det_lookup = {r["scenario_short"]: r for _, r in det_df.iterrows()}

    x = np.arange(len(dose_order))
    map_med: list[float] = []
    map_lo: list[float] = []
    map_hi: list[float] = []

    flow_metrics = {
        "Cerebral": "Q_Cer_ml_s",
        "Renal": "Q_Ren_ml_s",
        "Hepatic": "Q_HepIn_ml_s",
    }
    flow_med: dict[str, list[float]] = {k: [] for k in flow_metrics}
    flow_lo: dict[str, list[float]] = {k: [] for k in flow_metrics}
    flow_hi: dict[str, list[float]] = {k: [] for k in flow_metrics}

    for short in dose_order:
        if short in uncertainty and not uncertainty[short].empty:
            udf = uncertainty[short]
            med, lo, hi = _ci_summary(udf, "MAP_proxy_mmHg")
            map_med.append(med)
            map_lo.append(lo)
            map_hi.append(hi)

            for label, metric in flow_metrics.items():
                vals = udf[metric].to_numpy(dtype=float) / float(baseline[metric])
                flow_med[label].append(float(np.median(vals)))
                flow_lo[label].append(float(np.quantile(vals, 0.025)))
                flow_hi[label].append(float(np.quantile(vals, 0.975)))
        else:
            map_val = float(det_lookup[short]["MAP_proxy_mmHg"])
            map_med.append(map_val)
            map_lo.append(map_val)
            map_hi.append(map_val)

            for label, metric in flow_metrics.items():
                v = float(det_lookup[short][metric] / baseline[metric])
                flow_med[label].append(v)
                flow_lo[label].append(v)
                flow_hi[label].append(v)

    fig, ax1 = plt.subplots(figsize=(11, 6), dpi=300)

    # MAP (left y-axis)
    ax1.plot(x, map_med, marker="o", linewidth=2.4, color="#1f77b4", label="MAP (median)")
    ax1.fill_between(x, map_lo, map_hi, alpha=0.2, color="#1f77b4", label="MAP 95% interval")
    ax1.axhline(65.0, linestyle="--", linewidth=1.5, color="black", label="MAP target (65 mmHg)")
    ax1.set_ylabel("MAP ($P_{AA}$) [mmHg]")

    # Normalized organ flows (right y-axis)
    ax2 = ax1.twinx()
    colors = {"Cerebral": "#2ca02c", "Renal": "#d62728", "Hepatic": "#9467bd"}
    for label in ["Cerebral", "Renal", "Hepatic"]:
        ax2.plot(x, flow_med[label], marker="s", linewidth=2.1, color=colors[label], label=f"{label} flow (norm)")
        ax2.fill_between(x, flow_lo[label], flow_hi[label], alpha=0.15, color=colors[label])

    ax2.set_ylabel("Normalized flow (vs baseline)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(dose_order)
    ax1.set_xlabel("Intervention dose level")
    ax1.grid(alpha=0.25)
    # ax1.set_title("Dose-response: MAP restoration vs organ perfusion trade-offs")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="center right", frameon=False)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run high-impact intervention hypothesis study.")
    parser.add_argument("--n-cycles", type=int, default=5, help="Number of cycles before extracting final cycle metrics.")
    parser.add_argument("--tspan", type=int, default=200, help="Samples across final cycle.")
    parser.add_argument("--n-samples", type=int, default=1000, help="Uncertainty samples per scenario.")
    parser.add_argument(
        "--uq-n-cycles",
        type=int,
        default=None,
        help="Optional cycle count for uncertainty simulations only (defaults to --n-cycles).",
    )
    parser.add_argument(
        "--uq-tspan",
        type=int,
        default=None,
        help="Optional tspan for uncertainty simulations only (defaults to --tspan).",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for uncertainty sampling.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "output",
        help="Output directory for figures/tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    uq_n_cycles = args.uq_n_cycles if args.uq_n_cycles is not None else args.n_cycles
    uq_tspan = args.uq_tspan if args.uq_tspan is not None else args.tspan
    fig_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    params_path = CODE_DIR / "parameters" / "params_bsln.json"
    state_path = CODE_DIR / "parameters" / "state_bsln.json"

    base_params_dict = _load_json(params_path)
    init_state = State.from_dict(_load_json(state_path))

    rng = np.random.default_rng(args.seed)

    deterministic_rows: list[dict[str, Any]] = []
    uncertainty_map: dict[str, pd.DataFrame] = {}
    scenario_payload: dict[str, Any] = {}

    for scenario in SCENARIOS:
        print(f"[INFO] Scenario {scenario.short}: {scenario.name}", flush=True)
        p = build_scenario_params(base_params_dict, scenario)
        scenario_payload[scenario.short] = p

        det_metrics = run_single_simulation(init_state, p, n_cycles=args.n_cycles, tspan=args.tspan)
        deterministic_rows.append({"scenario": scenario.name, "scenario_short": scenario.short, **det_metrics})

        if args.n_samples > 0:
            print(
                f"[INFO]   uncertainty sampling: n={args.n_samples}, "
                f"n_cycles={uq_n_cycles}, tspan={uq_tspan}",
                flush=True,
            )
            uncertainty_map[scenario.short] = _compute_uncertainty_samples(
                init_state,
                p,
                n_cycles=uq_n_cycles,
                tspan=uq_tspan,
                n_samples=args.n_samples,
                rng=rng,
            )

    det_df = pd.DataFrame(deterministic_rows)
    det_df.to_csv(table_dir / "deterministic_metrics_by_scenario.csv", index=False)

    _save_effect_size_table(det_df, table_dir / "intervention_effect_sizes.csv")

    # Long-form uncertainty summary table
    summary_rows: list[dict[str, Any]] = []
    for short, udf in uncertainty_map.items():
        for metric in ["MAP_proxy_mmHg", "CO_ml_s", "Q_Cer_ml_s", "Q_Ren_ml_s", "Q_HepIn_ml_s"]:
            med, lo, hi = _ci_summary(udf, metric)
            summary_rows.append(
                {
                    "scenario_short": short,
                    "metric": metric,
                    "median": med,
                    "ci2p5": lo,
                    "ci97p5": hi,
                }
            )
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(table_dir / "uncertainty_summary.csv", index=False)

    _plot_dose_response(
        det_df=det_df,
        uncertainty=uncertainty_map,
        figure_path=fig_dir / "Figure_H1_intervention_dose_response.png",
    )

    _write_json(output_dir / "scenario_parameters.json", scenario_payload)

    print("[DONE] Intervention study artifacts written:")
    print(f"  - {table_dir / 'deterministic_metrics_by_scenario.csv'}")
    print(f"  - {table_dir / 'intervention_effect_sizes.csv'}")
    if summary_rows:
        print(f"  - {table_dir / 'uncertainty_summary.csv'}")
    print(f"  - {fig_dir / 'Figure_H1_intervention_dose_response.png'}")
    print(f"  - {output_dir / 'scenario_parameters.json'}")


if __name__ == "__main__":
    main()
