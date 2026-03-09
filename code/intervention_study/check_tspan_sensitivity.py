#!/usr/bin/env python3
"""Assess final-cycle sampling (tspan) sensitivity for intervention metrics.

The script compares candidate tspans against a higher-resolution reference tspan and
reports relative errors for MAP and organ-flow metrics.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_intervention_study as ris


DEFAULT_METRICS: tuple[tuple[str, str], ...] = (
    ("MAP", "MAP_proxy_mmHg"),
    ("Q_Cer", "Q_Cer_ml_s"),
    ("Q_Ren", "Q_Ren_ml_s"),
    ("Q_HepIn", "Q_HepIn_ml_s"),
)


def _parse_tspan_list(raw: str) -> list[int]:
    tspans = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not tspans:
        raise ValueError("At least one candidate tspan is required.")
    if any(x <= 0 for x in tspans):
        raise ValueError("All tspan values must be > 0.")
    return tspans


def _relative_error_pct(value: float, reference: float) -> float:
    denom = abs(reference) if abs(reference) > 1e-12 else 1.0
    return abs(value - reference) / denom * 100.0


def _apply_uncertainty_perturbations(params_dict: dict, rng: np.random.Generator) -> dict:
    p = copy.deepcopy(params_dict)
    for key_path in ris.UNCERTAINTY_KEYS:
        ris._multiply_at_path(p, key_path, float(rng.uniform(0.9, 1.1)))
    return p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check tspan sensitivity for intervention metrics.")
    parser.add_argument("--scenario", type=str, default="S4", help="Scenario short name (S0..S4).")
    parser.add_argument(
        "--candidate-tspans",
        type=str,
        default="100,200,400,800",
        help="Comma-separated candidate tspan values.",
    )
    parser.add_argument(
        "--reference-tspan",
        type=int,
        default=1600,
        help="Reference tspan used as high-resolution baseline.",
    )
    parser.add_argument(
        "--n-cycles",
        type=int,
        default=41,
        help="Cycle count used for all tspan comparisons.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="Number of parameter samples (with uncertainty perturbations unless --deterministic).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Disable uncertainty perturbations and use nominal scenario parameters.",
    )
    parser.add_argument(
        "--include-co",
        action="store_true",
        help="Also evaluate cardiac output (CO) sensitivity.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for perturbations.")
    parser.add_argument(
        "--tol-pct",
        type=float,
        default=0.5,
        help="Tolerance (%) used to recommend the smallest acceptable tspan.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "tspan_sensitivity_output",
        help="Directory to save sensitivity summary tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    candidate_tspans = _parse_tspan_list(args.candidate_tspans)
    if args.reference_tspan <= 0:
        raise ValueError("--reference-tspan must be > 0.")
    if args.reference_tspan in candidate_tspans:
        candidate_tspans = [x for x in candidate_tspans if x != args.reference_tspan]
    if not candidate_tspans:
        raise ValueError("Candidate tspans cannot be empty after removing --reference-tspan.")
    if args.n_samples <= 0:
        raise ValueError("--n-samples must be > 0.")

    scenario_lookup = {s.short: s for s in ris.SCENARIOS}
    if args.scenario not in scenario_lookup:
        raise ValueError(f"Unknown scenario '{args.scenario}'. Choose from: {sorted(scenario_lookup)}")
    scenario = scenario_lookup[args.scenario]

    metrics = list(DEFAULT_METRICS)
    if args.include_co:
        metrics.append(("CO", "CO_ml_s"))

    base_params_dict = ris._load_json(ris.CODE_DIR / "parameters" / "params_bsln.json")
    init_state = ris.State.from_dict(ris._load_json(ris.CODE_DIR / "parameters" / "state_bsln.json"))
    nominal_scenario_params = ris.build_scenario_params(base_params_dict, scenario)
    rng = np.random.default_rng(args.seed)

    sample_rows: list[dict[str, float | int | str | bool]] = []
    print(
        f"[INFO] tspan sensitivity: scenario={args.scenario}, n_cycles={args.n_cycles}, "
        f"reference_tspan={args.reference_tspan}, n_samples={args.n_samples}, "
        f"deterministic={args.deterministic}"
    )

    for sample_idx in range(args.n_samples):
        if args.deterministic:
            params_dict = copy.deepcopy(nominal_scenario_params)
        else:
            params_dict = _apply_uncertainty_perturbations(nominal_scenario_params, rng)

        ref_metrics = ris.run_single_simulation(
            init_state,
            params_dict,
            n_cycles=args.n_cycles,
            tspan=args.reference_tspan,
        )

        for tspan in candidate_tspans:
            tst_metrics = ris.run_single_simulation(
                init_state,
                params_dict,
                n_cycles=args.n_cycles,
                tspan=tspan,
            )
            for metric_name, metric_key in metrics:
                err_pct = _relative_error_pct(tst_metrics[metric_key], ref_metrics[metric_key])
                sample_rows.append(
                    {
                        "sample_index": sample_idx,
                        "tspan": tspan,
                        "metric": metric_name,
                        "value_candidate": tst_metrics[metric_key],
                        "value_reference": ref_metrics[metric_key],
                        "rel_error_pct": err_pct,
                    }
                )

    sample_df = pd.DataFrame(sample_rows)
    summary_rows: list[dict[str, float | int | bool]] = []

    for tspan in sorted(sample_df["tspan"].unique()):
        g = sample_df.loc[sample_df["tspan"] == tspan]
        max_by_metric = g.groupby("metric")["rel_error_pct"].max()
        summary_rows.append(
            {
                "tspan": int(tspan),
                "max_error_pct_all_metrics": float(g["rel_error_pct"].max()),
                "p95_error_pct_all_metrics": float(g["rel_error_pct"].quantile(0.95)),
                "max_error_pct_MAP": float(max_by_metric.get("MAP", np.nan)),
                "max_error_pct_Q_Cer": float(max_by_metric.get("Q_Cer", np.nan)),
                "max_error_pct_Q_Ren": float(max_by_metric.get("Q_Ren", np.nan)),
                "max_error_pct_Q_HepIn": float(max_by_metric.get("Q_HepIn", np.nan)),
                "passes_tol": bool((max_by_metric <= args.tol_pct).all()),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("tspan").reset_index(drop=True)
    passing = summary_df.loc[summary_df["passes_tol"]]
    recommended_tspan = int(passing["tspan"].min()) if not passing.empty else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_csv = args.output_dir / "tspan_sensitivity_sample_level.csv"
    summary_csv = args.output_dir / "tspan_sensitivity_summary.csv"
    config_json = args.output_dir / "tspan_sensitivity_config.json"
    sample_df.to_csv(sample_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    with config_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "scenario": args.scenario,
                "n_cycles": args.n_cycles,
                "candidate_tspans": candidate_tspans,
                "reference_tspan": args.reference_tspan,
                "n_samples": args.n_samples,
                "deterministic": args.deterministic,
                "tol_pct": args.tol_pct,
                "recommended_tspan": recommended_tspan,
            },
            f,
            indent=2,
        )

    print("[DONE] tspan sensitivity summary")
    print(summary_df.to_string(index=False))
    if recommended_tspan is None:
        print(f"[WARN] No candidate tspan satisfied tol={args.tol_pct:.3f}%.")
    else:
        print(f"[INFO] Recommended tspan at tol={args.tol_pct:.3f}%: {recommended_tspan}")
    print(f"[INFO] Sample-level table: {sample_csv}")
    print(f"[INFO] Summary table: {summary_csv}")
    print(f"[INFO] Config: {config_json}")


if __name__ == "__main__":
    main()
