#!/usr/bin/env python3
"""Sobol sensitivity analysis utilities for manuscript figures."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jax import jit
from matplotlib.colors import LinearSegmentedColormap
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

from data_structures import Params, State
from dynamics import cardiovascular_model

jax.config.update("jax_enable_x64", True)


# Union of parameters needed to reproduce the comprehensive + base Sobol figures.
SOBOL_PARAM_NAMES: list[str] = [
    "BPM",
    "EA_LV",
    "EA_RV",
    "EB_LA",
    "EB_LV",
    "EB_RA",
    "EB_RV",
    "V0_LV",
    "R_AA",
    "R_CerT",
    "R_FacT",
    "R_HepT",
    "R_IVC",
    "R_LLimbT",
    "R_RenT",
    "R_SVC",
    "R_ULimbT",
    "R_Ven_Por",
    "tC_LA",
    "tC_RA",
    "tC_LV",
    "tC_RV",
    "TR_LV",
    "TR_RV",
    "C_IVC",
    "C_SVC",
]

COMPREHENSIVE_BAR_OUTPUTS: list[str] = ["V_LV", "V_RV", "P_AA", "P_Ar_Pul", "Q_AA", "P_SVC"]
BASE_BAR_OUTPUTS: list[str] = ["V_LV", "V_RV", "P_Ar_sys", "P_Ar_Pul", "Q_Ar_sys", "Q_Ar_Pul"]

COMPREHENSIVE_HEATMAP_PARAMS: list[str] = [
    "C_IVC",
    "EB_LA",
    "EB_LV",
    "EB_RA",
    "R_CerT",
    "R_FacT",
    "R_HepT",
    "R_IVC",
    "R_LLimbT",
    "R_RenT",
    "R_SVC",
    "R_ULimbT",
    "R_Ven_Por",
    "tC_LA",
    "BPM",
]

BASE_HEATMAP_PARAMS: list[str] = [
    "EA_LV",
    "EA_RV",
    "EB_LA",
    "EB_LV",
    "EB_RA",
    "EB_RV",
    "R_AA",
    "R_SVC",
    "V0_LV",
    "BPM",
    "tC_LA",
    "tC_RA",
]

BASE_HEATMAP_OUTPUTS: list[str] = [
    "V_LA",
    "V_LV",
    "V_RA",
    "V_RV",
    "P_Ar_sys",
    "P_Ven_sys",
    "P_Ar_Pul",
    "P_Ven_Pul",
    "Q_Ar_sys",
    "Q_Ven_sys",
    "Q_Ar_Pul",
    "Q_Ven_Pul",
]

COMPREHENSIVE_HEATMAP_OUTPUTS: list[str] = list(State._fields)

PARAM_LABELS: dict[str, str] = {
    "BPM": "BPM",
    "EA_LV": r"$EA_{LV}$",
    "EA_RV": r"$EA_{RV}$",
    "EB_LA": r"$EB_{LA}$",
    "EB_LV": r"$EB_{LV}$",
    "EB_RA": r"$EB_{RA}$",
    "EB_RV": r"$EB_{RV}$",
    "V0_LV": r"$V0_{LV}$",
    "R_AA": r"$R_{Ar}^{Sys}$",
    "R_CerT": r"$R_{CerT}$",
    "R_FacT": r"$R_{FacT}$",
    "R_HepT": r"$R_{HepT}$",
    "R_IVC": r"$R_{IVC}$",
    "R_LLimbT": r"$R_{LLimbT}$",
    "R_RenT": r"$R_{RenT}$",
    "R_SVC": r"$R_{SVC}$",
    "R_ULimbT": r"$R_{ULimbT}$",
    "R_Ven_Por": r"$R_{Ven\,Por}$",
    "tC_LA": r"$tC_{LA}$",
    "tC_RA": r"$tC_{RA}$",
    "tC_LV": r"$tC_{LV}$",
    "tC_RV": r"$tC_{RV}$",
    "TR_LV": r"$TR_{LV}$",
    "TR_RV": r"$TR_{RV}$",
    "C_IVC": r"$C_{IVC}$",
    "C_SVC": r"$C_{SVC}$",
}

OUTPUT_LABELS: dict[str, str] = {
    "P_AA": r"$P_{AA}$",
    "P_SVC": r"$P_{SVC}$",
    "P_Ar_Pul": r"$P_{Ar}^{Pul}$",
    "P_Ven_Pul": r"$P_{Ven}^{Pul}$",
    "Q_AA": r"$Q_{AA}$",
    "Q_Ar_Pul": r"$Q_{Ar}^{Pul}$",
    "Q_Ven_Pul": r"$Q_{Ven}^{Pul}$",
    "P_Ar_sys": r"$P_{Ar}^{Sys}$",
    "P_Ven_sys": r"$P_{Ven}^{Sys}$",
    "Q_Ar_sys": r"$Q_{Ar}^{Sys}$",
    "Q_Ven_sys": r"$Q_{Ven}^{Sys}$",
}


@dataclass(frozen=True)
class SobolConfig:
    n_samples: int = 1024
    n_cycles: int = 40
    tspan: int = 400
    variation: float = 0.10
    seed: int = 42
    threshold: float = 0.20


def _pretty_label(name: str, overrides: dict[str, str]) -> str:
    if name in overrides:
        return overrides[name]
    if "_" not in name:
        return name
    head, tail = name.split("_", 1)
    return rf"${head}_{{{tail}}}$"


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
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6),
        max_steps=int(1e8),
    )
    return sol.ys


def _run_final_cycle(init_state: State, params: Params, n_cycles: int, tspan: int) -> tuple[State, np.ndarray]:
    cycle_period = 60.0 / float(params.BPM)
    all_times = jnp.linspace(0.0, n_cycles * cycle_period, n_cycles * tspan)
    save_times = all_times[-tspan:]
    output = _solve_model(init_state, params, float(all_times[-1]), save_times)

    save_times_np = np.asarray(save_times)
    final_times = save_times_np - save_times_np[0]
    return output, final_times


def _extract_output_series(output: State) -> dict[str, np.ndarray]:
    series = {name: np.asarray(getattr(output, name)) for name in State._fields}
    # Synthetic outputs used by base-model-style figures.
    series["P_Ar_sys"] = series["P_AA"]
    series["P_Ven_sys"] = 0.5 * (series["P_SVC"] + series["P_IVC"])
    series["Q_Ar_sys"] = series["Q_AA"]
    series["Q_Ven_sys"] = series["Q_SVC"] + series["Q_IVC"]
    return series


def _build_problem(base_params: Params, param_names: Iterable[str], variation: float) -> dict:
    names: list[str] = []
    bounds: list[list[float]] = []

    for name in param_names:
        nominal = float(getattr(base_params, name))
        delta = abs(nominal) * variation
        if delta == 0.0:
            delta = 1e-6
        low = nominal - delta
        high = nominal + delta
        bounds.append([min(low, high), max(low, high)])
        names.append(name)

    return {"num_vars": len(names), "names": names, "bounds": bounds}


def _sample_problem(problem: dict, config: SobolConfig) -> np.ndarray:
    return sobol_sample.sample(
        problem,
        N=config.n_samples,
        calc_second_order=False,
        scramble=True,
        seed=config.seed,
    )


def _evaluate_samples(
    init_state: State,
    base_params: Params,
    problem: dict,
    sample_matrix: np.ndarray,
    output_names: list[str],
    config: SobolConfig,
) -> tuple[dict[str, np.ndarray], int]:
    y = {name: np.empty(sample_matrix.shape[0], dtype=float) for name in output_names}

    # Warm-up compile.
    _run_final_cycle(init_state, base_params, max(config.n_cycles // 2, 2), max(config.tspan // 2, 50))

    failures = 0
    total = sample_matrix.shape[0]
    print(f"[INFO] Sobol evaluations: {total} model runs")

    for i, row in enumerate(sample_matrix):
        params = base_params._replace(**{k: float(v) for k, v in zip(problem["names"], row)})
        try:
            output, final_times = _run_final_cycle(init_state, params, config.n_cycles, config.tspan)
            series = _extract_output_series(output)

            for out_name in output_names:
                y[out_name][i] = float(np.trapezoid(series[out_name], final_times))
        except Exception:
            failures += 1
            for out_name in output_names:
                y[out_name][i] = np.nan

        if (i + 1) % 50 == 0 or i == 0 or i + 1 == total:
            print(f"[INFO] Sobol progress: {i + 1}/{total}")

    # Keep Sobol analysis numerically stable even if a few runs fail.
    for out_name, values in y.items():
        invalid = ~np.isfinite(values)
        if not invalid.any():
            continue
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise RuntimeError(f"All Sobol samples failed for output '{out_name}'.")
        fill_value = float(np.median(finite))
        values[invalid] = fill_value

    return y, failures


def _analyze_sobol(problem: dict, y: dict[str, np.ndarray], outputs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    st = pd.DataFrame(index=problem["names"], columns=outputs, dtype=float)
    st_conf = pd.DataFrame(index=problem["names"], columns=outputs, dtype=float)

    for out_name in outputs:
        result = sobol_analyze.analyze(
            problem,
            y[out_name],
            calc_second_order=False,
            print_to_console=False,
        )
        st[out_name] = result["ST"]
        st_conf[out_name] = result["ST_conf"]

    return st, st_conf


def _plot_top5_bars(
    st: pd.DataFrame,
    st_conf: pd.DataFrame,
    outputs: list[str],
    out_path: Path,
) -> None:
    fig, axs = plt.subplots(2, 3, figsize=(18, 8), dpi=300)
    axes = axs.flatten()

    for idx, out_name in enumerate(outputs):
        ax = axes[idx]
        top = st[out_name].sort_values(ascending=False).head(5)
        errs = st_conf.loc[top.index, out_name].clip(lower=0.0)

        x = np.arange(len(top))
        ax.bar(x, top.to_numpy(), color="#b0a1c7", alpha=0.9)
        ax.errorbar(x, top.to_numpy(), yerr=errs.to_numpy(), fmt="none", ecolor="black", elinewidth=1.8, capsize=5, capthick=1.8)

        labels = [_pretty_label(name, PARAM_LABELS) for name in top.index]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel(_pretty_label(out_name, OUTPUT_LABELS), fontsize=22)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, axis="y", linestyle="--", linewidth=1.0, alpha=0.6)

    for ax in axes[len(outputs):]:
        ax.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap(
    st: pd.DataFrame,
    param_order: list[str],
    output_order: list[str],
    out_path: Path,
    threshold: float,
    figsize: tuple[float, float],
) -> None:
    mat = st.loc[param_order, output_order].copy()
    mat = mat.where(mat >= threshold)
    mask = mat.isna()

    cmap = LinearSegmentedColormap.from_list(
        "sobol_lavender",
        ["#eeeeee", "#d8d3e2", "#afa4c2", "#8f84aa"],
    )

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    sns.heatmap(
        mat,
        mask=mask,
        cmap=cmap,
        vmin=threshold,
        vmax=0.7,
        linewidths=0.5,
        linecolor="#9e9e9e",
        cbar_kws={"label": "ST Index"},
        ax=ax,
    )
    ax.set_facecolor("#ececec")
    ax.set_xlabel("Output Variable")
    ax.set_ylabel("Parameter")

    ax.set_xticklabels([_pretty_label(name, OUTPUT_LABELS) for name in output_order], rotation=35, ha="right")
    ax.set_yticklabels([_pretty_label(name, PARAM_LABELS) for name in param_order], rotation=0)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def generate_sobol_figures(
    init_state: State,
    base_params: Params,
    figures_main_dir: Path,
    figures_supp_dir: Path,
    analysis_dir: Path,
    config: SobolConfig,
) -> dict[str, object]:
    required_outputs = sorted(
        set(COMPREHENSIVE_HEATMAP_OUTPUTS)
        | set(COMPREHENSIVE_BAR_OUTPUTS)
        | set(BASE_HEATMAP_OUTPUTS)
        | set(BASE_BAR_OUTPUTS)
    )

    problem = _build_problem(base_params, SOBOL_PARAM_NAMES, config.variation)
    sample_matrix = _sample_problem(problem, config)

    y, failures = _evaluate_samples(
        init_state=init_state,
        base_params=base_params,
        problem=problem,
        sample_matrix=sample_matrix,
        output_names=required_outputs,
        config=config,
    )

    st, st_conf = _analyze_sobol(problem, y, required_outputs)

    _plot_top5_bars(
        st=st,
        st_conf=st_conf,
        outputs=COMPREHENSIVE_BAR_OUTPUTS,
        out_path=figures_main_dir / "Figure6.png",
    )
    _plot_heatmap(
        st=st,
        param_order=COMPREHENSIVE_HEATMAP_PARAMS,
        output_order=COMPREHENSIVE_HEATMAP_OUTPUTS,
        out_path=figures_supp_dir / "Fig7.png",
        threshold=config.threshold,
        figsize=(22, 9),
    )

    _plot_top5_bars(
        st=st,
        st_conf=st_conf,
        outputs=BASE_BAR_OUTPUTS,
        out_path=figures_supp_dir / "Top_5_ST_Sobol_Indices_base.png",
    )
    _plot_heatmap(
        st=st,
        param_order=BASE_HEATMAP_PARAMS,
        output_order=BASE_HEATMAP_OUTPUTS,
        out_path=figures_supp_dir / "global_ST_basemodel_heatmap.png",
        threshold=config.threshold,
        figsize=(11, 9),
    )

    analysis_dir.mkdir(parents=True, exist_ok=True)
    st.to_csv(analysis_dir / "sobol_st_indices.csv", index_label="Parameter")
    st_conf.to_csv(analysis_dir / "sobol_st_confidence.csv", index_label="Parameter")

    settings = {
        "config": asdict(config),
        "num_variables": problem["num_vars"],
        "sample_count": int(sample_matrix.shape[0]),
        "failed_runs": int(failures),
        "parameters": problem["names"],
        "outputs": required_outputs,
    }
    with (analysis_dir / "sobol_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)

    return settings
