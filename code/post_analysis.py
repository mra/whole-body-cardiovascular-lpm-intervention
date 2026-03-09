#!/usr/bin/env python3
"""Reproducible post-analysis pipeline for manuscript figures and tables.

This script replaces notebook-driven plotting by generating manuscript and
supplementary figures directly from model simulation outputs and reference CSVs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jax import jit
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

from common import ANALYSIS_DIR, CODE_DIR, FIGURES_DIR, MODEL_OUTPUT_DIR, ensure_pipeline_dirs, parse_cycle_list
from data_structures import HeartChamberPressureFlowRate, Params, State
from dynamics import cardiovascular_model, get_heart_chamber_pressures_and_flow_rates

try:
    from scipy.interpolate import splprep, splev

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


jax.config.update("jax_enable_x64", True)


@dataclass
class SimulationArtifacts:
    params: Params
    init_state: State
    output: State
    heart: object
    save_times: np.ndarray
    final_times: np.ndarray
    tspan: int


def configure_matplotlib() -> None:
    """Set publication-style defaults for all figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "DejaVu Serif",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 2.2,
        }
    )
    sns.set_style("whitegrid")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _csv_path(code_dir: Path, filename: str) -> Path:
    csv_dir = code_dir / "csv_files"
    candidate = csv_dir / filename
    if candidate.exists():
        return candidate
    return code_dir / filename


def load_state_and_params(data_dir: Path) -> tuple[State, Params]:
    init_state = State.from_dict(_load_json(data_dir / "state_bsln.json"))
    params = Params.from_dict(_load_json(data_dir / "params_bsln.json"))
    return init_state, params


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


def run_simulation(init_state: State, params: Params, n_cycles: int, tspan: int, last_cycles: int) -> SimulationArtifacts:
    cycle_period = 60.0 / float(params.BPM)
    all_times = jnp.linspace(0.0, n_cycles * cycle_period, n_cycles * tspan)
    save_times = all_times[-last_cycles * tspan :]

    output = _solve_model(init_state, params, float(all_times[-1]), save_times)
    heart = get_heart_chamber_pressures_and_flow_rates(save_times, output, params)

    save_times_np = np.asarray(save_times)
    final_times = save_times_np[-tspan:] - save_times_np[-tspan]

    return SimulationArtifacts(
        params=params,
        init_state=init_state,
        output=output,
        heart=heart,
        save_times=save_times_np - save_times_np[0],
        final_times=final_times,
        tspan=tspan,
    )


def _final_cycle(arr: jnp.ndarray, tspan: int) -> np.ndarray:
    return np.asarray(arr)[-tspan:]


def _styled_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time (s)")
    ax.tick_params(direction="out", which="major", length=4)
    ax.grid(True, alpha=0.25)


def _save(fig: plt.Figure, *paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _smooth_loop(volume: np.ndarray, pressure: np.ndarray, smooth: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    if not SCIPY_AVAILABLE:
        return volume, pressure
    tck, _ = splprep([volume, pressure], s=smooth)
    unew = np.linspace(0.0, 1.0, 300)
    v_new, p_new = splev(unew, tck)
    return np.asarray(v_new), np.asarray(p_new)


def plot_pv_loops(art: SimulationArtifacts, code_dir: Path, supp_dir: Path) -> None:
    df_lv_ref = pd.read_csv(_csv_path(code_dir, "phy_PV_loop.csv"))
    df_rv_ref = pd.read_csv(_csv_path(code_dir, "RV loop Ref1.csv"))
    df_base = pd.read_csv(_csv_path(code_dir, "base PV loop.csv"))

    v_lv_std, p_lv_std = _smooth_loop(df_lv_ref["Volume"].to_numpy(), df_lv_ref["Pressure"].to_numpy())
    v_rv_std, p_rv_std = _smooth_loop(df_rv_ref["Volume"].to_numpy(), df_rv_ref["Pressure"].to_numpy())

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    colors = sns.color_palette("Dark2", 8)

    axs[0].plot(_final_cycle(art.output.V_LV, art.tspan), _final_cycle(art.heart.P_LV, art.tspan), lw=2.8, color=colors[0], label="Simulated")
    axs[0].plot(v_lv_std, p_lv_std, lw=2.8, color="black", label="Standard")
    axs[0].plot(df_base["V_LV"], df_base["P_LV"], lw=2.8, color=colors[1], label="Base model")
    axs[0].set_title("Left Ventricle")
    axs[0].set_xlabel("Volume [mL]")
    axs[0].set_ylabel("Pressure [mmHg]")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(frameon=False)

    axs[1].plot(_final_cycle(art.output.V_RV, art.tspan), _final_cycle(art.heart.P_RV, art.tspan), lw=2.8, color=colors[0], label="Simulated")
    axs[1].plot(v_rv_std, p_rv_std, lw=2.8, color="black", label="Standard")
    axs[1].plot(df_base["V_RV"], df_base["P_RV"], lw=2.8, color=colors[1], label="Base model")
    axs[1].set_title("Right Ventricle")
    axs[1].set_xlabel("Volume [mL]")
    axs[1].set_ylabel("Pressure [mmHg]")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(frameon=False)

    _save(fig, supp_dir / "Fig9.png")


def plot_volume_conservation(art: SimulationArtifacts, supp_dir: Path) -> None:
    p = art.params
    o = art.output

    v_heart = o.V_LA + o.V_LV + o.V_RA + o.V_RV
    v_lung = p.C_Ven_Pul * o.P_Ven_Pul + p.C_Lung * o.P_Lung + p.C_LL * o.P_LL + p.C_RL * o.P_RL + p.C_Ar_Pul * o.P_Ar_Pul
    v_vasc = (
        p.C_SVC * o.P_SVC
        + p.C_IVC * o.P_IVC
        + p.C_Cer * o.P_Cer
        + p.C_Fac * o.P_Fac
        + p.C_ULimb * o.P_ULimb
        + p.C_AArc * o.P_AArc
        + p.C_CCar * o.P_CCar
        + p.C_AA * o.P_AA
        + p.C_DscA * o.P_DscA
        + p.C_AbdA * o.P_AbdA
        + p.C_Ren * o.P_Ren
        + p.C_Spl * o.P_Spl
        + p.C_Mes * o.P_Mes
        + p.C_Ven_Por * o.P_Ven_Por
        + p.C_Hep * o.P_Hep
        + p.C_LLimb * o.P_LLimb
        + p.C_Cel * o.P_Cel
    )

    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=300)
    palette = ["#70C16C", "#FF9933", "#B37CC2"]
    ax.plot(art.save_times, np.asarray(v_heart), color=palette[0], label="Heart volume", linestyle="-", marker="o", markevery=50)
    ax.plot(art.save_times, np.asarray(v_lung), color=palette[1], label="Lung volume", linestyle="--", marker="D", markevery=50)
    ax.plot(art.save_times, np.asarray(v_vasc), color=palette[2], label="Systemic vascular volume", linestyle="-.", marker="^", markevery=50)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Volume (mL)")
    ax.legend(loc="center right")
    ax.grid(True, alpha=0.3)

    _save(fig, supp_dir / "Fig8.png")


def extract_flow_groups(art: SimulationArtifacts) -> dict:
    o = art.output
    h = art.heart
    t = art.tspan

    organ = {
        "Right lung": _final_cycle(o.Q_RL2, t),
        "Left lung": _final_cycle(o.Q_LL2, t),
        "Cerebral": _final_cycle(o.Q_Cer, t),
        "Upper limb": _final_cycle(o.Q_ULimb, t),
        "Facial": _final_cycle(o.Q_Fac, t),
        "Renal": _final_cycle(o.Q_Ren2, t),
        "Mesenteric": _final_cycle(o.Q_Mes2, t),
        "Spleen, pancreas, gastric": _final_cycle(o.Q_Spl2, t),
        "Hepatic": _final_cycle(o.Q_Hep2, t),
        "Lower limb": _final_cycle(o.Q_LLimb2, t),
    }

    artery = {
        "Pulmonary artery trunk": _final_cycle(o.Q_Ar_Pul, t),
        "Pulmonary artery": _final_cycle(o.Q_LL1 + o.Q_RL1, t),
        "Pulmonary vein": _final_cycle(o.Q_Ven_Pul, t),
        "Ascending Aorta": _final_cycle(o.Q_AA, t),
        "Descending Aorta": _final_cycle(o.Q_DscA, t),
        "Abdominal Aorta": _final_cycle(o.Q_AbdA, t),
        "Carotid artery": _final_cycle(o.Q_CCar, t),
        "Celiac artery": _final_cycle(o.Q_Cel, t),
        "Subclavian": _final_cycle(o.Q_Sub, t),
        "External carotid": _final_cycle(o.Q_ECar, t),
        "Internal carotid": _final_cycle(o.Q_ICar, t),
        "Hepatic artery": _final_cycle(o.Q_Hep1, t),
        "Renal artery": _final_cycle(o.Q_Ren1, t),
        "Mesenteric artery": _final_cycle(o.Q_Mes1, t),
        "Splenic artery": _final_cycle(o.Q_Spl1, t),
        "Iliac artery": _final_cycle(o.Q_LLimb1, t),
    }

    veins = {
        "Inferior VC": _final_cycle(o.Q_IVC, t),
        "Superior VC": _final_cycle(o.Q_SVC, t),
        "Portal vein": _final_cycle(o.Q_Ven_Por, t),
    }

    # Flow-balance residuals used by convergence metrics (same definitions as notebooks).
    flow_diff_organ = {
        "Right lung": np.asarray(_final_cycle(o.Q_RL1 - o.Q_RL2, t)),
        "Left lung": np.asarray(_final_cycle(o.Q_LL1 - o.Q_LL2, t)),
        "Cerebral": np.asarray(_final_cycle(o.Q_ICar - o.Q_Cer, t)),
        "Upper limb": np.asarray(_final_cycle(o.Q_Sub - o.Q_ULimb, t)),
        "Facial": np.asarray(_final_cycle(o.Q_ECar - o.Q_Fac, t)),
        "Renal": np.asarray(_final_cycle(o.Q_Ren1 - o.Q_Ren2, t)),
        "Mesenteric": np.asarray(_final_cycle(o.Q_Mes1 - o.Q_Mes2, t)),
        "Spleen, pancreas, gastric": np.asarray(_final_cycle(o.Q_Spl1 - o.Q_Spl2, t)),
        "Hepatic": np.asarray(_final_cycle(o.Q_Hep1 - o.Q_Hep2, t)),
        "Lower limb": np.asarray(_final_cycle(o.Q_LLimb1 - o.Q_LLimb2, t)),
    }

    flow_diff_artery = {
        "Pulmonary artery trunk": np.asarray(_final_cycle(h.Q_PV - o.Q_Ar_Pul, t)),
        "Pulmonary artery": np.asarray(_final_cycle(o.Q_Ar_Pul - o.Q_LL1 - o.Q_RL1, t)),
        "Pulmonary vein": np.asarray(_final_cycle(o.Q_LL2 + o.Q_RL2 - o.Q_Ven_Pul, t)),
        "Ascending Aorta": np.asarray(_final_cycle(h.Q_AV - o.Q_AA, t)),
        "Descending Aorta": np.asarray(_final_cycle((o.Q_AA - o.Q_Sub - o.Q_CCar) - o.Q_DscA, t)),
        "Carotid artery": np.asarray(_final_cycle(o.Q_AA - o.Q_DscA - o.Q_Sub - o.Q_CCar, t)),
        "Celiac artery": np.asarray(_final_cycle(o.Q_AbdA - o.Q_Mes1 - o.Q_LLimb1 - o.Q_Cel, t)),
        "Subclavian": np.asarray(_final_cycle(o.Q_AA - o.Q_CCar - o.Q_DscA - o.Q_Sub, t)),
        "External carotid": np.asarray(_final_cycle(o.Q_CCar - o.Q_ICar - o.Q_ECar, t)),
        "Internal carotid": np.asarray(_final_cycle(o.Q_CCar - o.Q_ECar - o.Q_ICar, t)),
        "Hepatic artery": np.asarray(_final_cycle(o.Q_Cel - o.Q_Spl1 - o.Q_Hep1, t)),
        "Renal artery": np.asarray(_final_cycle(o.Q_DscA - o.Q_AbdA - o.Q_Ren1, t)),
        "Mesenteric artery": np.asarray(_final_cycle(o.Q_AbdA - o.Q_LLimb1 - o.Q_Cel - o.Q_Mes1, t)),
        "Splenic artery": np.asarray(_final_cycle(o.Q_Cel - o.Q_Hep1 - o.Q_Spl1, t)),
    }

    flow_diff_vein = {
        "Inferior VC": np.asarray(_final_cycle(o.Q_LLimb2 + o.Q_Ven_Por + o.Q_Hep2 + o.Q_Ren2 - o.Q_IVC, t)),
        "Superior VC": np.asarray(_final_cycle(o.Q_Fac + o.Q_Cer + o.Q_ULimb - o.Q_SVC, t)),
        "Portal vein": np.asarray(_final_cycle(o.Q_Mes2 + o.Q_Spl2 - o.Q_Ven_Por, t)),
    }

    return {
        "organ": organ,
        "artery": artery,
        "vein": veins,
        "flow_diff_organ": flow_diff_organ,
        "flow_diff_artery": flow_diff_artery,
        "flow_diff_vein": flow_diff_vein,
    }


def plot_cardiac_output_distribution(art: SimulationArtifacts, groups: dict, root_dir: Path) -> None:
    artery_labels = list(groups["artery"].keys())
    vein_labels = list(groups["vein"].keys())
    organ_labels = ["Right lung", "Left lung"]

    artery_means = [float(np.mean(groups["artery"][k])) for k in artery_labels]
    vein_means = [float(np.mean(groups["vein"][k])) for k in vein_labels]
    organ_means = [float(np.mean(groups["organ"][k])) for k in organ_labels]

    labels = artery_labels + vein_labels + organ_labels
    means = artery_means + vein_means + organ_means
    category = ["Artery"] * len(artery_labels) + ["Systemic vessels"] * len(vein_labels) + ["Organ"] * len(organ_labels)

    stroke_volume = float(np.max(_final_cycle(art.output.V_LV, art.tspan)) - np.min(_final_cycle(art.output.V_LV, art.tspan)))
    cardiac_output = stroke_volume * float(art.params.BPM) / 60.0
    co_model = (np.asarray(means) / cardiac_output) * 100.0

    # Physiological ranges used in the manuscript notebook.
    co_phys_min = [100, 100, 100, 87.8, 54.9, 32.9, 17.6, 7.7, 6.6, 4.4, 13.2, 7.4, 17.6, 11.0, 3.3, 8.8, 43.9, 16.5, 19.8, 50, 45]
    co_phys_max = [100, 100, 100, 100, 76.8, 52.8, 28.5, 13.2, 11.0, 8.8, 17.6, 12.8, 24.1, 16.5, 5.5, 11.0, 65.9, 27.4, 27.4, 55, 50]

    df = pd.DataFrame(
        {
            "Compartment": labels,
            "Group": category,
            "% CO (Model)": co_model,
            "% CO (Physiology Min)": co_phys_min,
            "% CO (Physiology Max)": co_phys_max,
        }
    )

    set2 = sns.color_palette("Set2", 8)
    group_color_map = {"Artery": set2[0], "Systemic vessels": set2[1], "Organ": set2[2]}

    def set_alpha(color: tuple[float, float, float], alpha: float = 1.0) -> tuple[float, float, float, float]:
        rgba = list(to_rgba(color))
        rgba[3] = alpha
        return tuple(rgba)

    fig, ax = plt.subplots(figsize=(22, 12), dpi=300)
    df["Compartment"] = pd.Categorical(df["Compartment"], categories=labels, ordered=True)
    df = df.sort_values("Compartment")

    bar_colors = [set_alpha(group_color_map[g]) for g in df["Group"]]
    x_idx = np.arange(len(df))
    ax.bar(x_idx, df["% CO (Model)"].to_numpy(), color=bar_colors, width=0.9, zorder=3)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(df["Compartment"], rotation=45, ha="right")

    for i, (_, row) in enumerate(df.iterrows()):
        x = i
        y_min = row["% CO (Physiology Min)"]
        y_max = row["% CO (Physiology Max)"]
        ax.vlines(x=x, ymin=y_min, ymax=y_max, color="gray", alpha=0.75, linewidth=3.0, zorder=5)
        ax.hlines(y=y_min, xmin=x - 0.18, xmax=x + 0.18, color="gray", alpha=0.9, linewidth=2.0, zorder=5)
        ax.hlines(y=y_max, xmin=x - 0.18, xmax=x + 0.18, color="gray", alpha=0.9, linewidth=2.0, zorder=5)

    for p in ax.patches[: len(df)]:
        h = p.get_height()
        if abs(h) > 1e-2:
            ax.annotate(f"{h:.1f}", (p.get_x() + p.get_width() / 2.0 - 0.15, h + 0.6), ha="right", va="bottom", fontsize=14, rotation=90)

    ax.set_xlabel("Compartment", fontsize=16)
    ax.set_ylabel("Percent Cardiac Output (%)", fontsize=16)
    ax.set_ylim(0, 110)
    ax.tick_params(axis="x", rotation=45)

    legend_patches = [
        Patch(facecolor=set_alpha(set2[0]), label="Artery"),
        Patch(facecolor=set_alpha(set2[1]), label="Systemic vessels"),
        Patch(facecolor=set_alpha(set2[2]), label="Organ"),
        Patch(facecolor="gray", alpha=0.6, label="Physiology range"),
    ]
    ax.legend(handles=legend_patches, title="Group", loc="upper right")

    _save(fig, root_dir / "Figure7.png")


def plot_heart_figure(art: SimulationArtifacts, code_dir: Path, root_dir: Path) -> None:
    base = pd.read_csv(_csv_path(code_dir, "base_sim.csv"))
    t = art.final_times
    o = art.output
    h = art.heart
    n = art.tspan

    ref_max0 = [120, 120, 15, 120, 600, 480, 10, 60, 7, 60, 480, 380]
    ref_max1 = [130, 160, 30, 160, 700, 500, 12, 100, 8, 110, 520, 420]
    ref_min0 = [3, 30, 0, 40, 0, 0, 13, 20, 4, 20, 0, 0]
    ref_min1 = [12, 60, 8, 60, 0, 0, 15, 40, 5, 40, 0, 0]

    base_colors = sns.color_palette("tab10", 12)

    def lighten(c: tuple[float, float, float], alpha: float) -> tuple[float, float, float]:
        return tuple(np.array(c) * (1 - alpha) + np.array([1, 1, 1]) * alpha)

    def darken(c: tuple[float, float, float], alpha: float) -> tuple[float, float, float]:
        return tuple(np.array(c) * (1 - alpha))

    min_band = [lighten(c, 0.5) for c in base_colors]
    max_band = [darken(c, 0.5) for c in base_colors]

    fig, axs = plt.subplots(6, 2, figsize=(17, 21), dpi=300)

    # Row 1
    axs[0, 0].plot(t, _final_cycle(h.P_LV, n), color=base_colors[0], label="$P_{LV}$")
    axs[0, 0].plot(t, base["P_LV"], color=base_colors[0], linestyle="dashed", label="base $P_{LV}$")
    axs[0, 0].fill_between(t, ref_max0[0], ref_max1[0], color=max_band[0], alpha=0.3, label="Max ref")
    axs[0, 0].fill_between(t, ref_min0[0], ref_min1[0], color=min_band[0], alpha=0.3, label="Min ref")
    _styled_axis(axs[0, 0], "Pressure [mmHg]")

    axs[0, 1].plot(t, _final_cycle(o.V_LV, n), color=base_colors[1], label="$V_{LV}$")
    axs[0, 1].plot(t, base["V_LV"], color=base_colors[1], linestyle="dashed", label="base $V_{LV}$")
    axs[0, 1].fill_between(t, ref_max0[1], ref_max1[1], color=max_band[1], alpha=0.3, label="Max ref")
    axs[0, 1].fill_between(t, ref_min0[1], ref_min1[1], color=min_band[1], alpha=0.3, label="Min ref")
    _styled_axis(axs[0, 1], "Volume [mL]")

    # Row 2
    axs[1, 0].plot(t, _final_cycle(h.P_RV, n), color=base_colors[0], label="$P_{RV}$")
    axs[1, 0].plot(t, base["P_RV"], color=base_colors[0], linestyle="dashed", label="base $P_{RV}$")
    axs[1, 0].fill_between(t, ref_max0[2], ref_max1[2], color=max_band[0], alpha=0.3, label="Max ref")
    axs[1, 0].fill_between(t, ref_min0[2], ref_min1[2], color=min_band[0], alpha=0.3, label="Min ref")
    _styled_axis(axs[1, 0], "Pressure [mmHg]")

    axs[1, 1].plot(t, _final_cycle(o.V_RV, n), color=base_colors[1], label="$V_{RV}$")
    axs[1, 1].plot(t, base["V_RV"], color=base_colors[1], linestyle="dashed", label="base $V_{RV}$")
    axs[1, 1].fill_between(t, ref_max0[3], ref_max1[3], color=max_band[1], alpha=0.3, label="Max ref")
    axs[1, 1].fill_between(t, ref_min0[3], ref_min1[3], color=min_band[1], alpha=0.3, label="Min ref")
    _styled_axis(axs[1, 1], "Volume [mL]")

    # Row 3
    axs[2, 0].plot(t, _final_cycle(h.Q_AV, n), color=base_colors[2], label="$Q_{AV}$")
    axs[2, 0].plot(t, base["Q_AV"], color=base_colors[2], linestyle="dashed", label="base $Q_{AV}$")
    axs[2, 0].fill_between(t, ref_max0[10], ref_max1[10], color=max_band[2], alpha=0.3, label="Peak ref")
    _styled_axis(axs[2, 0], "Flow rate [mL/s]")

    axs[2, 1].plot(t, _final_cycle(h.Q_PV, n), color=base_colors[2], label="$Q_{PV}$")
    axs[2, 1].plot(t, base["Q_PV"], color=base_colors[2], linestyle="dashed", label="base $Q_{PV}$")
    axs[2, 1].fill_between(t, ref_max0[11], ref_max1[11], color=max_band[2], alpha=0.3, label="Peak ref")
    _styled_axis(axs[2, 1], "Flow rate [mL/s]")

    # Row 4
    axs[3, 0].plot(t, _final_cycle(h.Q_MV, n), color=base_colors[2], label="$Q_{MV}$")
    axs[3, 0].plot(t, base["Q_MV"], color=base_colors[2], linestyle="dashed", label="base $Q_{MV}$")
    axs[3, 0].fill_between(t, ref_max0[4], ref_max1[4], color=max_band[2], alpha=0.3, label="Peak ref")
    _styled_axis(axs[3, 0], "Flow rate [mL/s]")

    axs[3, 1].plot(t, _final_cycle(h.Q_TV, n), color=base_colors[2], label="$Q_{TV}$")
    axs[3, 1].plot(t, base["Q_TV"], color=base_colors[2], linestyle="dashed", label="base $Q_{TV}$")
    axs[3, 1].fill_between(t, ref_max0[5], ref_max1[5], color=max_band[2], alpha=0.3, label="Peak ref")
    _styled_axis(axs[3, 1], "Flow rate [mL/s]")

    # Row 5
    axs[4, 0].plot(t, _final_cycle(h.P_LA, n), color=base_colors[0], label="$P_{LA}$")
    axs[4, 0].plot(t, base["P_LA"], color=base_colors[0], linestyle="dashed", label="base $P_{LA}$")
    axs[4, 0].fill_between(t, ref_max0[6], ref_max1[6], color=max_band[0], alpha=0.3, label="Max ref")
    axs[4, 0].fill_between(t, ref_min0[6], ref_min1[6], color=min_band[0], alpha=0.3, label="Min ref")
    _styled_axis(axs[4, 0], "Pressure [mmHg]")

    axs[4, 1].plot(t, _final_cycle(o.V_LA, n), color=base_colors[1], label="$V_{LA}$")
    axs[4, 1].plot(t, base["V_LA"], color=base_colors[1], linestyle="dashed", label="base $V_{LA}$")
    axs[4, 1].fill_between(t, ref_max0[7], ref_max1[7], color=max_band[1], alpha=0.3, label="Max ref")
    axs[4, 1].fill_between(t, ref_min0[7], ref_min1[7], color=min_band[1], alpha=0.3, label="Min ref")
    _styled_axis(axs[4, 1], "Volume [mL]")

    # Row 6
    axs[5, 0].plot(t, _final_cycle(h.P_RA, n), color=base_colors[0], label="$P_{RA}$")
    axs[5, 0].plot(t, base["P_RA"], color=base_colors[0], linestyle="dashed", label="base $P_{RA}$")
    axs[5, 0].fill_between(t, ref_max0[8], ref_max1[8], color=max_band[0], alpha=0.3, label="Max ref")
    axs[5, 0].fill_between(t, ref_min0[8], ref_min1[8], color=min_band[0], alpha=0.3, label="Min ref")
    _styled_axis(axs[5, 0], "Pressure [mmHg]")

    axs[5, 1].plot(t, _final_cycle(o.V_RA, n), color=base_colors[1], label="$V_{RA}$")
    axs[5, 1].plot(t, base["V_RA"], color=base_colors[1], linestyle="dashed", label="base $V_{RA}$")
    axs[5, 1].fill_between(t, ref_max0[9], ref_max1[9], color=max_band[1], alpha=0.3, label="Max ref")
    axs[5, 1].fill_between(t, ref_min0[9], ref_min1[9], color=min_band[1], alpha=0.3, label="Min ref")
    _styled_axis(axs[5, 1], "Volume [mL]")

    for ax in axs.flat:
        for line in ax.get_lines():
            line.set_linewidth(2.3)
        ax.legend(loc="upper right")

    _save(fig, root_dir / "Figure4.png")


def plot_main_vessel_pressure_figure(art: SimulationArtifacts, root_dir: Path) -> None:
    t = art.final_times
    o = art.output
    n = art.tspan
    colors = sns.color_palette("tab10", 10)

    def shade(ax: plt.Axes, ylo0: float, ylo1: float, yhi0: float, yhi1: float, color: tuple[float, float, float]) -> None:
        lo = tuple(np.array(color) * 0.5 + np.array([1, 1, 1]) * 0.5)
        hi = tuple(np.array(color) * 0.5)
        ax.fill_between(t, yhi0, yhi1, color=hi, alpha=0.30, label="Max ref")
        ax.fill_between(t, ylo0, ylo1, color=lo, alpha=0.30, label="Min ref")

    fig, axs = plt.subplots(2, 2, figsize=(15, 7), dpi=300)

    axs[0, 0].plot(t, _final_cycle(o.P_AA, n), color=colors[8], label="$P_{AscA}$")
    shade(axs[0, 0], 75, 80, 120, 125, colors[8])
    _styled_axis(axs[0, 0], "Pressure [mmHg]")

    axs[0, 1].plot(t, _final_cycle(o.P_Ar_Pul, n), color=colors[6], label="$P_{Ar}^{Pul}$")
    shade(axs[0, 1], 8, 10, 25, 30, colors[6])
    _styled_axis(axs[0, 1], "Pressure [mmHg]")

    axs[1, 0].plot(t, _final_cycle(o.P_Cer, n), color=colors[3], label="$P_{Cer}$")
    shade(axs[1, 0], 70, 80, 110, 120, colors[3])
    _styled_axis(axs[1, 0], "Pressure [mmHg]")

    axs[1, 1].plot(t, _final_cycle(o.P_Ren, n), color=colors[5], label="$P_{Ren}$")
    shade(axs[1, 1], 75, 80, 100, 110, colors[5])
    _styled_axis(axs[1, 1], "Pressure [mmHg]")

    for ax in axs.flat:
        for line in ax.get_lines():
            line.set_linewidth(2.3)
        ax.legend(loc="upper right")

    _save(fig, root_dir / "Figure5.png")


def plot_pulmonary_waveform(art: SimulationArtifacts, supp_dir: Path) -> None:
    t = art.final_times
    o = art.output
    n = art.tspan
    colors = sns.color_palette("tab10", 10)

    fig, axs = plt.subplots(2, 2, figsize=(16, 7), dpi=300)

    axs[0, 0].plot(t, _final_cycle(o.P_Ven_Pul, n), color=colors[6], label="$P_{Ven}^{Pul}$")
    axs[0, 0].plot(t, _final_cycle(o.P_Ar_Pul, n), color=colors[7], label="$P_{Ar}^{Pul}$")
    _styled_axis(axs[0, 0], "Pressure [mmHg]")

    axs[0, 1].plot(t, _final_cycle(o.Q_Ven_Pul, n), color=colors[6], label="$Q_{Ven}^{Pul}$")
    axs[0, 1].plot(t, _final_cycle(o.Q_Ar_Pul, n), color=colors[7], label="$Q_{Ar}^{Pul}$")
    _styled_axis(axs[0, 1], "Flow rate [mL/s]")

    axs[1, 0].plot(t, _final_cycle(o.P_Lung, n), color=colors[0], label="$P_{Lung}$")
    axs[1, 0].plot(t, _final_cycle(o.P_RL, n), color=colors[1], label="$P_{RL}$")
    axs[1, 0].plot(t, _final_cycle(o.P_LL, n), color=colors[2], label="$P_{LL}$")
    _styled_axis(axs[1, 0], "Pressure [mmHg]")

    axs[1, 1].plot(t, _final_cycle(o.Q_LL1, n), color=colors[2], label="$Q^{LPul}_{Ar}$")
    axs[1, 1].plot(t, _final_cycle(o.Q_RL1, n), color=colors[1], label="$Q^{RPul}_{Ar}$")
    axs[1, 1].plot(t, _final_cycle(o.Q_LL2, n), color=colors[3], label="$Q_{LL}$")
    axs[1, 1].plot(t, _final_cycle(o.Q_RL2, n), color=colors[4], label="$Q_{RL}$")
    _styled_axis(axs[1, 1], "Flow rate [mL/s]")

    for ax in axs.flat:
        ax.legend(loc="upper right")

    _save(fig, supp_dir / "pulmonary_waveform.png")


def plot_major_arteries_waveform(art: SimulationArtifacts, supp_dir: Path) -> None:
    t = art.final_times
    o = art.output
    n = art.tspan
    colors = sns.color_palette("tab10", 10)

    fig, axs = plt.subplots(2, 2, figsize=(16, 7), dpi=300)

    axs[0, 0].plot(t, _final_cycle(o.P_AA, n), color=colors[0], label="$P_{AA}$")
    axs[0, 0].plot(t, _final_cycle(o.P_AArc, n), color=colors[1], label="$P_{AArc}$")
    axs[0, 0].plot(t, _final_cycle(o.P_CCar, n), color=colors[2], label="$P_{CCar}$")
    axs[0, 0].plot(t, _final_cycle(o.P_DscA, n), color=colors[3], label="$P_{DscA}$")
    _styled_axis(axs[0, 0], "Pressure [mmHg]")

    axs[0, 1].plot(t, _final_cycle(o.Q_AA, n), color=colors[0], label="$Q_{AA}$")
    axs[0, 1].plot(t, _final_cycle(o.Q_CCar, n), color=colors[2], label="$Q_{CCar}$")
    axs[0, 1].plot(t, _final_cycle(o.Q_Sub, n), color=colors[4], label="$Q_{Sub}$")
    axs[0, 1].plot(t, _final_cycle(o.Q_DscA, n), color=colors[3], label="$Q_{DscA}$")
    _styled_axis(axs[0, 1], "Flow rate [mL/s]")

    axs[1, 0].plot(t, _final_cycle(o.P_DscA, n), color=colors[1], label="$P_{DscA}$")
    axs[1, 0].plot(t, _final_cycle(o.P_AbdA, n), color=colors[2], label="$P_{AbdA}$")
    axs[1, 0].plot(t, _final_cycle(o.P_Cel, n), color=colors[3], label="$P_{Cel}$")
    axs[1, 0].plot(t, _final_cycle(o.P_Ren, n), color=colors[4], label="$P_{Ren}$")
    _styled_axis(axs[1, 0], "Pressure [mmHg]")

    axs[1, 1].plot(t, _final_cycle(o.Q_DscA, n), color=colors[1], label="$Q_{DscA}$")
    axs[1, 1].plot(t, _final_cycle(o.Q_AbdA, n), color=colors[2], label="$Q_{AbdA}$")
    axs[1, 1].plot(t, _final_cycle(o.Q_Cel, n), color=colors[3], label="$Q_{Cel}$")
    axs[1, 1].plot(t, _final_cycle(o.Q_Ren1, n), color=colors[4], label="$Q^{Ren}_{Ar}$")
    axs[1, 1].plot(t, _final_cycle(o.Q_Ren2, n), color=colors[5], label="$Q_{Ren}$")
    _styled_axis(axs[1, 1], "Flow rate [mL/s]")

    for ax in axs.flat:
        ax.legend(loc="upper right")

    _save(fig, supp_dir / "major_arteries_waveform.png")


def plot_upperbody_waveform(art: SimulationArtifacts, supp_dir: Path) -> None:
    t = art.final_times
    o = art.output
    n = art.tspan
    colors = sns.color_palette("tab10", 10)

    fig, axs = plt.subplots(2, 2, figsize=(16, 7), dpi=300)

    axs[0, 0].plot(t, _final_cycle(o.P_Cer, n), color=colors[0], label="$P_{Cer}$")
    axs[0, 0].plot(t, _final_cycle(o.P_Fac, n), color=colors[1], label="$P_{Fac}$")
    _styled_axis(axs[0, 0], "Pressure [mmHg]")

    axs[0, 1].plot(t, _final_cycle(o.Q_ECar, n), color=colors[2], label="$Q_{ECar}$")
    axs[0, 1].plot(t, _final_cycle(o.Q_ICar, n), color=colors[3], label="$Q_{ICar}$")
    axs[0, 1].plot(t, _final_cycle(o.Q_Cer, n), color=colors[0], label="$Q_{Cer}$")
    axs[0, 1].plot(t, _final_cycle(o.Q_Fac, n), color=colors[1], label="$Q_{Fac}$")
    _styled_axis(axs[0, 1], "Flow rate [mL/s]")

    axs[1, 0].plot(t, _final_cycle(o.P_ULimb, n), color=colors[2], label="$P_{ULimb}$")
    axs[1, 0].plot(t, _final_cycle(o.P_SVC, n), color=colors[3], label="$P_{SVC}$")
    _styled_axis(axs[1, 0], "Pressure [mmHg]")

    axs[1, 1].plot(t, _final_cycle(o.Q_ULimb, n), color=colors[2], label="$Q_{ULimb}$")
    axs[1, 1].plot(t, _final_cycle(o.Q_SVC, n), color=colors[3], label="$Q_{SVC}$")
    _styled_axis(axs[1, 1], "Flow rate [mL/s]")

    for ax in axs.flat:
        ax.legend(loc="upper right")

    _save(fig, supp_dir / "upperbody_waveform.png")


def plot_lowerbody_waveform(art: SimulationArtifacts, supp_dir: Path) -> None:
    t = art.final_times
    o = art.output
    n = art.tspan
    colors = sns.color_palette("tab10", 10)

    fig, axs = plt.subplots(3, 2, figsize=(16, 10), dpi=300)

    axs[0, 0].plot(t, _final_cycle(o.P_Spl, n), color=colors[2], label="$P_{Spl}$")
    axs[0, 0].plot(t, _final_cycle(o.P_Mes, n), color=colors[3], label="$P_{Mes}$")
    _styled_axis(axs[0, 0], "Pressure [mmHg]")

    axs[0, 1].plot(t, _final_cycle(o.Q_Spl2, n), color=colors[2], label="$Q_{Spl}$")
    axs[0, 1].plot(t, _final_cycle(o.Q_Spl1, n), color=colors[4], label="$Q^{Spl}_{Ar}$")
    axs[0, 1].plot(t, _final_cycle(o.Q_Mes1, n), color=colors[3], label="$Q^{Mes}_{Ar}$")
    axs[0, 1].plot(t, _final_cycle(o.Q_Mes2, n), color=colors[5], label="$Q_{Mes}$")
    _styled_axis(axs[0, 1], "Flow rate [mL/s]")

    axs[1, 0].plot(t, _final_cycle(o.P_Hep, n), color=colors[1], label="$P_{Hep}$")
    axs[1, 0].plot(t, _final_cycle(o.P_Ven_Por, n), color=colors[2], label="$P_{Ven}^{Por}$")
    _styled_axis(axs[1, 0], "Pressure [mmHg]")

    axs[1, 1].plot(t, _final_cycle(o.Q_Ven_Por, n), color=colors[2], label="$Q_{Ven}^{Por}$")
    axs[1, 1].plot(t, _final_cycle(o.Q_Hep2, n), color=colors[1], label="$Q_{Hep2}$")
    axs[1, 1].plot(t, _final_cycle(o.Q_Hep1, n), color=colors[5], label="$Q^{Hep}_{Ar}$")
    _styled_axis(axs[1, 1], "Flow rate [mL/s]")

    axs[2, 0].plot(t, _final_cycle(o.P_LLimb, n), color=colors[3], label="$P_{LLimb}$")
    axs[2, 0].plot(t, _final_cycle(o.P_IVC, n), color=colors[4], label="$P_{IVC}$")
    _styled_axis(axs[2, 0], "Pressure [mmHg]")

    axs[2, 1].plot(t, _final_cycle(o.Q_LLimb1, n), color=colors[3], label="$Q_{Iliac}$")
    axs[2, 1].plot(t, _final_cycle(o.Q_LLimb2, n), color=colors[5], label="$Q_{LLimb}$")
    axs[2, 1].plot(t, _final_cycle(o.Q_IVC, n), color=colors[4], label="$Q_{IVC}$")
    _styled_axis(axs[2, 1], "Flow rate [mL/s]")

    for ax in axs.flat:
        ax.legend(loc="upper right")

    _save(fig, supp_dir / "lowerbody_waveform.png")


def plot_organ_flows(art: SimulationArtifacts, supp_dir: Path) -> None:
    t = art.final_times
    o = art.output
    n = art.tspan
    colors = sns.color_palette("tab10", 10)

    fig, axs = plt.subplots(1, 2, figsize=(16, 4), dpi=300)

    axs[0].plot(t, _final_cycle(o.Q_Cer, n), color=colors[0], label="$Q_{Cer}$")
    axs[0].plot(t, _final_cycle(o.Q_Fac, n), color=colors[1], label="$Q_{Fac}$")
    axs[0].plot(t, _final_cycle(o.Q_ULimb, n), color=colors[2], label="$Q_{ULimb}$")
    _styled_axis(axs[0], "Flow rate [mL/s]")

    axs[1].plot(t, _final_cycle(o.Q_Ren2, n), color=colors[3], label="$Q_{Ren}$")
    axs[1].plot(t, _final_cycle(o.Q_Hep2, n), color=colors[4], label="$Q_{Hep}$")
    axs[1].plot(t, _final_cycle(o.Q_Spl2, n), color=colors[5], label="$Q_{Spl}$")
    axs[1].plot(t, _final_cycle(o.Q_Mes2, n), color=colors[6], label="$Q_{Mes}$")
    axs[1].plot(t, _final_cycle(o.Q_LLimb2, n), color=colors[7], label="$Q_{LLimb}$")
    _styled_axis(axs[1], "Flow rate [mL/s]")

    for ax in axs:
        ax.legend(loc="upper right")

    _save(fig, supp_dir / "Organ_flows.png")


def plot_vein_flows(art: SimulationArtifacts, supp_dir: Path) -> None:
    t = art.final_times
    o = art.output
    n = art.tspan
    colors = sns.color_palette("tab10", 10)

    fig, axs = plt.subplots(1, 2, figsize=(16, 4), dpi=300)

    axs[0].plot(t, _final_cycle(o.P_SVC, n), color=colors[0], label="$P_{SVC}$")
    axs[0].plot(t, _final_cycle(o.P_IVC, n), color=colors[1], label="$P_{IVC}$")
    axs[0].plot(t, _final_cycle(o.P_Ven_Por, n), color=colors[2], label="$P_{Ven}^{Por}$")
    _styled_axis(axs[0], "Pressure [mmHg]")

    axs[1].plot(t, _final_cycle(o.Q_SVC, n), color=colors[0], label="$Q_{SVC}$")
    axs[1].plot(t, _final_cycle(o.Q_IVC, n), color=colors[1], label="$Q_{IVC}$")
    axs[1].plot(t, _final_cycle(o.Q_Ven_Por, n), color=colors[2], label="$Q_{Ven}^{Por}$")
    _styled_axis(axs[1], "Flow rate [mL/s]")

    for ax in axs:
        ax.legend(loc="upper right")

    _save(fig, supp_dir / "vein_flows.png")


def _load_cycle_file(path: Path) -> list[float]:
    return pd.read_csv(path, header=None).squeeze().tolist()


def plot_flow_convergence_metrics(data_dir: Path, groups: dict, cycles: Iterable[int], supp_dir: Path) -> None:
    mean_organ_flow = [float(np.mean(v)) for v in groups["organ"].values()]
    mean_artery_flow = [float(np.mean(v)) for v in groups["artery"].values()]
    mean_vein_flow = [float(np.mean(v)) for v in groups["vein"].values()]

    rows: list[dict] = []
    used_cycles: list[int] = []

    for n_cycles in cycles:
        f_org = data_dir / f"mean_organ_flow_diff_{n_cycles}.csv"
        f_art = data_dir / f"mean_artery_flow_diff_{n_cycles}.csv"
        f_ven = data_dir / f"mean_vein_flow_diff_{n_cycles}.csv"
        if not (f_org.exists() and f_art.exists() and f_ven.exists()):
            continue

        used_cycles.append(n_cycles)
        organ_diff = _load_cycle_file(f_org)
        artery_diff = _load_cycle_file(f_art)
        vein_diff = _load_cycle_file(f_ven)

        mean_abs_org = (np.sum(np.abs(organ_diff)) / np.sum(mean_organ_flow)) * 100.0
        mean_abs_art = (np.sum(np.abs(artery_diff)) / np.sum(mean_artery_flow)) * 100.0
        mean_abs_ven = (np.sum(np.abs(vein_diff)) / np.sum(mean_vein_flow)) * 100.0

        rmse_org = (np.sqrt(np.mean(np.square(organ_diff))) / np.mean(mean_organ_flow)) * 100.0
        rmse_art = (np.sqrt(np.mean(np.square(artery_diff))) / np.mean(mean_artery_flow)) * 100.0
        rmse_ven = (np.sqrt(np.mean(np.square(vein_diff))) / np.mean(mean_vein_flow)) * 100.0

        rows.extend(
            [
                {"Group": "Organ", "Metric": "Mean Absolute Flow Diff [%]", "Value": mean_abs_org, "N_CYCLES": n_cycles},
                {"Group": "Artery", "Metric": "Mean Absolute Flow Diff [%]", "Value": mean_abs_art, "N_CYCLES": n_cycles},
                {"Group": "Systemic vessels", "Metric": "Mean Absolute Flow Diff [%]", "Value": mean_abs_ven, "N_CYCLES": n_cycles},
                {"Group": "Organ", "Metric": "RMSE [%]", "Value": rmse_org, "N_CYCLES": n_cycles},
                {"Group": "Artery", "Metric": "RMSE [%]", "Value": rmse_art, "N_CYCLES": n_cycles},
                {"Group": "Systemic vessels", "Metric": "RMSE [%]", "Value": rmse_ven, "N_CYCLES": n_cycles},
            ]
        )

    if not rows:
        print("[WARN] Skipping Fig11: required mean_flow_diff CSVs were not found.")
        return

    df_all = pd.DataFrame(rows)
    df_all["N_CYCLES"] = df_all["N_CYCLES"].astype(str)

    g = sns.catplot(
        data=df_all,
        kind="bar",
        x="N_CYCLES",
        y="Value",
        hue="Metric",
        col="Group",
        palette=["#7fc97f", "#beaed4"],
        height=4.8,
        aspect=1.15,
        sharey=False,
    )
    g.set_axis_labels("Cardiac Cycles", "Flow Difference [%]")
    g.set_titles("{col_name}")
    g._legend.set_title("")
    g._legend.set_bbox_to_anchor((0.5, 1.05))

    out = supp_dir / "Fig11.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(g.fig)


def save_simulation_tables(art: SimulationArtifacts, analysis_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = art.tspan
    o = art.output
    h = art.heart

    sim_vars = {
        "V_LA": _final_cycle(o.V_LA, t),
        "V_LV": _final_cycle(o.V_LV, t),
        "V_RA": _final_cycle(o.V_RA, t),
        "V_RV": _final_cycle(o.V_RV, t),
        "P_Ar_Pul": _final_cycle(o.P_Ar_Pul, t),
        "P_Ven_Pul": _final_cycle(o.P_Ven_Pul, t),
        "Q_Ar_Pul": _final_cycle(o.Q_Ar_Pul, t),
        "Q_Ven_Pul": _final_cycle(o.Q_Ven_Pul, t),
        "P_Lung": _final_cycle(o.P_Lung, t),
        "Q_RL1": _final_cycle(o.Q_RL1, t),
        "Q_LL1": _final_cycle(o.Q_LL1, t),
        "P_LL": _final_cycle(o.P_LL, t),
        "P_RL": _final_cycle(o.P_RL, t),
        "Q_LL2": _final_cycle(o.Q_LL2, t),
        "Q_RL2": _final_cycle(o.Q_RL2, t),
        "P_Ren": _final_cycle(o.P_Ren, t),
        "Q_Ren2": _final_cycle(o.Q_Ren2, t),
        "P_ULimb": _final_cycle(o.P_ULimb, t),
        "Q_ULimb": _final_cycle(o.Q_ULimb, t),
        "P_Cer": _final_cycle(o.P_Cer, t),
        "Q_Cer": _final_cycle(o.Q_Cer, t),
        "P_SVC": _final_cycle(o.P_SVC, t),
        "Q_SVC": _final_cycle(o.Q_SVC, t),
        "P_AA": _final_cycle(o.P_AA, t),
        "Q_AA": _final_cycle(o.Q_AA, t),
        "P_AArc": _final_cycle(o.P_AArc, t),
        "Q_Sub": _final_cycle(o.Q_Sub, t),
        "Q_CCar": _final_cycle(o.Q_CCar, t),
        "P_DscA": _final_cycle(o.P_DscA, t),
        "P_AbdA": _final_cycle(o.P_AbdA, t),
        "P_Cel": _final_cycle(o.P_Cel, t),
        "Q_Cel": _final_cycle(o.Q_Cel, t),
        "Q_DscA": _final_cycle(o.Q_DscA, t),
        "Q_Ren1": _final_cycle(o.Q_Ren1, t),
        "Q_AbdA": _final_cycle(o.Q_AbdA, t),
        "Q_Iliac": _final_cycle(o.Q_LLimb1, t),
        "Q_Spl2": _final_cycle(o.Q_Spl2, t),
        "P_Spl": _final_cycle(o.P_Spl, t),
        "Q_Spl1": _final_cycle(o.Q_Spl1, t),
        "P_Hep": _final_cycle(o.P_Hep, t),
        "Q_Hep2": _final_cycle(o.Q_Hep2, t),
        "Q_Hep1": _final_cycle(o.Q_Hep1, t),
        "P_Ven_Por": _final_cycle(o.P_Ven_Por, t),
        "Q_Ven_Por": _final_cycle(o.Q_Ven_Por, t),
        "P_Mes": _final_cycle(o.P_Mes, t),
        "Q_Mes2": _final_cycle(o.Q_Mes2, t),
        "Q_Mes1": _final_cycle(o.Q_Mes1, t),
        "P_LLimb": _final_cycle(o.P_LLimb, t),
        "Q_LLimb": _final_cycle(o.Q_LLimb2, t),
        "P_IVC": _final_cycle(o.P_IVC, t),
        "Q_IVC": _final_cycle(o.Q_IVC, t),
        "P_CCar": _final_cycle(o.P_CCar, t),
        "Q_ECar": _final_cycle(o.Q_ECar, t),
        "Q_ICar": _final_cycle(o.Q_ICar, t),
        "P_Fac": _final_cycle(o.P_Fac, t),
        "Q_Fac": _final_cycle(o.Q_Fac, t),
        "P_LV": _final_cycle(h.P_LV, t),
        "P_LA": _final_cycle(h.P_LA, t),
        "P_RV": _final_cycle(h.P_RV, t),
        "P_RA": _final_cycle(h.P_RA, t),
        "Q_MV": _final_cycle(h.Q_MV, t),
        "Q_AV": _final_cycle(h.Q_AV, t),
        "Q_TV": _final_cycle(h.Q_TV, t),
        "Q_PV": _final_cycle(h.Q_PV, t),
    }

    min_df = pd.DataFrame(
        {
            "Variable": list(sim_vars.keys()),
            "Min Value": [np.round(np.min(v), 0) for v in sim_vars.values()],
            "Max Value": [np.round(np.max(v), 0) for v in sim_vars.values()],
        }
    )
    mean_df = pd.DataFrame(
        {
            "Variable": list(sim_vars.keys()),
            "Mean Value": [np.round(np.mean(v), 0) for v in sim_vars.values()],
        }
    )

    analysis_dir.mkdir(parents=True, exist_ok=True)
    min_df.to_csv(analysis_dir / "min_max_values.csv", index=False)
    mean_df.to_csv(analysis_dir / "mean_values.csv", index=False)

    return min_df, mean_df


def plot_dumbbell(min_df: pd.DataFrame, art: SimulationArtifacts, supp_dir: Path) -> None:
    # Base model and references from notebook.
    variables = [
        "${V}_{\\mathrm{LA}}$",
        "${V}_{\\mathrm{LV}}$",
        "${V}_{\\mathrm{RA}}$",
        "${V}_{\\mathrm{RV}}$",
        "${P}_{\\mathrm{LA}}$",
        "${P}_{\\mathrm{LV}}$",
        "${P}_{\\mathrm{RA}}$",
        "${P}_{\\mathrm{RV}}$",
        "${Q}_{\\mathrm{Ven}}^{\\mathrm{Pul}}$",
        "${Q}^{\\mathrm{Pul}}_{\\mathrm{Ar}}$",
        "${P}^{\\mathrm{Pul}}_{\\mathrm{Ven}}$",
        "${P}^{\\mathrm{Pul}}_{\\mathrm{Ar}}$",
        "${Q}_{\\mathrm{MV}}$",
        "${Q}_{\\mathrm{TV}}$",
        "${Q}_{\\mathrm{AV}}$",
        "${Q}_{\\mathrm{PV}}$",
    ]

    base_min = [108, 84, 69, 73, 20, 16, 5, 4, 16, 39, 26, 28, 0, 0, 0, 0]
    base_max = [26, 152, 9, 40, 151, 170, 116, 160, 221, 243, 28, 35, 704, 394, 1429, 1167]
    ref_max = [90, 110, 100, 120, 12, 110, 5, 30, 200, 250, 10, 19, 400, 400, 600, 600]
    ref_min = [30, 60, 40, 60, 2, 10, 2, 5, 0, 0, 10, 8, 0, 0, 0, 0]

    lookup_min = dict(zip(min_df["Variable"], min_df["Min Value"]))
    lookup_max = dict(zip(min_df["Variable"], min_df["Max Value"]))

    # Same ordering used in notebook.
    comp_max = [
        lookup_max["V_LA"],
        lookup_max["V_LV"],
        lookup_max["V_RA"],
        lookup_max["V_RV"],
        lookup_max["P_LA"],
        lookup_max["P_LV"],
        lookup_max["P_RA"],
        lookup_max["P_RV"],
        lookup_max["Q_Ven_Pul"],
        lookup_max["Q_Ar_Pul"],
        lookup_max["P_Ven_Pul"],
        lookup_max["P_Ar_Pul"],
        lookup_max["Q_MV"],
        lookup_max["Q_TV"],
        lookup_max["Q_AV"],
        lookup_max["Q_PV"],
    ]
    comp_min = [
        lookup_min["V_LA"],
        lookup_min["V_LV"],
        lookup_min["V_RA"],
        lookup_min["V_RV"],
        lookup_min["P_LA"],
        lookup_min["P_LV"],
        lookup_min["P_RA"],
        lookup_min["P_RV"],
        lookup_min["Q_Ven_Pul"],
        lookup_min["Q_Ar_Pul"],
        lookup_min["P_Ven_Pul"],
        lookup_min["P_Ar_Pul"],
        lookup_min["Q_MV"],
        lookup_min["Q_TV"],
        lookup_min["Q_AV"],
        lookup_min["Q_PV"],
    ]

    color_base = "#d8b365"
    color_comp = "#5ab4ac"

    fig = plt.figure(figsize=(10, 8), dpi=300)
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.3, hspace=0.4)

    y_ticks = np.arange(len(variables)) * 2
    offset = 0.3

    def panel(ax: plt.Axes, start: int, end: int, xlim: tuple[float, float]) -> None:
        for i in range(start, end):
            y = y_ticks[i]
            ax.plot([base_min[i], base_max[i]], [y - offset, y - offset], color=color_base, linewidth=2, label="$Base$ model" if i == start else None)
            ax.plot([comp_min[i], comp_max[i]], [y + offset, y + offset], color=color_comp, linewidth=2, label="$Comprehensive$ model" if i == start else None)
            ax.plot([ref_min[i], ref_max[i]], [y, y], color="gray", linewidth=1.2, linestyle="--", label="Reference" if i == start else None)
            ax.scatter([base_min[i], base_max[i]], [y - offset, y - offset], color=color_base, s=24)
            ax.scatter([comp_min[i], comp_max[i]], [y + offset, y + offset], color=color_comp, s=24)
        ax.set_yticks(y_ticks[start:end])
        ax.set_yticklabels(variables[start:end], fontsize=10)
        ax.set_xlim(*xlim)
        ax.grid(True, linestyle="--", alpha=0.3)

    ax1 = fig.add_subplot(gs[0:2, 0:2])
    panel(ax1, 6, 12, (-10, 265))

    ax2 = fig.add_subplot(gs[0:2, 2:4])
    panel(ax2, 0, 6, (-10, 180))

    ax3 = fig.add_subplot(gs[2:4, 0:4])
    panel(ax3, 12, 16, (-10, 1450))
    ax3.legend(fontsize=10, loc="lower right")
    ax3.text(-120, y_ticks[13], "Variables", rotation=90, fontsize=10)
    ax3.text(220, y_ticks[12] - 1, "Range of flow Q [mL/s], pressure P [mmHg], and volume V [mL]", fontsize=10)

    _save(fig, supp_dir / "dumbbell_plot_physical_vals.png")


def check_unreproducible_assets(root_dir: Path, supp_dir: Path) -> None:
    missing = []
    # Sensitivity analysis source tables are not present in this repository.
    for path in [
        root_dir / "Figure6.png",
        supp_dir / "Fig7.png",
        supp_dir / "Top_5_ST_Sobol_Indices_base.png",
        supp_dir / "global_ST_basemodel_heatmap.png",
    ]:
        if not path.exists():
            missing.append(str(path))

    if missing:
        print("[WARN] Missing sensitivity-figure assets:")
        for item in missing:
            print(f"  - {item}")
        print("[WARN] Provide Sobol input tables or source figures to fully regenerate sensitivity plots.")


def _load_namedtuple_npz(path: Path, field_names: tuple[str, ...], constructor):
    data = np.load(path)
    return constructor(**{name: data[name] for name in field_names})


def _discover_convergence_cycles(convergence_dir: Path) -> list[int]:
    cycles: list[int] = []
    for path in convergence_dir.glob("mean_organ_flow_diff_*.csv"):
        suffix = path.stem.replace("mean_organ_flow_diff_", "")
        if suffix.isdigit():
            cycles.append(int(suffix))
    return sorted(set(cycles))


def _load_artifacts(model_output_dir: Path) -> tuple[SimulationArtifacts, dict]:
    metadata = _load_json(model_output_dir / "metadata.json")
    params = Params.from_dict(_load_json(model_output_dir / "params_bsln.json"))
    init_state = State.from_dict(_load_json(model_output_dir / "state_bsln.json"))

    output = _load_namedtuple_npz(model_output_dir / "state.npz", State._fields, State)
    heart = _load_namedtuple_npz(
        model_output_dir / "heart.npz",
        HeartChamberPressureFlowRate._fields,
        HeartChamberPressureFlowRate,
    )

    times = np.load(model_output_dir / "times.npz")
    save_times = np.asarray(times["save_times"])
    final_times = np.asarray(times["final_times"])
    tspan = int(metadata.get("tspan", len(final_times)))

    artifacts = SimulationArtifacts(
        params=params,
        init_state=init_state,
        output=output,
        heart=heart,
        save_times=save_times,
        final_times=final_times,
        tspan=tspan,
    )
    return artifacts, metadata


def run_post_analysis(
    convergence_cycles: list[int] | None = None,
    model_output_dir: Path = MODEL_OUTPUT_DIR,
    figures_dir: Path = FIGURES_DIR,
    analysis_dir: Path = ANALYSIS_DIR,
    run_sobol: bool = False,
    sobol_samples: int = 64,
    sobol_n_cycles: int = 40,
    sobol_tspan: int = 400,
    sobol_variation: float = 0.10,
    sobol_seed: int = 42,
    sobol_threshold: float = 0.20,
) -> dict:
    ensure_pipeline_dirs()
    figures_main_dir = figures_dir / "main"
    figures_supp_dir = figures_dir / "supp"
    figures_main_dir.mkdir(parents=True, exist_ok=True)
    figures_supp_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()
    artifacts, metadata = _load_artifacts(model_output_dir)
    groups = extract_flow_groups(artifacts)

    plot_heart_figure(artifacts, CODE_DIR, figures_main_dir)
    plot_main_vessel_pressure_figure(artifacts, figures_main_dir)
    plot_cardiac_output_distribution(artifacts, groups, figures_main_dir)

    plot_pv_loops(artifacts, CODE_DIR, figures_supp_dir)
    plot_volume_conservation(artifacts, figures_supp_dir)
    plot_pulmonary_waveform(artifacts, figures_supp_dir)
    plot_major_arteries_waveform(artifacts, figures_supp_dir)
    plot_upperbody_waveform(artifacts, figures_supp_dir)
    plot_lowerbody_waveform(artifacts, figures_supp_dir)
    plot_organ_flows(artifacts, figures_supp_dir)
    plot_vein_flows(artifacts, figures_supp_dir)

    convergence_dir = model_output_dir / "convergence"
    if convergence_cycles is None:
        meta_cycles = metadata.get("convergence_cycles", [])
        convergence_cycles = [int(c) for c in meta_cycles] if meta_cycles else _discover_convergence_cycles(convergence_dir)
    if convergence_cycles:
        plot_flow_convergence_metrics(convergence_dir, groups, convergence_cycles, figures_supp_dir)

    min_df, _ = save_simulation_tables(artifacts, analysis_dir)
    plot_dumbbell(min_df, artifacts, figures_supp_dir)

    sobol_generated = False
    sobol_error = ""
    if run_sobol:
        try:
            from sobol_analysis import SobolConfig, generate_sobol_figures

            print("[INFO] Running Sobol sensitivity analysis...")
            sobol_config = SobolConfig(
                n_samples=int(sobol_samples),
                n_cycles=int(sobol_n_cycles),
                tspan=int(sobol_tspan),
                variation=float(sobol_variation),
                seed=int(sobol_seed),
                threshold=float(sobol_threshold),
            )
            sobol_meta = generate_sobol_figures(
                init_state=artifacts.init_state,
                base_params=artifacts.params,
                figures_main_dir=figures_main_dir,
                figures_supp_dir=figures_supp_dir,
                analysis_dir=analysis_dir,
                config=sobol_config,
            )
            sobol_generated = True
            print(
                "[DONE] Sobol figures generated "
                f"(runs={sobol_meta.get('sample_count')}, failed_runs={sobol_meta.get('failed_runs')})."
            )
        except Exception as exc:
            sobol_error = str(exc)
            print(f"[WARN] Sobol sensitivity analysis failed: {exc}")

    sobol_expected = [
        figures_main_dir / "Figure6.png",
        figures_supp_dir / "Fig7.png",
        figures_supp_dir / "Top_5_ST_Sobol_Indices_base.png",
        figures_supp_dir / "global_ST_basemodel_heatmap.png",
    ]
    sobol_missing = any(not p.exists() for p in sobol_expected)

    return {
        "figures_dir": str(figures_dir),
        "analysis_dir": str(analysis_dir),
        "sobol_missing": sobol_missing,
        "sobol_generated": sobol_generated,
        "sobol_error": sobol_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper figures/tables from output/model artifacts.")
    parser.add_argument(
        "--convergence-cycles",
        type=str,
        default="",
        help="Optional comma-separated cycles for Fig11. If omitted, metadata/discovered cycles are used.",
    )
    parser.add_argument("--run-sobol", action="store_true", help="Compute Sobol ST indices and regenerate sensitivity figures.")
    parser.add_argument("--sobol-samples", type=int, default=64, help="Sobol base sample size N (total runs: N*(D+2)).")
    parser.add_argument("--sobol-n-cycles", type=int, default=40, help="Simulation cycles per Sobol sample.")
    parser.add_argument("--sobol-tspan", type=int, default=400, help="Time samples in final cycle per Sobol run.")
    parser.add_argument("--sobol-variation", type=float, default=0.10, help="Uniform perturbation fraction around nominal parameters.")
    parser.add_argument("--sobol-seed", type=int, default=42, help="Sobol sampler random seed.")
    parser.add_argument("--sobol-threshold", type=float, default=0.20, help="Heatmap display threshold for ST values.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cycles = parse_cycle_list(args.convergence_cycles) if args.convergence_cycles else None
    result = run_post_analysis(
        convergence_cycles=cycles,
        run_sobol=args.run_sobol,
        sobol_samples=args.sobol_samples,
        sobol_n_cycles=args.sobol_n_cycles,
        sobol_tspan=args.sobol_tspan,
        sobol_variation=args.sobol_variation,
        sobol_seed=args.sobol_seed,
        sobol_threshold=args.sobol_threshold,
    )
    if result.get("sobol_missing"):
        print("[WARN] Sobol sensitivity plots are missing. Re-run with --run-sobol to regenerate.")
    if result.get("sobol_error"):
        print(f"[WARN] Sobol generation error: {result['sobol_error']}")
    print(f"[DONE] Figures written to: {result['figures_dir']}")
    print(f"[DONE] Analysis written to: {result['analysis_dir']}")


if __name__ == "__main__":
    main()
