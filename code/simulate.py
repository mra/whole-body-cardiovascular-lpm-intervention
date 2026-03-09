#!/usr/bin/env python3
"""Model simulation stage that writes reproducible artifacts to output/model."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

from common import (
    MODEL_OUTPUT_DIR,
    PARAMETERS_DIR,
    ensure_pipeline_dirs,
    load_baseline_inputs,
    parse_cycle_list,
    utc_now_iso,
    write_json,
)
from data_structures import HeartChamberPressureFlowRate, Params, State
from dynamics import cardiovascular_model, get_heart_chamber_pressures_and_flow_rates


jax.config.update("jax_enable_x64", True)


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


def _run_state_output(
    init_state: State,
    params: Params,
    n_cycles: int,
    tspan: int,
    last_cycles: int,
) -> tuple[State, HeartChamberPressureFlowRate, np.ndarray, np.ndarray]:
    cycle_period = 60.0 / float(params.BPM)
    all_times = jnp.linspace(0.0, n_cycles * cycle_period, n_cycles * tspan)
    save_times = all_times[-last_cycles * tspan :]

    output = _solve_model(init_state, params, float(all_times[-1]), save_times)
    heart = get_heart_chamber_pressures_and_flow_rates(save_times, output, params)

    save_times_np = np.asarray(save_times)
    final_times = save_times_np[-tspan:] - save_times_np[-tspan]
    return output, heart, save_times_np - save_times_np[0], final_times


def _namedtuple_npz_dict(obj: Any) -> dict[str, np.ndarray]:
    return {name: np.asarray(getattr(obj, name)) for name in obj._fields}


def _mean_flow_diffs_for_final_cycle(
    output: State,
    heart: HeartChamberPressureFlowRate,
    tspan: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def final_cycle(arr: Any) -> np.ndarray:
        return np.asarray(arr)[-tspan:]

    organ_diff = np.array(
        [
            np.mean(final_cycle(output.Q_RL1 - output.Q_RL2)),
            np.mean(final_cycle(output.Q_LL1 - output.Q_LL2)),
            np.mean(final_cycle(output.Q_ICar - output.Q_Cer)),
            np.mean(final_cycle(output.Q_Sub - output.Q_ULimb)),
            np.mean(final_cycle(output.Q_ECar - output.Q_Fac)),
            np.mean(final_cycle(output.Q_Ren1 - output.Q_Ren2)),
            np.mean(final_cycle(output.Q_Mes1 - output.Q_Mes2)),
            np.mean(final_cycle(output.Q_Spl1 - output.Q_Spl2)),
            np.mean(final_cycle(output.Q_Hep1 - output.Q_Hep2)),
            np.mean(final_cycle(output.Q_LLimb1 - output.Q_LLimb2)),
        ]
    )

    artery_diff = np.array(
        [
            np.mean(final_cycle(heart.Q_PV - output.Q_Ar_Pul)),
            np.mean(final_cycle(output.Q_Ar_Pul - output.Q_LL1 - output.Q_RL1)),
            np.mean(final_cycle(output.Q_LL2 + output.Q_RL2 - output.Q_Ven_Pul)),
            np.mean(final_cycle(heart.Q_AV - output.Q_AA)),
            np.mean(final_cycle((output.Q_AA - output.Q_Sub - output.Q_CCar) - output.Q_DscA)),
            np.mean(final_cycle(output.Q_AA - output.Q_DscA - output.Q_Sub - output.Q_CCar)),
            np.mean(final_cycle(output.Q_AbdA - output.Q_Mes1 - output.Q_LLimb1 - output.Q_Cel)),
            np.mean(final_cycle(output.Q_AA - output.Q_CCar - output.Q_DscA - output.Q_Sub)),
            np.mean(final_cycle(output.Q_CCar - output.Q_ICar - output.Q_ECar)),
            np.mean(final_cycle(output.Q_CCar - output.Q_ECar - output.Q_ICar)),
            np.mean(final_cycle(output.Q_Cel - output.Q_Spl1 - output.Q_Hep1)),
            np.mean(final_cycle(output.Q_DscA - output.Q_AbdA - output.Q_Ren1)),
            np.mean(final_cycle(output.Q_AbdA - output.Q_LLimb1 - output.Q_Cel - output.Q_Mes1)),
            np.mean(final_cycle(output.Q_Cel - output.Q_Hep1 - output.Q_Spl1)),
        ]
    )

    vein_diff = np.array(
        [
            np.mean(final_cycle(output.Q_LLimb2 + output.Q_Ven_Por + output.Q_Hep2 + output.Q_Ren2 - output.Q_IVC)),
            np.mean(final_cycle(output.Q_Fac + output.Q_Cer + output.Q_ULimb - output.Q_SVC)),
            np.mean(final_cycle(output.Q_Mes2 + output.Q_Spl2 - output.Q_Ven_Por)),
        ]
    )

    return organ_diff, artery_diff, vein_diff


def _write_convergence_metrics(
    init_state: State,
    params: Params,
    cycles: list[int],
    tspan: int,
    model_output_dir: Path,
) -> None:
    conv_dir = model_output_dir / "convergence"
    conv_dir.mkdir(parents=True, exist_ok=True)

    for n_cycles in cycles:
        output, heart, _, _ = _run_state_output(
            init_state=init_state,
            params=params,
            n_cycles=n_cycles,
            tspan=tspan,
            last_cycles=1,
        )
        org, art, ven = _mean_flow_diffs_for_final_cycle(output, heart, tspan=tspan)

        np.savetxt(conv_dir / f"mean_organ_flow_diff_{n_cycles}.csv", org, delimiter=",")
        np.savetxt(conv_dir / f"mean_artery_flow_diff_{n_cycles}.csv", art, delimiter=",")
        np.savetxt(conv_dir / f"mean_vein_flow_diff_{n_cycles}.csv", ven, delimiter=",")


def run_model_simulation(
    n_cycles: int = 101,
    tspan: int = 800,
    last_cycles: int = 5,
    convergence_cycles: list[int] | None = None,
    parameters_dir: Path = PARAMETERS_DIR,
    model_output_dir: Path = MODEL_OUTPUT_DIR,
) -> dict[str, Any]:
    ensure_pipeline_dirs()
    model_output_dir.mkdir(parents=True, exist_ok=True)

    init_state, params = load_baseline_inputs(parameters_dir)

    output, heart, save_times, final_times = _run_state_output(
        init_state=init_state,
        params=params,
        n_cycles=n_cycles,
        tspan=tspan,
        last_cycles=last_cycles,
    )

    np.savez_compressed(model_output_dir / "state.npz", **_namedtuple_npz_dict(output))
    np.savez_compressed(model_output_dir / "heart.npz", **_namedtuple_npz_dict(heart))
    np.savez_compressed(model_output_dir / "times.npz", save_times=save_times, final_times=final_times)

    shutil.copy2(parameters_dir / "params_bsln.json", model_output_dir / "params_bsln.json")
    shutil.copy2(parameters_dir / "state_bsln.json", model_output_dir / "state_bsln.json")

    cycles = convergence_cycles[:] if convergence_cycles else [n_cycles]
    cycles = [c for c in cycles if c > 0]
    if not cycles:
        cycles = [n_cycles]
    cycles = list(dict.fromkeys(cycles))
    _write_convergence_metrics(init_state, params, cycles, tspan=tspan, model_output_dir=model_output_dir)

    metadata = {
        "generated_utc": utc_now_iso(),
        "n_cycles": int(n_cycles),
        "tspan": int(tspan),
        "last_cycles": int(last_cycles),
        "convergence_cycles": cycles,
        "bpm": float(params.BPM),
    }
    write_json(model_output_dir / "metadata.json", metadata)

    return {**metadata, "model_output_dir": str(model_output_dir)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model simulation and save artifacts to output/model.")
    parser.add_argument("--n-cycles", type=int, default=101, help="Simulation cycles for primary output.")
    parser.add_argument("--tspan", type=int, default=800, help="Time samples per cycle.")
    parser.add_argument("--last-cycles", type=int, default=5, help="Trailing cycles retained in state/heart outputs.")
    parser.add_argument(
        "--convergence-cycles",
        type=str,
        default="101",
        help="Comma-separated cycle counts for convergence CSVs (e.g., '21,31,41,51,81,101').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cycles = parse_cycle_list(args.convergence_cycles)
    metadata = run_model_simulation(
        n_cycles=args.n_cycles,
        tspan=args.tspan,
        last_cycles=args.last_cycles,
        convergence_cycles=cycles,
    )
    print(f"[DONE] Model output saved to: {metadata['model_output_dir']}")


if __name__ == "__main__":
    main()
