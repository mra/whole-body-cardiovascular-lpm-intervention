#!/usr/bin/env python3
"""Primary entrypoint for model simulation and post-analysis."""

from __future__ import annotations

import argparse

from common import parse_cycle_list
from post_analysis import run_post_analysis
from simulate import run_model_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model simulation and manuscript post-analysis.")
    parser.add_argument("--n-cycles", type=int, default=101, help="Simulation cycles for saved model output.")
    parser.add_argument("--tspan", type=int, default=800, help="Time samples per cardiac cycle.")
    parser.add_argument("--last-cycles", type=int, default=5, help="Number of trailing cycles retained in output.")
    parser.add_argument(
        "--convergence-cycles",
        type=str,
        default="101",
        help="Comma-separated cycle counts for convergence CSV generation and Fig11 (e.g., '21,31,41,51,81,101').",
    )
    parser.add_argument("--run-sobol", action="store_true", help="Compute Sobol ST indices and regenerate sensitivity figures.")
    parser.add_argument("--sobol-samples", type=int, default=64, help="Sobol base sample size N (total runs: N*(D+2)).")
    parser.add_argument("--sobol-n-cycles", type=int, default=40, help="Simulation cycles per Sobol sample.")
    parser.add_argument("--sobol-tspan", type=int, default=400, help="Time samples in final cycle per Sobol run.")
    parser.add_argument("--sobol-variation", type=float, default=0.10, help="Uniform perturbation fraction around nominal parameters.")
    parser.add_argument("--sobol-seed", type=int, default=42, help="Sobol sampler random seed.")
    parser.add_argument("--sobol-threshold", type=float, default=0.20, help="Heatmap display threshold for ST values.")
    parser.add_argument("--skip-simulation", action="store_true", help="Skip simulation and reuse existing output/model.")
    parser.add_argument("--skip-post-analysis", action="store_true", help="Skip figure/table generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convergence_cycles = parse_cycle_list(args.convergence_cycles)

    if not args.skip_simulation:
        print("[INFO] Running model simulation...")
        metadata = run_model_simulation(
            n_cycles=args.n_cycles,
            tspan=args.tspan,
            last_cycles=args.last_cycles,
            convergence_cycles=convergence_cycles,
        )
        print(f"[DONE] Model output saved to: {metadata['model_output_dir']}")
        print(
            "[INFO] Metadata: "
            f"{{'generated_utc': '{metadata['generated_utc']}', "
            f"'n_cycles': {metadata['n_cycles']}, "
            f"'tspan': {metadata['tspan']}, "
            f"'last_cycles': {metadata['last_cycles']}, "
            f"'convergence_cycles': {metadata['convergence_cycles']}, "
            f"'bpm': {metadata['bpm']}}}"
        )

    if not args.skip_post_analysis:
        print("[INFO] Running post-analysis...")
        result = run_post_analysis(
            convergence_cycles=convergence_cycles if convergence_cycles else None,
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
