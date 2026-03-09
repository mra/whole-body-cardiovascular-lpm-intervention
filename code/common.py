#!/usr/bin/env python3
"""Shared constants and helper functions for the code pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data_structures import Params, State


CODE_DIR = Path(__file__).resolve().parent
PARAMETERS_DIR = CODE_DIR / "parameters"
OUTPUT_DIR = CODE_DIR / "output"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "model"
FIGURES_DIR = OUTPUT_DIR / "figures"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_cycle_list(value: str | None) -> list[int]:
    if not value:
        return []
    out: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def ensure_pipeline_dirs() -> None:
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_OUTPUT_DIR / "convergence").mkdir(parents=True, exist_ok=True)


def load_baseline_inputs(parameters_dir: Path = PARAMETERS_DIR) -> tuple[State, Params]:
    init_state = State.from_dict(load_json(parameters_dir / "state_bsln.json"))
    params = Params.from_dict(load_json(parameters_dir / "params_bsln.json"))
    return init_state, params
