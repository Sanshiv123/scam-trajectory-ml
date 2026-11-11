"""Claim fresh run directories atomically and leave inspectable failure states."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from experiments.config import ROOT, RunConfig


def output_path(config: RunConfig, run_name=None) -> Path:
    if run_name is None:
        return ROOT / "outputs" / config.mode
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_name):
        raise ValueError("run-name must be 1-64 letters, digits, hyphens or underscores")
    return ROOT / "outputs" / "runs" / f"{config.mode}-{run_name}"


@contextmanager
def new_run(path: Path, config: RunConfig):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()  # Atomic: concurrent launches cannot own the same directory.
    except FileExistsError:
        raise FileExistsError(
            f"Run directory already exists: {path}. Choose a fresh --run-name; "
            "existing completed, prepared and failed runs are never overwritten."
        ) from None

    def status(stage, error=None):
        state = {
            "stage": stage,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "config": config.as_dict(),
        }
        if error is not None:
            state["error"] = f"{type(error).__name__}: {error}"
        (path / "run_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    status("running")
    try:
        yield path
    except BaseException as error:
        status("failed", error)
        raise
    else:
        status("prepared" if config.prepare_only else "completed")
