"""Plot rendering and optional debug-image persistence."""

from __future__ import annotations

import base64
from datetime import UTC, date, datetime, timedelta
import io
import logging
import os
from pathlib import Path
import threading
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.graph_objects as go

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_OUTPUT_DIR = _PROJECT_ROOT / "output"

# Each kaleido render boots its own headless Chromium process (150-300MB+ RSS).
# Tool calls run concurrently (one per chat request), so without a cap, a burst
# of chart requests can spawn enough Chromium instances to OOM the container.
# Extra calls beyond the limit block here rather than piling on more renders.
_MAX_CONCURRENT_CHART_RENDERS = max(
    1, int(os.environ.get("SCANNER_MCP_CHART_RENDER_WORKERS", "2"))
)
_CHART_RENDER_SEMAPHORE = threading.Semaphore(_MAX_CONCURRENT_CHART_RENDERS)


def debug_png_enabled() -> bool:
    """Return whether debug chart PNGs should be persisted to disk."""
    return os.environ.get("ENABLE_DEBUG_PNG", "").strip().lower() in {"1", "true", "yes", "on"}


def save_debug_png(chart_type: str, png_bytes: bytes) -> None:
    """Persist a generated PNG under project-root/output for local debugging."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    filename = f"{chart_type}_{timestamp}_{uuid4().hex[:8]}.png"
    (_OUTPUT_DIR / filename).write_bytes(png_bytes)


def _json_safe_plotly_value(value: object) -> object:
    """Convert datetime-like Pandas/NumPy values into JSON-safe primitives."""
    if isinstance(value, dict):
        return {key: _json_safe_plotly_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_plotly_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe_plotly_value(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def fig_to_b64(fig: go.Figure, chart_type: str) -> str:
    """Render a Plotly figure to PNG bytes, save a debug copy, and return base64."""
    fig = go.Figure(_json_safe_plotly_value(fig.to_dict()))
    buf = io.BytesIO()
    with _CHART_RENDER_SEMAPHORE:
        # No `engine=` kwarg: plotly 7.x removed it (kaleido is the only engine now)
        # and passing it raises TypeError. Omitting it works on both 6.x and 7.x.
        fig.write_image(buf, format="png", scale=1.5)
    buf.seek(0)
    png_bytes = buf.getvalue()
    if debug_png_enabled():
        try:
            save_debug_png(chart_type, png_bytes)
        except OSError:
            log.exception("Failed to save debug chart image")
    return base64.b64encode(png_bytes).decode("ascii")
