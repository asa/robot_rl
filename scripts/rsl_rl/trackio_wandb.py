"""wandb-shaped LIVE metrics writer over trackio's storage layer.

Installed as sys.modules["wandb"] so rsl_rl's vendored
WandbSummaryWriter logs here. Trackio's OWN client (init/log/finish)
is a queue-and-server design that cannot survive Isaac Kit: the
smoke matrix (2026-08-27) showed init/log/finish all execute with
real data and zero metric rows persist — the writer thread dies
with Kit's event loop and Kit exits via os._exit, skipping atexit.
The proof that DIRECT writes work came from the same smoke: trackio's
system-metrics thread (a direct sqlite path) landed its rows fine.

So this shim writes through trackio.sqlite_storage.SQLiteStorage
synchronously — one merged row per training iteration, flushed when
the global step advances — into the SHARED store (TRACKIO_DIR =
/opt/tinh/data/lpa/trackio), which the dashboard service reads
live. The closeout's tfevents ingestion then REPLACES the run with
the canonical tensorboard series (idempotent), reconciling anything
a crash dropped.

Telemetry must never sink a training run: every storage call is
fenced; after repeated failures the shim disarms loudly and
training continues without live metrics (tensorboard remains the
source of truth).
"""

from __future__ import annotations

import datetime
import os
import time

from trackio.sqlite_storage import SQLiteStorage

run = None
_project = os.environ.get("TRACKIO_PROJECT", "lpa")
_run_name = None
_buf: dict = {}
_step: int | None = None
_last_flush = 0.0
_failures = 0
import re as _re
_INCLUDE = _re.compile(os.environ.get(
    "TRACKIO_METRICS_INCLUDE",
    r"^(Episode_Reward|Episode_Termination|Train|Loss|"
    r"Curriculum|Perf|Policy|Metrics/base_velocity)"))
_MAX_FAILURES = 20


class _Config:
    def update(self, d, **kw):
        pass


config = _Config()


class _Run:
    def __init__(self, name):
        self.name = name


def _fenced(fn):
    global _failures
    if _failures >= _MAX_FAILURES:
        return
    try:
        fn()
    except Exception as e:
        _failures += 1
        if _failures in (1, _MAX_FAILURES):
            print(f"[trackio-live] storage write failed ({e}); "
                  f"{'DISARMED — tensorboard remains authoritative' if _failures >= _MAX_FAILURES else 'will keep trying'}",
                  flush=True)


def init(project=None, entity=None, name=None, **kw):
    # rsl_rl passes its log-dir name (<timestamp>_<run_id>); the
    # dashboard should show the run id — launch_run exports it
    # (user 2026-08-27: date + zero-padded name, no timestamp).
    global run, _run_name
    _run_name = os.environ.get("TRACKIO_RUN_NAME") or name or "unnamed"
    run = _Run(_run_name)
    return run


def _flush():
    global _buf, _last_flush
    if not _buf or _run_name is None:
        return
    payload = dict(_buf)
    step = int(_step or 0)
    _fenced(lambda: SQLiteStorage.bulk_log(
        project=_project, run=_run_name,
        metrics_list=[payload], steps=[step],
        # ISO-8601 or the dashboard renders nothing (2026-08-27)
        timestamps=[datetime.datetime.now(
            datetime.timezone.utc).isoformat()]))
    _buf.clear()
    _last_flush = time.monotonic()


def log(data, step=None, **kw):
    global _step
    if step is not None and _step is not None and step != _step:
        _flush()
    if step is not None:
        _step = int(step)
    # rsl_rl hands raw torch Tensors as scalar values; the storage
    # layer JSON-serializes. Coerce to float, drop what will not.
    for k, v in data.items():
        # dashboard-worthy series only: the full tag set (~115
        # per-joint traj_ref channels) renders hundreds of charts
        # and kills the browser; deep traces live in tensorboard.
        if not _INCLUDE.match(k):
            continue
        try:
            _buf[k] = float(v)
        except (TypeError, ValueError):
            pass
    if time.monotonic() - _last_flush > 60:
        _flush()


def finish(**kw):
    _flush()


def save(path, base_path=None, **kw):
    pass
