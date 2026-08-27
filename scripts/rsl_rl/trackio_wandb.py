"""wandb-shaped adapter over trackio (Hugging Face).

rsl_rl's WandbSummaryWriter is vendored — it stays byte-for-byte and
keeps calling the wandb module surface. train_policy installs THIS
module as sys.modules["wandb"] before the runner constructs, so the
writer's five calls (init / config.update / log / finish / save)
land on trackio instead: local-first metrics (sqlite under
TRACKIO_DIR, set by launch_run to <run_dir>/work/trackio so the
series lives inside the run dataset and its @closeout snapshot),
no external telemetry, dashboard via `trackio show`.

Deliberately minimal: anything rsl_rl does not call is absent, so a
future rsl_rl bump that widens its wandb usage fails loudly here
instead of silently uploading somewhere.
"""

from __future__ import annotations

import trackio

run = None


class _Config:
    """wandb.config lookalike. Values are recorded at init time via
    trackio's config= when possible; later .update() calls are
    forwarded as config metadata if trackio's run supports it and
    dropped otherwise — the run.yaml spec is the durable config
    record, not the tracker."""

    def update(self, d, **kw):
        r = trackio.run if hasattr(trackio, "run") else None
        cfg = getattr(r, "config", None)
        if cfg is not None and hasattr(cfg, "update"):
            cfg.update(d)


config = _Config()


class _Run:
    def __init__(self, name):
        self.name = name


def init(project=None, entity=None, name=None, **kw):
    # entity is a wandb-ism; trackio has no accounts
    global run
    trackio.init(project=project or "robot_rl", name=name)
    run = _Run(name)
    return run


def log(data, step=None, **kw):
    trackio.log(data, step=step)


_finished = False


def finish(**kw):
    # Idempotent: rsl_rl never calls writer.stop(), so train_policy
    # calls finish() explicitly after learn() — and trackio's own
    # atexit may fire too on clean interpreters. Isaac's Kit teardown
    # exits via os._exit, which skips atexit entirely: without the
    # explicit call the entire metric series is dropped (smoke
    # 2026-08-27: 10 iterations, 0 rows).
    global _finished
    if _finished:
        return
    _finished = True
    trackio.finish()


def save(path, base_path=None, **kw):
    # checkpoints already live in the run dataset; nothing to upload
    pass
