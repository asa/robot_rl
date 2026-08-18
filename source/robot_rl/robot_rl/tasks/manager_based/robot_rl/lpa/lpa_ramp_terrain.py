"""Ramp terrain for LPA slope walking (tinh-lpa-ramp.5).

A TRUE INCLINED PLANE per column: height rises linearly along +x at
a fixed slope, so the fall line is +x everywhere and a robot facing
+x is always walking straight up (or down) the ramp.

Pyramid sub-terrains were tried first and produced a policy that
walked 10 m while gaining ~0 m altitude (climb gate 2026-08-18):
their slope is RADIAL, so a robot whose heading is not aligned to
the radius simply traverses the hill at constant height. The CLF
reward tracks the reference in the stance frame and never mentions
altitude, so training looked healthy the whole time.

With `curriculum=True` the generator assigns sub-terrains across
columns by proportion, so `terrain_types[env]` is an exact index
into `RAMP_COLUMNS` — the env's slope is known without any height
sensing, and the matching reference is selected explicitly (slope is
terrain, not a velocity command; the conditioner cannot separate
these gaits — they all sit at vel_x 0.27-0.35).

Angles are the gate-verified chained slopes of the solved gaits
(tinh-lpa-ramp.3), not the nominal commands.
"""

import math

import numpy as np
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass


@height_field_to_mesh
def inclined_plane(difficulty: float, cfg) -> np.ndarray:
    """Constant-slope plane rising along +x (slope is a ratio =
    tan(theta); negative descends). Flat run-in at the low-x edge,
    zeroed at the patch centre so env origins sit on the surface."""
    w = int(cfg.size[0] / cfg.horizontal_scale)
    l = int(cfg.size[1] / cfg.horizontal_scale)
    x = np.arange(w) * cfg.horizontal_scale
    run = np.clip(x - cfg.flat_start, 0.0, None)
    h = (cfg.slope * run) / cfg.vertical_scale
    h = h - h[w // 2]
    return np.tile(h[:, None], (1, l)).astype(np.int16)


@configclass
class HfInclinedPlaneCfg(HfTerrainBaseCfg):
    """A constant-slope plane. Unlike the pyramid terrains the fall
    line is the SAME direction everywhere (+x), so 'walk forward'
    means 'walk up the ramp'.

    NOTE: `function` must be declared in the CLASS BODY — assigning
    it after @configclass leaves the dataclass default at None and
    the generator raises 'NoneType is not callable'.
    """

    function = inclined_plane
    slope: float = 0.0
    flat_start: float = 1.5


# (column name, reference gait name, signed degrees) — ORDER IS THE
# COLUMN ORDER the generator assigns, and the event maps
# terrain_types -> this list by index.
RAMP_COLUMNS = [
    # The proven flat stomp keeper, unmodified (vel_x 0.596).
    ("flat", "walk_forward", 0.0),
    ("up5", "lpa_ramp_up5", 5.63),
    ("up10", "lpa_ramp_up10", 10.36),
    ("up12", "lpa_ramp_up12", 12.56),
    ("dn4", "lpa_ramp_dn4", -4.05),
    ("dn7", "lpa_ramp_dn7", -7.36),
]


def _sub(deg: float):
    return HfInclinedPlaneCfg(
        proportion=1.0 / len(RAMP_COLUMNS),
        slope=math.tan(math.radians(deg)),   # signed: +climb, -descend
        flat_start=1.5,
        border_width=0.25,
    )


RAMP_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=5.0,
    num_rows=5,        # replicas; difficulty is inert (fixed slopes)
    num_cols=len(RAMP_COLUMNS),
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=None,   # keep the ramp a true plane, no stair-stepping
    curriculum=True,        # columns <-> sub-terrains, deterministically
    sub_terrains={name: _sub(deg) for name, _, deg in RAMP_COLUMNS},
)
