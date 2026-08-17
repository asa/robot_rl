"""Ramp terrain for LPA slope walking (tinh-lpa-ramp.5).

One sub-terrain PER COLUMN, each a fixed slope matching a solved
reference gait. With `curriculum=True` the generator assigns
sub-terrains across columns by proportion, so `terrain_types[env]`
is an exact index into `RAMP_COLUMNS` — the env's slope is known
without any height sensing, and the matching reference is selected
explicitly (slope is terrain, not a velocity command; the conditioner
cannot separate these gaits — they all sit at vel_x 0.27-0.35).

Slopes are RATIOS (height/width = tan(theta)), matching IsaacLab's
`pyramid_sloped_terrain`. `inverted=True` puts the platform at the
BOTTOM with the surface rising outward, so walking away from the
spawn point is a CLIMB; the plain pyramid descends.

Angles are the gate-verified chained slopes of the solved gaits
(tinh-lpa-ramp.3), not the nominal commands.
"""

import math

from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.height_field import (
    HfInvertedPyramidSlopedTerrainCfg,
    HfPyramidSlopedTerrainCfg,
)

# (column name, reference gait name, signed degrees) — ORDER IS THE
# COLUMN ORDER the generator assigns, and the event maps
# terrain_types -> this list by index.
RAMP_COLUMNS = [
    ("flat", "lpa_ramp_flat", 0.0),
    ("up5", "lpa_ramp_up5", 5.63),
    ("up10", "lpa_ramp_up10", 10.36),
    ("up12", "lpa_ramp_up12", 12.56),
    ("dn4", "lpa_ramp_dn4", -4.05),
    ("dn7", "lpa_ramp_dn7", -7.36),
]


def _sub(deg: float):
    t = abs(math.tan(math.radians(deg)))
    common = dict(
        proportion=1.0 / len(RAMP_COLUMNS),
        slope_range=(t, t),          # fixed: difficulty must not move it
        platform_width=2.0,          # room to stand up before climbing
        border_width=0.25,
    )
    if deg > 0:
        return HfInvertedPyramidSlopedTerrainCfg(**common)
    return HfPyramidSlopedTerrainCfg(**common)


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
