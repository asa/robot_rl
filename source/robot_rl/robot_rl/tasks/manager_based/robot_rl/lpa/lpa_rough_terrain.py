"""Mildly rough terrain for ROBUSTNESS, not for climbing.

User 2026-08-21: "add terrain randomization into the walking so we can
handle uneven terrain, even if we are only walking on flat ground."

The standard ladder in legged RL (legged_gym / IsaacLab, Rudin et al.)
is a grid whose COLUMNS are terrain types and whose ROWS are
difficulty, with a curriculum promoting a robot up a row when it
traverses past half the patch and demoting it when it does not:

    random rough (uniform noise)  0.02 -> 0.10 m
    pyramid slope up / down       0 -> 0.4 rad
    pyramid stairs up / down      0.05 -> 0.23 m step
    discrete obstacles            0.05 -> 0.2 m boxes

Those upper rungs are quadruped numbers. This config deliberately
takes only the FIRST rung and only its low end -- 0.01 to 0.05 m of
noise -- because the goal here is a flat-ground policy that stops
assuming a perfect plane, not a stair climber. Slopes already have
their own env (lpa_ramp_terrain, tinh-lpa-ramp); stairs and obstacles
are a separate behaviour, not free robustness.

The curriculum IS enabled here (unlike the ramp env, where difficulty
is inert because the slopes are fixed): noise amplitude genuinely
scales with row, so terrain_levels has something to climb.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=5,      # difficulty rows -- amplitude scales across these
    num_cols=4,      # replicas
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        # Flat stays in the mix so the policy keeps seeing the case it
        # actually ships on.
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.3),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.7,
            noise_range=(0.01, 0.05),
            noise_step=0.005,
            border_width=0.25,
        ),
    },
)
