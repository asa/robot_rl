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
takes only the FIRST rung and only its low end -- 0.005 to 0.02 m of
gently-correlated noise -- because the goal here is a flat-ground
policy that stops assuming a perfect plane, not a stair climber. Slopes already have
their own env (lpa_ramp_terrain, tinh-lpa-ramp); stairs and obstacles
are a separate behaviour, not free robustness.

WHAT THE FIRST VERSION OF THIS FILE GOT WRONG (rough1, 2026-08-21 --
14 h of training that produced a policy which flails and falls on
flat ground). Three claims in this docstring were asserted about
IsaacLab without checking, and all three were false:

  1. "noise amplitude scales with row." It does not. Upstream's
     random_uniform_terrain says outright: "The difficulty parameter
     is IGNORED for this terrain." All num_rows rows are statistically
     identical, so there was never an easy end to start from.
  2. "The curriculum IS enabled here." It was not. The base LPA env
     sets curriculum.terrain_levels = None and this cfg's env
     inherited that, so nothing was ever promoted. It is still None
     below, now DELIBERATELY: promoting between identical rows buys
     nothing, and pretending otherwise is what caused this.
  3. The roughness was maximal, not mild. downsampled_scale was
     unset, so it defaults to horizontal_scale (0.1 m) and every
     10 cm cell is an INDEPENDENT draw -- white-noise rubble with no
     spatial correlation. A foot spans 2-3 such cells and can meet a
     4 cm edge on any step. Upstream pairs that terrain with a height
     scanner; this policy is deliberately BLIND (height_scan = None,
     to keep the obs layout identical for resume), so it must walk it
     on proprioception alone.

The measured consequence: the reference stomp gait is untrackable on
that surface, so the policy abandoned it. Per-step joint_pos reward
fell 62% within ~1500 iterations, floored at 19% of baseline, and
never recovered in the remaining 12,000. Falls went 0.9% -> 19.4%.

So this version makes the ground genuinely mild:

  downsampled_scale 0.4  heights drawn on a 40 cm grid and
                         interpolated -- undulation the foot can sit
                         flat on, instead of per-cell rubble. This is
                         the single biggest lever.
  noise_range 0.5-2 cm   the actual low end of the first rung.
  flat 0.4               more of the surface the robot ships on.

Difficulty stays uniform because upstream makes it so; a real ladder
needs a difficulty-respecting terrain function, which is a follow-up
rather than a claim made in a comment.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    # Rows are NOT a difficulty ladder here -- random_uniform ignores
    # difficulty (see docstring). They are simply more patch variety.
    num_rows=5,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    # No ladder to climb: every row is the same distribution.
    curriculum=False,
    sub_terrains={
        # Flat stays a large share: it is the surface that actually
        # ships, and the previous run proved the styled gait is lost
        # if the policy rarely sees it.
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.4),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.6,
            noise_range=(0.005, 0.02),
            noise_step=0.005,
            # THE fix: without this, heights are independent per
            # horizontal_scale (10 cm) cell. 0.4 m draws a coarse grid
            # and interpolates, giving undulation with a wavelength
            # longer than the foot rather than rubble under it.
            downsampled_scale=0.4,
            border_width=0.25,
        ),
    },
)
