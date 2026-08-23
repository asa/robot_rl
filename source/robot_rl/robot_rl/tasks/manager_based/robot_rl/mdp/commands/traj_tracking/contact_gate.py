"""Contact-gated reference clock — pure tensor logic, no Isaac imports.

WHY THIS FILE EXISTS SEPARATELY: the first two versions of the gate
shipped inside trajectory_cmd.py, and both were wrong in ways a unit
test would have caught in seconds. trajectory_cmd imports isaaclab,
so testing it means booting Isaac; this module imports only torch, so
test_contact_gate.py runs in milliseconds on the training box.

THE SEMANTICS, AND WHY EDGE-TRIGGERED.

Version 1 held the clock whenever (expected_stance & !measured) on any
foot. That predicate cannot tell two opposite situations apart:

  late GROUND    reference entered stance, foot still in the air
                 -> robot is BEHIND, holding is correct
  early TOE-OFF  robot lifted a moment before the reference ended its
                 stance -> robot is AHEAD, holding is exactly wrong

Ordinary gait variation produces the second case at nearly every
step, so the uniform predicate would have parked the clock at its
hold ceiling during perfectly normal flat walking. Symmetrically, the
catch-up branch fired on late toe-off (foot still down after the
reference entered swing) and sped the clock up when the robot was
BEHIND. Both misfires come from reading a level where the information
lives in an EDGE.

So the gate is edge-triggered: when the REFERENCE's expected contact
for a foot RISES (swing -> stance) and the foot has not measured
ground, that foot is "awaiting contact" and the clock holds until it
lands or the hold cap expires. A mismatch that begins anywhere else —
toe-off timing slop, a mid-stance force dropout, scuffing during
swing — never arms the gate.

Catch-up (running the clock fast after an early landing) is REMOVED,
not deferred silently: a correct version needs the same edge scoping
plus a window test, and the hold-only gate is the measurable first
step. Hold-only also means the offset is monotone non-decreasing,
which makes the telemetry trivially interpretable.

Version 2's second bug: expected contact was evaluated at NOMINAL
time, but the robot tracks the reference at EFFECTIVE time
(nominal - offset). Once the gate had ever held, every later
comparison was made against a frame the robot was not tracking, and
the two clocks could disagree forever. The caller must pass expected
contact evaluated at t_eff; contact_gate_step is written so it cannot
be called any other way (it takes the already-evaluated tensor).

The deadlock guard stays: a fallen robot never lands the awaited
foot, so an uncapped hold stops its clock forever and the reference
never asks it to step again. After max_hold_s the clock resumes and
the episode fails honestly.
"""

from __future__ import annotations

import torch


def contact_gate_step(
    expected: torch.Tensor,       # [N, C] bool — reference contact AT t_eff
    prev_expected: torch.Tensor,  # [N, C] bool — same, previous step
    awaiting: torch.Tensor,       # [N, C] bool — feet armed by a rising edge
    elapsed: torch.Tensor,        # [N] float — seconds of the current hold
    offset: torch.Tensor,         # [N] float — total seconds held (monotone)
    measured: torch.Tensor,       # [N, C] bool — feet actually on the ground
    fresh: torch.Tensor,          # [N] bool — first step of an episode
    dt: float,
    max_hold_s: float,
):
    """One step of the gate. Returns
    (prev_expected', awaiting', elapsed', offset', holding, alpha).

    alpha = d(t_eff)/d(t_real): 0 while holding, else 1. The caller
    MUST scale the reference velocity by it (the chain rule for
    y_des(t_eff(t_real))) or the reference is frozen in position while
    still moving at full mid-swing speed.
    """
    fresh_c = fresh.unsqueeze(-1)
    # A fresh episode carries no history: no armed feet, no hold, no
    # offset, and prev_expected seeded from the CURRENT frame so the
    # reset itself can never fabricate a rising edge.
    prev_expected = torch.where(fresh_c, expected, prev_expected)
    awaiting = awaiting & ~fresh_c
    zeros_f = torch.zeros_like(elapsed)
    elapsed = torch.where(fresh, zeros_f, elapsed)
    offset = torch.where(fresh, zeros_f, offset)

    rising = expected & ~prev_expected
    # Arm on a rising edge with no ground; disarm the moment the foot
    # lands, and also if the reference has moved on past that stance
    # (post-cap: the clock resumed and the domain ended — a stale arm
    # must not re-hold the next unrelated mismatch).
    awaiting = (awaiting | (rising & ~measured)) & ~measured & expected

    awaiting_any = awaiting.any(dim=1)
    elapsed = torch.where(awaiting_any, elapsed + dt, zeros_f)
    holding = awaiting_any & (elapsed <= max_hold_s)

    offset = torch.where(holding, offset + dt, offset)
    alpha = torch.where(holding, zeros_f, torch.ones_like(offset))

    return expected, awaiting, elapsed, offset, holding, alpha
