# Copyright 2026 Zachary Olkin. All rights reserved.

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .trajectory_cmd import TrajectoryCommand


@configclass
class TrajectoryCommandCfg(CommandTermCfg):
    """
    Configuration for trajectory commands.
    """

    class_type: type = TrajectoryCommand
    asset_name: str = "robot"
    contact_bodies: list[str] = None
    manager_type: str = ""
    conditioner_generator_name: str = ""
    num_outputs: int = -1
    path: str = ""
    hf_repo: str = None
    Q_weights: list[float] = None
    R_weights: list[float] = None
    resampling_time_range: tuple[float, float] = (5.0, 15.0)    # TODO: How can I remove this?
    heuristic_func = None
    random_start_time_max: float = -1
    percent_hold_phi: float = -1
    hold_phi_threshold: float = -1

    # --- contact-gated phase advance (tinh: rough-terrain fix) ---
    #
    # The reference clock is t = episode_length_buf * step_dt, a pure
    # step counter with no robot input. On uneven ground the robot is
    # delayed by ground contact, the reference does not wait, and the
    # tracking error runs past the knee of exp(-KAPPA*V/sigma) where
    # the gradient dies -- the amplitude-independent failure measured
    # across rough1-8 (5 cm rubble 50% style retention, 2 cm smooth
    # undulation 51%).
    #
    # Semantics are EDGE-TRIGGERED and hold-only; see contact_gate.py
    # for the two shipped bugs that forced that design (a uniform
    # level test misfires at every toe-off, and a catch-up branch with
    # the same defect sped the clock up when the robot was behind).
    contact_gate: bool = False
    # Newtons on a foot to count as contact. Matches the threshold the
    # contact rewards already use.
    contact_gate_force: float = 1.0
    # Hard ceiling per hold. WITHOUT THIS the gate deadlocks: a fallen
    # robot never lands the awaited foot, the clock stops forever, and
    # the reference never asks it to step again.
    contact_gate_max_hold_s: float = 0.20
    phasing_boundaries: float = 1
    # Behavior-graph skill observations (tinh-lpa-clfrl.8.5d): the
    # declared obs layout. skill_slots is the one-hot vocabulary
    # (slot 0 should be "locomotion"); param_channels the named
    # continuous conditioner params (throw azimuth, wall height, ...).
    # None = skill obs disabled (legacy envs unchanged).
    skill_slots: list[str] = None
    param_channels: list[str] = None
    # Full-body imitation mode (tinh-lpa-clfrl.7.7.v3): during
    # explicit graph segments every joint error channel is scaled by
    # imitation_gain (equivalent to Q x gain^2), easing in over
    # imitation_ease_s. 1.0 = off (legacy envs unchanged).
    imitation_gain: float = 1.0
    imitation_ease_s: float = 0.3
    # Anchored yaw tracking (7.7.v3). Fires on ANY explicit
    # reference, so it is OFF by default — the laser/skill envs must
    # keep their original heading-relative behavior. Turn work is
    # parked (2026-08-17); re-enable per-env when it resumes.
    anchored_yaw: bool = False