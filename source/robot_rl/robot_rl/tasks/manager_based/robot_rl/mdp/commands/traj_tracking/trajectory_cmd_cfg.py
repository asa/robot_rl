# Copyright 2026 Zachary Olkin. All rights reserved.

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .trajectory_cmd import TrajectoryCommand


@configclass
class ArmSwingOverlayCfg:
    """Phase-locked elliptical arm swing overlaid on the tracked
    reference (v6 honest-gait, user direction 2026-08-26).

    The stand carriage holds shoulder yaw rotated IN (-0.70 rad) with
    the elbow flexed; the symforce collision bank measures 0.9 mm of
    forearm-hip clearance over any pitch swing from that pose -- the
    reference parks the arms where swinging is mechanically
    impossible, which is why rough21-24 never swung regardless of
    reward shaping. The style target (fists in at the swing extremes,
    out when passing the hips) is an ellipse in (pitch, yaw): at the
    pitch extremes the forearm is displaced fore/aft so yaw-in is
    safe; at hip passage yaw must open out. Bank clearance at the
    defaults below: 57-92 mm over the full cycle.

    Pitch is ADDED to the reference (L +amp*sin, R mirrored); yaw
    REPLACES it (the carriage value is the hazard being removed).
    Values are L-side signed; R is negated. Applies to locomotion
    envs only -- explicit behavior references keep their solved arms.
    """

    pitch_amp: float = 0.40
    yaw_in: float = -0.55
    yaw_out: float = -0.10
    # radians added to 2*pi*phi, aligning the arm cycle to the leg
    # cycle. Derived for the pendulum-stomp walk_forward library
    # (cycle 1.531 s): R heel strike at phi 0.794, and NEGATIVE
    # shoulder pitch is hand-forward (FK probe, tinh
    # modules/collision survey 2026-08-26), so L max-forward
    # (sin = -1) lands there: 3*pi/2 - 2*pi*0.794 = -0.277.
    # Cross-check: L heel strike (phi 0.294) -> theta = pi/2 -> L
    # max-back. Contralateral both ways.
    phase_offset: float = -0.277
    pitch_joints: tuple = ("L_SHOULDER_PITCH", "R_SHOULDER_PITCH")
    yaw_joints: tuple = ("L_SHOULDER_YAW", "R_SHOULDER_YAW")


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
    # Clock rate DURING a hold. 0 would be a freeze-frame, and a
    # freeze-frame is dynamically infeasible (a mid-swing foot at
    # ~1.5 m/s cannot track dy_des = 0 in one step) -- slow motion
    # keeps the reference feasible. Lag absorbed per hold =
    # (1 - alpha) * max_hold_s.
    contact_gate_hold_alpha: float = 0.4
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
    # v6 arm-swing overlay; None = reference arms unchanged.
    arm_swing: ArmSwingOverlayCfg | None = None