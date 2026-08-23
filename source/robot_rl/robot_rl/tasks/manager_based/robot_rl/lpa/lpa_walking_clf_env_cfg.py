# TINH LPA walking CLF env (bead tinh-wf62). Port of
# g1_walking_clf_env_cfg.py onto the LPA embodiment:
#
#   - robot: LPA_MINIMAL_CFG (21 joints, HEJ actuators, stand init)
#   - reference: OUR trajopt gait library (walk_stack solutions
#     exported by tinh's export_clfrl_library — 3 speeds, vel_x
#     PENDULUM STOMP keeper — 4-domain, vel 0.5585), local path
#   - outputs: joints + CORE/L_ANKLE/R_ANKLE bodies (LPA has no
#     wrists in the walking model, no ankle roll, and no com outputs
#     in the library — use_com stays False)
#   - velocity command clamped to the library's span, forward-only
#     (no lateral/yaw gaits solved yet), so the G1 sideways/turning
#     heuristic is disabled
#   - events retargeted: trunk link is TORSO_ROLL (G1: waist_yaw_link)

import math

from isaaclab.utils import configclass
from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import (
    HumanoidEnvCfg,
    HumanoidCommandsCfg,
    HumanoidEventsCfg,
)
from robot_rl.tasks.manager_based.robot_rl import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from robot_rl.assets.robots.lpa_21j import LPA_MINIMAL_CFG, LPA_ACTION_SCALE  # isort: skip
from ..mdp.commands.velocity_commands_cfg import VelocityTrackingCommandCfg
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.trajectory_cmd_cfg import TrajectoryCommandCfg

# The G1 trajopt obs/reward stacks are name-generic (they key off the
# command tensors, not robot names) — reuse them directly.
from ..g1.g1_trajopt_obs import G1TrajOptObservationsCfg
from ..g1.g1_trajopt_reward import G1TrajOptCLFRewards

##
# Tracked outputs: 21 joints + 3 bodies (pos xyz + ori xyz each).
# Must be a subset of the library's outputs (which also carry ori_w
# for quats and body velocities).
##
_LPA_JOINTS = (
    "L_HIP_PITCH", "L_HIP_ROLL", "L_HIP_YAW", "L_KNEE", "L_ANKLE",
    "R_HIP_PITCH", "R_HIP_ROLL", "R_HIP_YAW", "R_KNEE", "R_ANKLE",
    "WAIST_YAW", "TORSO_PITCH", "TORSO_ROLL",
    "L_SHOULDER_PITCH", "L_SHOULDER_ROLL", "L_SHOULDER_YAW", "L_ELBOW",
    "R_SHOULDER_PITCH", "R_SHOULDER_ROLL", "R_SHOULDER_YAW", "R_ELBOW",
)
_LPA_BODIES = ("CORE", "L_ANKLE", "R_ANKLE")

WALKING_Q_weights = {}
WALKING_R_weights = {}
for _b in _LPA_BODIES:
    for _ax in ("pos_x", "pos_y", "pos_z", "ori_x", "ori_y", "ori_z"):
        WALKING_Q_weights[f"{_b}:{_ax}"] = [1.0, 1.0]
        WALKING_R_weights[f"{_b}:{_ax}"] = [0.05]
for _j in _LPA_JOINTS:
    WALKING_Q_weights[f"joint:{_j}"] = [1.0, 1.0]
    WALKING_R_weights[f"joint:{_j}"] = [0.05]
# Shoulder roll/yaw 4x: the first walker pumped its shoulders
# vertically for momentum regulation instead of tracking the
# reference's pitch counter-swing (user viewer note 2026-08-02).
# Pitch stays at 1.0 — that's the DOF that SHOULD swing.
for _j in ("L_SHOULDER_ROLL", "R_SHOULDER_ROLL",
           "L_SHOULDER_YAW", "R_SHOULDER_YAW"):
    WALKING_Q_weights[f"joint:{_j}"] = [4.0, 4.0]
# STYLE CHANNELS (user viewer notes 2026-08-06/07): each washed-out
# aspect has a precise output channel. Weights are [position,
# velocity] against the phase-indexed reference.
#
# THE DWELL lives in CORE:pos_x — the base's forward profile vs phase
# (hold through double support, advance through the swing); its
# velocity slot rewards the stop-go rhythm itself. The vault/dip live
# in CORE:pos_z.
# [6,6] -> [10,10] (user 2026-08-07: RL still compresses the dwell;
# the reference's is right — track it harder).
WALKING_Q_weights["CORE:pos_x"] = [10.0, 10.0]
WALKING_Q_weights["CORE:pos_z"] = [4.0, 2.0]
# CHEST UPRIGHT = base roll + TORSO_ROLL. Upweighting only the joint
# (round 1) let the policy roll the PELVIS off-reference instead —
# both sides of the sum now carry weight.
# Round 3 (user 2026-08-07): [8,4] on BOTH roll components let the
# policy park the PELVIS level and swing the chest anti-phase with
# the tracked joint — the upright ask is the SUM, now rewarded
# directly (chest_upright below). Component weights back to mild.
WALKING_Q_weights["CORE:ori_x"] = [2.0, 2.0]
WALKING_Q_weights["joint:TORSO_ROLL"] = [4.0, 2.0]
# ANKLE FLIP is a velocity-shaped event (toe-off/heel-strike snaps).
for _j in ("L_ANKLE", "R_ANKLE"):
    WALKING_Q_weights[f"joint:{_j}"] = [4.0, 8.0]
# WAIST GUARD (2026-08-18, third recurrence of the twist exploit):
# the yaw_vel reward pays for commanded yaw rate, the reference
# cannot turn, and the cheapest fake yaw-rate is the waist. The
# guard trio (this weight + waist_twist fence + waist_quiet penalty
# in the BASE cfg below) was validated across gt15-gt18 — fence
# never fired, penalty ~-0.04 — and belongs to EVERY LPA env, not
# an env-name allowlist. It was graphturn-only while the ramp env
# re-discovered the exploit under 100% heading control.
WALKING_Q_weights["joint:WAIST_YAW"] = [8.0, 4.0]


##
# Commands
##
@configclass
class LpaGaitLibraryCommandsCfg(HumanoidCommandsCfg):
    """Gait library commands: OUR trajopt walk library, local path."""
    traj_ref = TrajectoryCommandCfg(
        contact_bodies=[".*_ANKLE"],
        manager_type="library",
        # Local library (copied from tinh's
        # /opt/tinh/data/trajopt/clfrl_lib) — no HF fetch.
        hf_repo=None,
        path="lpa_lib/trajectories/walk_forward",
        conditioner_generator_name="base_velocity",
        # 21 joints + 3 bodies * (pos 3 + ori 3) = 39 tracked outputs
        # (library carries more — ori_w + velocities; the cmd orders
        # its own outputs from the Q keys). VERIFY on first env build.
        num_outputs=39,
        Q_weights=WALKING_Q_weights,
        R_weights=WALKING_R_weights,
        hold_phi_threshold=0.1,
        # No sideways/turning heuristic: the library is forward-only
        # PENDULUM STOMP keeper (tinh-lpa-clfrl.5, user-approved
        # 2026-08-05): 4-domain, 0.44 stride, apex vault over the
        # ankle, brief momentum dwell, elbow secondary motion,
        # TRUE SONAX limits + chest counter-roll, synced punch,
        # feet-track-x, ankle flip. vel_x 0.5612 (pendulum7: +torso lean 0.10, stomp nod).
        heuristic_func=None,
        phasing_boundaries=4,
    )

    base_velocity = VelocityTrackingCommandCfg(
        asset_name="robot",
        resampling_time_range=(7.0, 10.0),
        rel_standing_envs=0.05,
        rel_closed_loop=0.5,
        rel_closed_loop_yaw=0.25,
        rel_open_loop=0.2,
        debug_vis=True,
        ranges=VelocityTrackingCommandCfg.VelRanges(
            # Centered on the library's natural vel_x — commanding
            # faster pays the policy to compress the dwell.
            # pendulum7 (lean+nod, dwell 0.315 s): vel_x 0.5612.
            # Multi-gait span (tinh-lpa-clfrl.7.7): full range down
            # to standing; the nearest-gait conditioner partitions
            # commands among straight (0.5612,0,0) and the two
            # turn-in-place gaits (0,0,+-0.2888).
            lin_vel_x=(0.0, 0.58),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.29, 0.29),
            heading=(-math.pi, math.pi),
            y_pos_offset=(-0.5, 0.5),
            y_kp=(1.2, 1.8),
            y_kd=(0.2, 0.4),
        ))


##
# Events
##
@configclass
class LpaWalkingEventsCfg(HumanoidEventsCfg):
    reset_on_ref = EventTerm(
        func=mdp.reset_on_reference,
        mode="reset",
        params={"command_name": "traj_ref",
                "base_frame_name": "CORE",
                "conditioner_command_name": "base_velocity",
                # library tops out at 0.74 m/s — no running traj
                "special_val": 10.0,
                # 0.5: at 0.9 the policy practiced continuing but never the
                # LAUNCH — the displacement gate starts from rest and
                # measured 0.18 m/s vs 0.38 m/s during training.
                "rel_envs_on_ref": 0.5,
                "joint_add_range": [-0.1, 0.1]}
    )

    joint_friction_params = EventTerm(
        func=mdp.randomize_joint_parameters_multi_friction,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "static_friction_distribution_params": (0.3, 1.6),
                "dynamic_friction_distribution_params": (0.3, 1.2),
                "viscous_friction_distribution_params": (0.01, 0.1),
                "operation": "add"},
    )

    joint_armature_params = EventTerm(
        func=mdp.randomize_joint_parameters_multi_friction,
        mode="startup",
        # Wider than G1's (0.95, 1.05): LPA armature values are
        # datasheet first guesses until tinh-lpa-sysid.
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "armature_distribution_params": (0.8, 1.2),
                "operation": "scale"},
    )

    gain_randomization = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform"
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="TORSO_ROLL"),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    reset_base = None
    reset_robot_joints = None


##
# Rewards
##
@configclass
class LpaWalkingRewardCfg(G1TrajOptCLFRewards):
    torque_lims = RewTerm(func=mdp.torque_limits, weight=-1.0)

    base_pos = RewTerm(
        func=mdp.base_pos_reward, weight=1.0,
        params={"command_name": "traj_ref", "sigma": 0.2})
    base_ori = RewTerm(
        func=mdp.base_ori_reward, weight=1.0,
        params={"command_name": "traj_ref", "sigma": 0.3})
    base_lin_vel = RewTerm(
        func=mdp.base_lin_vel_reward, weight=1.0,
        params={"command_name": "traj_ref", "sigma": 0.3})
    base_ang_vel = RewTerm(
        func=mdp.base_ang_vel_reward, weight=1.0,
        params={"command_name": "traj_ref", "sigma": 1.0})

    joint_pos = RewTerm(
        func=mdp.joint_pos_reward, weight=1.0,
        params={"command_name": "traj_ref", "sigma": 0.2 * math.sqrt(21)})
    joint_vel = RewTerm(
        func=mdp.joint_vel_reward, weight=1.0,
        params={"command_name": "traj_ref", "sigma": 3.0 * math.sqrt(21)})

    # 3 tracked bodies (G1 used 4 incl. wrists).
    body_pos = RewTerm(
        func=mdp.body_pos_reward, weight=1.0,
        params={"command_name": "traj_ref", "sigma": 0.1 * math.sqrt(3)})
    body_ori = RewTerm(
        func=mdp.body_ori_reward, weight=1.0,
        params={"command_name": "traj_ref", "sigma": 0.2 * math.sqrt(3)})
    body_lin_vel = RewTerm(
        func=mdp.body_lin_vel_reward, weight=1.0,
        params={"command_name": "traj_ref", "sigma": 1.0 * math.sqrt(3)})
    body_ang_vel = RewTerm(
        func=mdp.body_ang_vel_reward, weight=1.0,
        params={"command_name": "traj_ref", "sigma": 0.5 * math.sqrt(3)})

    # Weight 3.0 / std 0.25 (was 1.0/0.5): the first trained policy
    # found the standing optimum — the CLF reference is anchored to
    # the robot's own stance frame, so standing + gait-rhythm arm
    # motion tracked 'well' while translating 0.35 m in 12 s. At
    # std 0.5 standing only cost ~0.5 reward; at 3.0/0.25 it costs
    # ~2.9 — walking must pay.
    # exp kernel back to 3.0: at 10.0 the policy exploited the
    # INSTANTANEOUS kernel by rocking in place at the commanded speed
    # (reward 231, displacement 0.44 m/12 s). Fine-shaping only.
    # Weight 3.0/std 0.25 actively ERASED the stomp dwell: a constant
    # velocity command tracked tightly pays the policy to smooth out
    # the reference's stop-and-go rhythm (user viewer note
    # 2026-08-06). Softened to an average-speed anchor; the CLF
    # tracking carries the within-cycle rhythm and forward_progress
    # (integral) is profile-indifferent.
    # Chest stays LEVEL regardless of how pelvis/SONAX split the roll
    # (assignment-free sum penalty; see mdp.body_upright_roll).
    chest_upright = RewTerm(
        func=mdp.body_upright_roll, weight=-8.0,
        params={"asset_cfg": SceneEntityCfg(
            "robot", body_names=["TORSO_ROLL"])})

    xy_vel = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0,
        params={"command_name": "base_velocity", "std": 0.45})

    # The ungameable translation incentive: signed linear forward
    # velocity integrates to net displacement — rocking scores zero.
    # 10.0 (5.0 reached 0.18 m/s real travel): linear progress can't
    # be gamed, so it can dominate safely.
    progress = RewTerm(
        func=mdp.forward_progress, weight=10.0,
        params={"command_name": "base_velocity"})

    # Force actual stepping: the CLF tracking terms are all anchored
    # to the robot's own stance frame, so nothing but velocity pays
    # for translation — weight 3.0 alone got 1.5 m/12 s shuffling.
    # Air-time reward makes feet LEAVE the ground under a nonzero
    # command (G1 vanilla recipe, LPA ankle bodies).
    # Anti-drag round (user: 'one foot dragged behind the other'):
    # feet_slide penalizes foot velocity WHILE in contact — dragging
    # is sliding contact by definition; air time doubled so both
    # feet must actually swing.
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_ANKLE"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ANKLE"),
        },
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            # Desired swing air time: the gait library's half-step
            # swing is ~0.4 s (T=0.5 s domains).
            "threshold": 0.4,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_ANKLE"),
        },
    )
    yaw_vel = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5})

    clf_reward = None


@configclass
class LpaWalkingCLFEnvCfg(HumanoidEnvCfg):
    """LPA walking on the TINH trajopt gait library."""
    commands: LpaGaitLibraryCommandsCfg = LpaGaitLibraryCommandsCfg()
    observations: G1TrajOptObservationsCfg = G1TrajOptObservationsCfg()
    rewards: LpaWalkingRewardCfg = LpaWalkingRewardCfg()
    events: LpaWalkingEventsCfg = LpaWalkingEventsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.actions.joint_pos.scale = LPA_ACTION_SCALE

        self.scene.robot = LPA_MINIMAL_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot")

        # Forward-only, inside the library's conditioner span.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.58)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.29, 0.29)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.resampling_time_range = (4.0, 8.0)
        self.commands.gait_period = None

        self.events.push_robot.params["velocity_range"] = {
            "x": (-1, 1),
            "y": (-1, 1),
            "roll": (-0.4, 0.4),
            "pitch": (-0.4, 0.4),
            "yaw": (-0.4, 0.4),
        }
        self.events.base_com.params["asset_cfg"].body_names = ["TORSO_ROLL"]
        # Base cfg's foot-friction event targets G1 ankle links.
        self.events.randomize_ground_contact_friction.params[
            "asset_cfg"].body_names = [".*_ANKLE"]

        # The G1 obs/reward stacks carry two name-bearing terms:
        # the critic's contact_state sensor and the undesired-contact
        # penalty's everything-but-feet regex.
        self.observations.critic.contact_state.params[
            "sensor_cfg"].body_names = ".*_ANKLE"
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            r"^(?!L_ANKLE$)(?!R_ANKLE$).+$"
        ]
        self.events.base_com.params["com_range"] = {
            "x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)}
        self.events.base_external_force_torque = None

        self.rewards.holonomic_constraint.params["command_name"] = "traj_ref"
        self.rewards.holonomic_constraint_vel.params["command_name"] = "traj_ref"
        self.rewards.dof_torques_l2.weight = -1.0e-5
        self.rewards.clf_decreasing_condition.params = {
            "command_name": "traj_ref",
            "alpha": 0.5,
            "eta_max": 0.25,
            "eta_dot_max": 0.3,
        }
        self.rewards.clf_decreasing_condition.weight = -1

        # LPA stands at 1.06 m (G1: 0.785) — raise the fall floor.
        self.terminations.base_height.params["minimum_height"] = 0.6

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        # Waist guard trio, base-level (see WALKING_Q_weights note).
        from isaaclab.managers import TerminationTermCfg as _DoneT
        from isaaclab.managers import RewardTermCfg as _RewT
        from isaaclab.managers import SceneEntityCfg as _SceneC
        import isaaclab.envs.mdp as _ilmdp
        self.terminations.waist_twist = _DoneT(
            func=mdp.waist_twist, params={"limit": 2.2})
        self.rewards.waist_quiet = _RewT(
            func=_ilmdp.joint_deviation_l1,
            weight=-1.5,
            params={"asset_cfg": _SceneC(
                "robot", joint_names=["WAIST_YAW"])})


@configclass
class LpaWalkingCLFEnvCfg_PLAY(LpaWalkingCLFEnvCfg):
    """Playback / eval variant."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5

        # Close-up tracking camera for eval videos (follows env_0's
        # robot root).
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (4.5, 4.5, 2.2)
        self.viewer.lookat = (0.0, 0.0, 0.8)
        self.observations.policy.enable_corruption = False
        self.scene.terrain.size = (3, 3)
        self.scene.terrain.border_width = 0.0

        self.episode_length_s = 8.0

        self.events.randomize_ground_contact_friction = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.gain_randomization = None


@configclass
class LpaWalkingCLFSkillEnvCfg(LpaWalkingCLFEnvCfg):
    """Behavior-graph SKILL variant (tinh-lpa-clfrl.8.5d): the
    multi-skill library (gaits + laser enter/exit episodic edges),
    skill one-hot + param channels appended to BOTH obs groups, and
    the graph-skill curriculum driving stand -> enter -> hold ->
    exit -> locomotion sequences. Resume pendulum checkpoints via
    scripts/pad_checkpoint_obs.py (+5 obs channels each group).
    SEPARATE env id — the default lpa_walking_clf stays untouched
    (pendulum close-out replays depend on its obs layout)."""

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.managers import ObservationTermCfg as _ObsTerm

        self.commands.traj_ref.path = "lpa_lib/trajectories/skill_smoke"
        self.commands.traj_ref.skill_slots = [
            "locomotion", "walking", "laser"]
        self.commands.traj_ref.param_channels = [
            "azimuth", "wall_height"]

        for group in (self.observations.policy,
                      self.observations.critic):
            group.skill_onehot = _ObsTerm(
                func=mdp.skill_onehot,
                params={"command_name": "traj_ref"})
            group.skill_params = _ObsTerm(
                func=mdp.skill_params,
                params={"command_name": "traj_ref"})

        self.events.graph_skills = EventTerm(
            func=mdp.graph_skill_sampler,
            mode="interval",
            interval_range_s=(0.5, 0.5),
            params={"command_name": "traj_ref",
                    "enter_name": "laser_enter",
                    "exit_name": "laser_exit",
                    "p_enter": 0.5})


@configclass
class LpaWalkingCLFRampEnvCfg(LpaWalkingCLFEnvCfg):
    """Ramp walking (tinh-lpa-ramp.5): sloped terrain + per-env
    reference selection by slope.

    Slope is TERRAIN, not a command — the velocity conditioner cannot
    separate the ramp gaits (all vel_x 0.27-0.35, vel_yaw 0), so every
    env is pinned to the gait matching its terrain column and the
    signed slope is appended to both obs groups. Built on the plain
    walking env (NOT the skill/graph envs): ramp is a static
    condition, no traversal state machine needed.
    """

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.managers import ObservationTermCfg as _ObsTerm
        from .lpa_ramp_terrain import RAMP_COLUMNS, RAMP_TERRAINS_CFG

        # Sloped terrain replaces the flat plane.
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = RAMP_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None

        self.commands.traj_ref.path = "lpa_lib/trajectories/ramp_train"

        # Fall detection must be TERRAIN-RELATIVE here: the stock
        # world-frame height check kills every uphill env at spawn
        # (they sit at the bottom of an inverted pyramid, absolute
        # z 0.30-0.39 vs a 0.6 threshold).
        from isaaclab.managers import TerminationTermCfg as _DoneT
        self.terminations.base_height = _DoneT(
            func=mdp.base_height_above_feet,
            params={"minimum_height": 0.6})

        # Slope observation (+1 channel per group): the policy must
        # know which ramp it is on. Resume from a flat checkpoint via
        # scripts/pad_checkpoint_obs.py --extra 1.
        for group in (self.observations.policy,
                      self.observations.critic):
            group.terrain_slope = _ObsTerm(
                func=mdp.terrain_slope,
                params={"command_name": "traj_ref"})

        # 0.1 s, not 0.5: after every reset an env is briefly on
        # locomotion (-1), i.e. tracking the FLAT gait while standing
        # on a ramp. Steep-slope envs reset often early in training,
        # so that gap is not negligible.
        # FRICTION FLOOR: the inherited randomization spans down to
        # mu 0.3 — fine on flat, disqualifying on a 12.6 deg slope
        # (tan 0.22): push-off shear exceeds static grip, so
        # low-draw envs physically cannot climb and the policy
        # learns climbing is a lottery (ramp4: up-ratios REGRESSED
        # over a full run while flats/downs flourished).
        self.events.randomize_ground_contact_friction.params[
            "static_friction_range"] = (0.7, 1.6)
        self.events.randomize_ground_contact_friction.params[
            "dynamic_friction_range"] = (0.6, 1.2)

        # CLIMB REWARD: pay for stance-foot ground-height gain in
        # the slope's direction — the quantity the climb gate
        # measures and the one thing the CLF structurally cannot
        # (per-domain re-anchoring wipes altitude error each step;
        # the turn arc's leak, in z). At 0.05 cap and 50 Hz this
        # tops out ~= the progress term's scale.
        from isaaclab.managers import RewardTermCfg as _RewT
        self.rewards.ground_climb = _RewT(
            func=mdp.ground_climb,
            weight=40.0,
            params={"command_name": "traj_ref"})

        self.events.ramp_refs = EventTerm(
            func=mdp.ramp_ref_sampler,
            mode="interval",
            interval_range_s=(0.1, 0.1),
            params={"command_name": "traj_ref",
                    "ref_names": [r for _, r, _ in RAMP_COLUMNS],
                    "slope_degs": [d for _, _, d in RAMP_COLUMNS]})

        # Face UP THE FALL LINE. The inclined plane rises along +x,
        # and the reference gaits walk straight forward; with heading
        # randomised the robot traverses the slope instead of
        # climbing it (that, plus radial pyramid terrain, is why the
        # first ramp policy walked 10 m and gained ~0 m).
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # ...and heading control must apply to EVERY env. The
        # inherited mode split (open_loop 0.2 / closed 0.5 /
        # closed_yaw 0.25 / standing 0.05) leaves half the
        # population free to drift, and on a slope downhill is the
        # path of least resistance — the 800-iter A/B showed climb
        # ratios DEGRADING (up12 -0.56 -> -0.74) while distance
        # improved: the policy was learning to walk DOWN the ramps.
        self.commands.base_velocity.rel_open_loop = 0.0
        self.commands.base_velocity.rel_closed_loop = 1.0
        self.commands.base_velocity.rel_closed_loop_yaw = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.y_pos_offset = (0.0, 0.0)

        # Chase camera: on 40 m ramp patches a world-fixed viewer
        # renders the robot a few pixels tall, which makes the 6-up
        # useless for judging gait.
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (5.0, 5.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.8)

        # A climb needs room: the flat 8 s episode barely covers a
        # few strides (and was THE bug that broke turn training —
        # see tinh-lpa-clfrl.7.7.v3).
        self.episode_length_s = 20.0


@configclass
class LpaWalkingCLFGraphTurnEnvCfg(LpaWalkingCLFSkillEnvCfg):
    """Graph-turn retrain (pendulum9b post-mortem): turning as a
    graph traversal — walk -> handed walk_to_stand -> turn cycle
    entered at its stand -> stand_to_walk -> locomotion, with the
    velocity command pinned to the active skill."""

    def __post_init__(self):
        super().__post_init__()
        self.events.graph_skills = EventTerm(
            func=mdp.graph_turn_sampler,
            mode="interval",
            interval_range_s=(0.5, 0.5),
            # p_turn 0.02 -> 0.10 (gt16 forensics): turn attempts
            # were so rare the policy never learned them — it
            # twisted, died on the fence, and the termination stat
            # rounded to 0.0000 in training logs.
            params={"command_name": "traj_ref",
                    "p_turn": 0.10,
                    "turn_periods": 2.0})
        self.events.nan_tripwire = EventTerm(
            func=mdp.graph_nan_tripwire,
            mode="interval",
            interval_range_s=(0.5, 0.5),
            params={"command_name": "traj_ref"})
        from isaaclab.managers import TerminationTermCfg as _DoneTerm
        self.terminations.runaway = _DoneTerm(
            func=mdp.runaway_dynamics,
            params={"base_vel_limit": 10.0, "joint_vel_limit": 60.0})
        # Waist-twist exploit (gt14, twist probe 2026-08-14): the
        # chest is not a tracked body and chest_upright ignores yaw,
        # so at Q=1 the policy walked with the torso twisted 90-180
        # deg (64/64 envs at the +-pi joint limit). Same escalation
        # as the shoulder pump fix (2026-08-02): style channel up,
        # plus a hard termination far outside honest tracking
        # (references keep the waist within ~0.3 rad).
        # Waist guards now live in the BASE cfg (Q table + fence +
        # penalty) — nothing graphturn-specific remains here.
        # Full-body imitation during explicit segments (user
        # direction 2026-08-16 after the gt16 folding exploit):
        # segments are choreography — every joint channel x4
        # (Q x16), eased in over 0.3 s.
        self.commands.traj_ref.anchored_yaw = True
        self.commands.traj_ref.imitation_gain = 4.0
        self.commands.traj_ref.imitation_ease_s = 0.3
        # THE TURN BUG (found 2026-08-17): the sampler's turn timer
        # is _turn_until = t_now + turn_periods x cycle, with t_now
        # measured in EPISODE time. A turn needs 4.8 s and only
        # starts after walking + a stop segment — so inside the 8 s
        # episode inherited from the walking env, any turn beginning
        # after t~3.2 s is reset before its timer fires. _STARTING
        # was never reached in ANY run gt1..gt18: no policy has ever
        # experienced a COMPLETED turn. 24 s fits walk + stop + turn
        # + start with room for a second sequence.
        self.episode_length_s = 24.0

        # GRADED waist penalty (gt16 instrumented gate: 1-3 envs hit
        # the 2.2 fence EVERY step under turn demand — death gives
        # avoidance, not guidance). Continuous gradient toward the
        # honest pelvis-stepping turn; the fence stays as backstop.
        from isaaclab.managers import RewardTermCfg as _RewTerm
        from isaaclab.managers import SceneEntityCfg as _SceneCfg
        import isaaclab.envs.mdp as _ilmdp
        self.rewards.waist_quiet = _RewTerm(
            func=_ilmdp.joint_deviation_l1,
            weight=-1.5,
            params={"asset_cfg": _SceneCfg(
                "robot", joint_names=["WAIST_YAW"])})


# Observation history, shared by every env that trains or plays a
# history policy so the two cannot drift into different widths.
#
# 10 frames at our 50 Hz (sim dt 0.005 x decimation 4) = 0.20 s, which
# is SONIC's own decoder window at the same control rate (NVIDIA GEAR,
# arXiv 2511.07820). 5 frames was tried first and is too short on two
# counts: 0.10 s cannot contain a humanoid touchdown->loading->push-off
# event (stance is ~0.3-0.5 s), and the one direct humanoid ablation
# (PRIOR, arXiv 2603.18979 Table V) drops 27% between H=10 and H=6.
#
# Policy obs 74 -> 640. ACTOR ONLY: CriticCfg subclasses PolicyCfg but
# holds its own term instances, so the critic keeps its 249-wide
# privileged view and needs no checkpoint remap.
#
# Commands and the phase clock are excluded on purpose -- both are
# externally supplied and deterministic, so their history is redundant.
LPA_OBS_HISTORY = 10
_HISTORY_TERMS = ("base_ang_vel", "projected_gravity", "joint_pos",
                  "joint_vel", "actions")


def apply_obs_history(observations, history: int = LPA_OBS_HISTORY):
    """Give the POLICY group `history` frames of proprioception."""
    for name in _HISTORY_TERMS:
        term = getattr(observations.policy, name, None)
        if term is not None:
            term.history_length = history


@configclass
class LpaWalkingCLFRoughEnvCfg(LpaWalkingCLFEnvCfg):
    """Flat-ground walking, trained on MILDLY ROUGH ground for
    robustness (user 2026-08-21: "handle uneven terrain, even if we
    are only walking on flat ground").

    Deliberately the plain walking env plus terrain -- no skill slots,
    no graph sampler, same walk_forward library. That keeps the
    OBSERVATION LAYOUT identical to the flat walking policy, so this
    resumes from an existing checkpoint directly with no obs padding,
    which is the whole reason to keep it this narrow.

    Terrain is the first rung of the standard legged-RL ladder and
    only its low end (0.01-0.05 m noise, 30% flat in the mix). See
    lpa_rough_terrain for why the slope/stair/obstacle rungs are NOT
    here: slopes have their own env, and stairs are a behaviour rather
    than free robustness.
    """

    def __post_init__(self):
        super().__post_init__()
        from .lpa_rough_terrain import ROUGH_TERRAINS_CFG

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG

        # CONTACT-GATED REFERENCE CLOCK (rough1-8 forensics).
        #
        # The measured failure is a TIME-INDEXING pathology, not a
        # terrain-magnitude one: 5 cm rubble gave 50% style retention
        # and 2 cm smooth undulation gave 51%. Amplitude-independence
        # is the signature of a saturated exp(-KAPPA*V/sigma) -- once
        # the reference has walked away from a contact-delayed robot,
        # it does not matter by how much.
        #
        # Hold the reference while the ground has not arrived; give the
        # time back when a foot lands early. Bounded on both sides, and
        # the hold has a hard ceiling so a fallen robot cannot freeze
        # its own clock forever.
        # PARKED 2026-08-23 after rough12: with the gate on, falls at
        # matched iteration +450 were 0.988 vs the no-gate control's
        # 0.627 and episode length 141 vs 429 -- ~4x slower recovery,
        # same checkpoint and seed. A 64-env probe refuted the alpha-
        # flicker mechanism (toggle rate 0.02/step, vdot blackout
        # 5.7%), so the harm has NO confirmed mechanism after six
        # versions. Machinery, telemetry and unit tests stay; do not
        # re-enable without a mechanism and a matched-control read.
        self.commands.traj_ref.contact_gate = False

        # Proprioceptive history -- see apply_obs_history above for
        # why, and why it is the actor only.
        apply_obs_history(self.observations)

        # (The 0.58 -> 0.29 speed reduction was prepared for rough6
        # and pulled back out: observation history is the change
        # under test now, and shipping both at once would repeat the
        # two-variables mistake that made rough4 uninterpretable.
        # Speed remains available as a later knob if history alone is
        # not enough.)

        # SPLIT STYLE REWARD (rough1/2/3 forensics).
        #
        # Three runs collapsed to ~50% of their opening style within
        # ~950 iterations, and terrain amplitude made NO difference:
        # 5 cm rubble gave 50%, 2 cm smooth undulation gave 51%. What
        # is invariant is that ANY terrain forces the legs off a
        # flat-ground reference.
        #
        # The combined joint_pos term puts all 21 joints under one
        # exp(-KAPPA*V/sigma). Once the legs make V large, the
        # gradient for ARM tracking is suppressed by that same
        # exponential no matter how mild the ground is -- which is
        # exactly the amplitude-independence observed. The arms then
        # settle wherever balance prefers: measured 2.53x position
        # drift against the legs' 1.87x, worst at SHOULDER_ROLL
        # (4.1x), the arms-out inertia-widening posture.
        #
        # Splitting gives each group its own exponential, so:
        #   sigma  = TOLERANCE, how far this group may deviate
        #   weight = PRIORITY, how hard it is pulled back
        # PURE SPLIT -- nothing but the shared exponential changes.
        #
        # The first attempt (rough4) widened the arm sigma to 0.3 at
        # the same time, to "permit" balance work. That made the
        # reward LESS sensitive near small errors, i.e. it pulled the
        # arms back more weakly, plausibly cancelling whatever the
        # split bought: arm drift came out at 1.58x against the
        # combined run's 1.50x, no better. Two variables, one run,
        # no conclusion.
        #
        # So: per-joint sigma convention preserved (0.2*sqrt(n), the
        # same normalisation the combined term used over its 21), and
        # weights proportional to joint count so each joint carries
        # the influence it had before. The ONLY difference from the
        # combined term is that each group now has its own
        # exponential and cannot have its gradient flattened by
        # another group's error. The combined term is
        # switched off; note IsaacLab SKIPS zero-weight terms
        # entirely, so Episode_Reward/joint_pos will read 0.0 for
        # this env -- use `style_gate --metric arms` here.
        self.rewards.joint_pos.weight = 0.0
        from isaaclab.managers import RewardTermCfg as _RewGrp
        for _grp, _n in (("legs", 10), ("arms", 8), ("torso", 3)):
            _s, _w = 0.2, _n / 21.0
            setattr(self.rewards, f"joint_pos_{_grp}", _RewGrp(
                func=mdp.joint_pos_group_reward, weight=_w,
                params={"command_name": "traj_ref",
                        "sigma": _s * math.sqrt(_n), "group": _grp}))

        # Terrain-relative fall detection. The stock check is
        # world-frame and misreads a robot standing in a dip -- the
        # same failure that killed every uphill env in the ramp work.
        from isaaclab.managers import TerminationTermCfg as _DoneT
        self.terminations.base_height = _DoneT(
            func=mdp.base_height_above_feet,
            params={"minimum_height": 0.6})


@configclass
class LpaWalkingCLFHistEnvCfg(LpaWalkingCLFEnvCfg):
    """FLAT walking with proprioceptive history — the play/export env
    for history policies.

    Exists because a history policy cannot be played in the plain
    flat env at all: it has a 640-wide actor and that env presents
    74 observations. Without this, a rough-terrain history run has no
    flat ablation video and no exportable ONNX, and it would surface
    only at close-out time, hours after training finished — the same
    shape of failure as the missing SIM_ENVIRONMENTS entry.

    Flat ground on purpose. This is the surface the robot ships on,
    so it answers "did terrain training cost us anything at rate",
    which is the question the whole rough programme is judged on.

    The plain `lpa_walking_clf` env stays as it is, because the
    74-obs pendulum9b baseline still has to play somewhere for the
    like-for-like comparison.
    """

    def __post_init__(self):
        super().__post_init__()
        apply_obs_history(self.observations)
