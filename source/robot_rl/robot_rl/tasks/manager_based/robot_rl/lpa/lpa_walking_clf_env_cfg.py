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
# STYLE CHANNELS 4x (user viewer note 2026-08-06: the trained policy
# washed out the dwell, chest counter-roll, and ankle flip): the
# choreography lives in TORSO_ROLL (SONAX counter-roll) and the
# ankles (toe-off/heel-strike) — default weight let the policy
# sacrifice them first.
for _j in ("TORSO_ROLL", "L_ANKLE", "R_ANKLE"):
    WALKING_Q_weights[f"joint:{_j}"] = [4.0, 4.0]


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
        # feet-track-x, ankle flip. vel_x 0.5964.
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
            lin_vel_x=(0.58, 0.62),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
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
        self.commands.base_velocity.ranges.lin_vel_x = (0.58, 0.62)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
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
