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
from robot_rl.assets.robots.lpa_21j import (  # isort: skip
    LPA_MINIMAL_CFG, LPA_ACTION_SCALE, LPA_ARM_ROM, LPA_LEG_ROM)
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

    # OVERHEAD TRAVELLER (automaton am-gj0). The LPA is never operated
    # off the rig for safety (user 2026-08-28), so training it unaided
    # trains against a machine that does not exist — and a HARDER one,
    # since the rig carries part of the weight. On the clad 110.5 kg
    # robot the reference gait already saturates knees, hips, ankle,
    # waist yaw and both shoulders at 100% of limit.
    #
    # NOT mdp.apply_external_force_torque: that helper never passes
    # is_global, so its wrench lands in the BODY frame and would tilt
    # with the torso. A rope pulls world-up. mdp.overhead_traveller
    # applies is_global=True.
    #
    # 15.4 kg is the same fraction of body weight (13.9%) that 10 kg
    # was on the 71.9 kg unclad robot.
    overhead_traveller = EventTerm(
        func=mdp.overhead_traveller,
        mode="reset",
        params={
            # WAIST_YAW, the rotor the ropes actually attach to --
            # not TORSO_ROLL, which sits +0.0253 x / +0.1186 z from it
            # in the zero pose. Applying the lift 12 cm high and 2.5 cm
            # forward of the rotor makes a moment the rig does not,
            # and the whole point of this term is the ankle moment.
            # MuJoCo (lpa_sim.build_lpa_mjcf) and trajopt
            # (walk_stack.constraints.traveller) both act at WAIST_YAW.
            "asset_cfg": SceneEntityCfg("robot", body_names="WAIST_YAW"),
            "lift_kg": 15.4,
        },
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

        # SURFACE: butyl tread on plywood (user, 2026-08-28). The BOM
        # confirms the feet are butyl (TREAD, 8x, 1.79 kg) — a
        # high-hysteresis rubber, so grip is high and rebound is low.
        # The inherited G1 range (static 0.3-1.6, dynamic 0.3-1.2,
        # restitution 0-0.2) spans ice to tacky track and asks the
        # policy to be robust to surfaces it will never meet; the cost
        # is paid as conservatism on the one surface it does.
        # Rubber on dry wood runs ~0.6-0.95 static in the literature,
        # varying with finish, dust and humidity, so centre ~0.8 and
        # keep +-0.2 for that variation rather than for a different
        # planet. make_consistent already enforces dynamic <= static.
        # Restitution drops to near zero: butyl barely rebounds, and
        # the inherited 0.2 was flagged "consider upping this" for a
        # different robot on a different floor.
        self.events.randomize_ground_contact_friction.params[
            "static_friction_range"] = (0.6, 1.0)
        self.events.randomize_ground_contact_friction.params[
            "dynamic_friction_range"] = (0.5, 0.85)
        self.events.randomize_ground_contact_friction.params[
            "restitution_range"] = (0.0, 0.08)

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

        # HARD ARM ROM (bead tinh-ed92). Written into PhysX at startup
        # rather than into the URDF: the URDF has ~29 consumers and
        # feeds the trajopt NLP that produced the committed CLF
        # library, and a URDF change would not even reach a trajopt
        # solve (registry.py ASSIGNS custom_limits rather than
        # intersecting, and every LPA config declares all 21 joints).
        # write_joint_position_limit_to_sim also recomputes the soft
        # limits, so dof_pos_limits becomes meaningful for the first
        # time. Grouped by identical band to keep the term count down.
        # dof_pos_limits penalises against the SOFT limits, and
        # soft_joint_pos_limit_factor shrinks about the band MIDPOINT.
        # On the deliberately asymmetric shoulder-pitch band that puts
        # the soft edge at -2.763 while the USER-APPROVED throw
        # chambers to -2.8134 -- the reward would punish a behaviour
        # the user signed off on. Verified by reading the limits back
        # out of PhysX (tools: /tmp/rom_verify.py). Scope the penalty
        # to legs + waist, where the bands are symmetric and the soft
        # edge is honest; the ARMS are bounded by the HARD ROM above,
        # which PhysX enforces without any reward help.
        self.rewards.dof_pos_limits.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[
                ".*_HIP_.*", ".*_KNEE", ".*_ANKLE",
                "WAIST_YAW", "TORSO_PITCH", "TORSO_ROLL"])

        # Arms AND ankles: same startup-event mechanism, one event per
        # distinct band so joints sharing a band share a term.
        _rom_groups: dict[tuple[float, float], list[str]] = {}
        for _j, _band in {**LPA_ARM_ROM, **LPA_LEG_ROM}.items():
            _rom_groups.setdefault(_band, []).append(_j)
        for _i, (_band, _joints) in enumerate(sorted(_rom_groups.items())):
            setattr(self.events, f"arm_rom_{_i}", EventTerm(
                func=mdp.randomize_joint_parameters,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=_joints),
                    "lower_limit_distribution_params": (_band[0], _band[0]),
                    "upper_limit_distribution_params": (_band[1], _band[1]),
                    "operation": "abs",
                },
            ))

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

        # TERRAIN CURRICULUM (2026-08-25). Two halves, both needed:
        # lpa_rough_terrain now uses a difficulty-RESPECTING function
        # (isaaclab's random_uniform ignores difficulty, so the ladder
        # was five identical rows), and the promotion term below is
        # re-enabled here because the base LPA env sets it to None for
        # its flat plane and the rough env inherited that.
        #
        # Robots start on row 0 (5 mm ceiling, walkable by the incoming
        # flat keeper) and are promoted a row when they traverse past
        # half a patch. Roughness then tracks competence instead of
        # sitting at the 20 mm ceiling from iteration zero -- which is
        # what rough1..rough16 all did.
        from isaaclab.managers import CurriculumTermCfg as _CurrT

        from robot_rl.tasks.manager_based.robot_rl.mdp.curriculums import (
            curriculums as _curr,
        )

        self.curriculum.terrain_levels = _CurrT(func=_curr.terrain_levels)

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

        # Proprioceptive history: OFF for rough (2026-08-24).
        #
        # It was added here during the clear4 work and was one of the
        # hypotheses this hunt ELIMINATED -- a 10-frame history was
        # measured no better than none. It also inflates the actor
        # input from 74 to 695, so the flat keeper cannot be resumed
        # into this env without a checkpoint remap, adding a step and a
        # variable to every rough experiment.
        #
        # With it off, a rough run resumed from the flat keeper differs
        # by terrain and the collision asset alone, which is the
        # question actually being asked. Re-enable via
        # apply_obs_history(self.observations) if history is ever
        # tested on purpose, and remap the checkpoint with
        # scripts/expand_obs_history.py.

        # CLEARANCE REFERENCE (tinh-nwgy, user-approved 2026-08-23).
        # Same stomp solve plus a hard 4 cm swing-foot clearance floor
        # on the mid-swing nodes: the z=0 keeper leaves swing height
        # emergent, and a blind policy trips on bumps the reference
        # never asked it to clear (undesired_contacts 3.2x on rough).
        # Conditioner vel_x=0.5612 is identical to the keeper, so
        # command spans need no recentering. ROUGH ENV ONLY -- flat
        # keeps the keeper library for the like-for-like ablation.
        # walk_forward, NOT walk_forward_clear4 (2026-08-25).
        #
        # This line said clear4 while the class docstring above says
        # "same walk_forward library". rough15 resumed pendulum9b --
        # trained on walk_forward -- and tracked clear4 for 600
        # iterations before the swap was visible, making the run
        # uninterpretable: it changed terrain, the collision asset AND
        # the reference at once.
        #
        # clear4 is also not what its own docstring claims: it changed
        # the arms wholesale rather than only adding a swing-clearance
        # floor, its conditioner is 0.5612 against the keeper's 0.5964
        # (6.8% period difference), and refcheck flags 11 undeclared
        # channels, worst SHOULDER_YAW at 0.538 rad RMS.
        self.commands.traj_ref.path = "lpa_lib/trajectories/walk_forward"

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

        # ---- fall pricing (tinh-as7q.22) -------------------------
        # rough17 measured 50.7% of steps tipped past 60 deg from
        # vertical WITHOUT terminating: base_height is measured
        # above the FEET, so a face-down torso trailing its legs
        # still satisfies it. Prone time was therefore accruing
        # reward, and the backwards arm fling that buys a few more
        # prone steps was being reinforced. Both terms below end
        # the episode instead.
        #
        # bad_orientation is projected-gravity based -- no reference
        # outputs needed (mdp.base_orientation reads
        # cmd.ordered_output_names, which exists nowhere in this
        # tree, hence it is commented out in both g1 envs) and it
        # reads true tilt on a slope, which is what we want: the
        # torso balances against gravity, not the terrain normal.
        # 50 deg sits well clear of any commanded walking lean.
        self.terminations.bad_orientation = _DoneT(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(50.0)})

        # Chest/head hitting the ground. Only detectable as of the
        # 2026-08-25 collider pass -- TORSO_ROLL carries the torso
        # spheres and the head sphere; before that it had no
        # collision geometry at all, so a faceplant registered zero
        # contact force.
        self.terminations.torso_contact = _DoneT(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg(
                        "contact_forces", body_names="TORSO_ROLL"),
                    "threshold": 1.0})


@configclass
class LpaWalkingCLFRoughV2EnvCfg(LpaWalkingCLFRoughEnvCfg):
    """Rough walking with the fall priced and the CLF gradient restored.

    Four changes against LpaWalkingCLFRoughEnvCfg, all aimed at the
    same measured failure: rough17 spent 50.66% of steps beyond 60 deg
    from vertical, and the reward stack was partly PAYING for it.

    1. progress -> forward_progress_heading. The old term read
       body-frame X velocity, an axis that tilts with pitch, so a
       forward fall projected onto it positively. At weight 10 that
       was 43% of all positive reward.

    2. torso_pitch, new. body_upright_roll penalises roll only and its
       docstring says "Pitch is left free", so nothing priced a
       forward fall except the termination cliff. This is the ramp
       leading up to it, outside a 15 deg deadband so the reference's
       own forward lean stays free.

    3. clf_reward back on. Per Olkin et al. (arXiv 2605.01978) Thm 1/3,
       exponential stability follows from the DECAY condition, not the
       exponential -- so running without it was never a stability bug.
       But the decay term measures Vdot (is it converging) while the
       exponential measures V (how far off-orbit it is), and ours is
       clipped to [0,1]. With the exponential off AND the decay
       saturated, a robot deeply off-orbit converging slowly scores
       identically to one near the orbit converging slowly. That flat
       region is where the arm flail lives. Note the CLF-RL ablation
       (arXiv 2508.09354) compared tracking-only vs tracking+decay and
       never tested decay-only, which is what we were running.

    4. num_steps_per_env 24 -> 48 (see agents/rsl_rl_ppo_cfg.py).
       gamma*lam = 0.9405 gives an 11.3-step credit half-life = 0.23 s
       at 50 Hz, and the rollout was 24 steps = 0.48 s. The arm fling
       precedes torso contact by longer than that, so the terminal
       signal decayed before reaching the action that caused it.
    """

    def __post_init__(self):
        super().__post_init__()
        import math as _math
        from isaaclab.managers import RewardTermCfg as _RewT

        # 1. Progress in the yaw-only heading frame, gated on upright.
        self.rewards.progress = _RewT(
            func=mdp.forward_progress_heading, weight=10.0,
            params={"command_name": "base_velocity", "upright_gate": True})

        # 2. Price forward pitch, with a slope rather than a cliff.
        #    Same body and weight as chest_upright, which handles roll.
        self.rewards.torso_pitch = _RewT(
            func=mdp.body_upright_pitch, weight=-8.0,
            params={"deadband_deg": 15.0,
                    "asset_cfg": SceneEntityCfg(
                        "robot", body_names=["TORSO_ROLL"])})

        # 3. Restore the off-orbit DISTANCE signal. max_eta_err 0.8 is
        #    the G1TrajOptCLFRewards default and the largest of the
        #    values tried upstream; larger widens max_clf, which is
        #    what pushes the saturation knee further out.
        self.rewards.clf_reward = _RewT(
            func=mdp.clf_reward, weight=10.0,
            params={"command_name": "traj_ref", "max_eta_err": 0.8})


@configclass


class LpaWalkingCLFRoughV4EnvCfg(LpaWalkingCLFRoughV2EnvCfg):
    """V2 + LINEAR arm retrieval toward the reference (supersedes the
    V3 joint_deviation_l1 experiment before its results are in).

    V3's arms_home pulled toward the STATIC carriage default, which
    fights the reference whenever the style moves the arms. The
    corpus fix (Stability of CLF-RL, Lemma 5: linear costs sandwich
    the exponential on sublevel sets) is a linear penalty on the arm
    subgroup's tracking error itself: -w * ||eta_arms||. Constant
    retrieval gradient at any distance, zero on-orbit, aimed at the
    reference.

    Kept from the V3 reasoning: joint_pos_arms sigma 0.2 -> 0.5 per
    joint (basin reach). NOT touched: the clf_decreasing_condition
    clamp (eta_max 0.25 flattens off-orbit V-dot gradient, measured
    max_violation 1.95) — raising it rescales the WHOLE-BODY penalty
    and is the next lever, not this run's.
    """

    def __post_init__(self):
        super().__post_init__()
        import math as _math
        from isaaclab.managers import RewardTermCfg as _RewT
        from robot_rl.tasks.manager_based.robot_rl import mdp as _mdp

        self.rewards.joint_pos_arms.params["sigma"] = 0.5 * _math.sqrt(8)
        self.rewards.arms_track_linear = _RewT(
            func=_mdp.joint_pos_group_error_l2, weight=-0.5,
            params={"command_name": "traj_ref", "group": "arms"})


class LpaWalkingCLFRoughV3EnvCfg(LpaWalkingCLFRoughV2EnvCfg):
    """V2 + the arm-retrieval package (rough21 wrapped-arms fix).

    rough21 (v2, 4500 iters) walked with both arms wrapped at chest
    level -- L forearm across the front, R behind. Diagnosis from the
    run data: the wrap is INSIDE the hard ROM (roll pinned at its
    floor, yaw +-90 + elbow flexion carries the forearm across), and
    joint_pos_arms = exp(-KAPPA*v/sigma) has no gradient out there --
    Episode_Reward/joint_pos_arms collapsed to ~0.13 in the first
    1100 iters as the terrain curriculum hit 5.5 and flatlined for
    the remaining 3400 while clf_reward kept climbing. The exp can
    HOLD arms near reference but cannot RETRIEVE them; wrapped arms
    then do balance work for free (corr(arm pitch rate, base pitch
    rate) = -0.162, torque cost 1e-5).

    Two changes, both retrieval-side (dof_pos_limits stays scoped to
    legs+waist -- the soft-edge midpoint shrink would punish the
    user-approved throw chamber, see the HARD ARM ROM comment):

    1. arms_home: joint_deviation_l1 on the 8 arm joints, the
       waist_quiet pattern -- LINEAR gradient at any distance, so a
       wrapped arm always feels a pull toward the carriage default.
       Weight -0.5 over 8 joints is gentle next to waist_quiet's
       -1.5 over 3: it must not fight reference tracking, only make
       far-field drift cost something.
    2. joint_pos_arms sigma 0.2 -> 0.5 per joint (x sqrt(8) carry):
       stretches the exp basin so the gradient reaches ~2.5x further
       before saturating (the closeout's own prescription: restore
       gradient, widen sigma).
    """

    def __post_init__(self):
        super().__post_init__()
        import math as _math
        from isaaclab.managers import RewardTermCfg as _RewT
        from isaaclab.managers import SceneEntityCfg as _Scene
        import isaaclab.envs.mdp as _isaac_mdp

        self.rewards.joint_pos_arms.params["sigma"] = 0.5 * _math.sqrt(8)
        self.rewards.arms_home = _RewT(
            func=_isaac_mdp.joint_deviation_l1, weight=-0.5,
            params={"asset_cfg": _Scene(
                "robot",
                joint_names=[".*_SHOULDER_.*", ".*_ELBOW"])})


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


@configclass
class LpaWalkingCLFHistClear4EnvCfg(LpaWalkingCLFHistEnvCfg):
    """FLAT ground + history + the clear4 clearance reference.

    The discriminator for tinh-nwgy's first rough run: rough13 asked
    the policy to learn a NEW gait and NEW terrain simultaneously and
    pinned at ~100% falls through +800 while the rough8 control (same
    checkpoint, keeper refs) was at 0.577. This env removes terrain:

      falls converge here  -> clear4 is trackable; the rough failure
                              was the double change. Curriculum:
                              adapt on flat FIRST, then resume rough
                              from the adapted checkpoint.
      falls stay high here -> the clearance reference itself is
                              RL-hostile; back to the solver (lower
                              floor / different nodes), no GPU spent
                              on terrain.
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.traj_ref.path = "lpa_lib/trajectories/walk_forward_clear4"


class LpaWalkingCLFRoughV5EnvCfg(LpaWalkingCLFRoughV4EnvCfg):
    """V4 + anti-reward-hacking package (user direction 2026-08-26).

    Observed across rough21-23: arms parked behind the torso as
    forward-lean ballast, lean riding the 15-deg deadband edge
    (rough23 mean tilt 12.04), step length shrinking for stability.
    One theme: make the honest gait the cheap strategy.

    1. arms_vel_track (0.4): a parked arm pays the full reference
       swing amplitude CONTINUOUSLY; a small phase lag pays
       ~sin(lag/2) — phase tolerance with no phase-search machinery.
    2. arms_deviation_cap (-3.0, theta_max 0.4 rad): in-band
       deviation/lag FREE; the 0.5-1.0 rad parked offset is priced
       quadratically.
    3. arms_track_linear -0.5 -> -0.2: retrieval only; its flat tax
       favored parking at the mean.
    4. upright (+1.0, sigma 9 deg): duty-cycle bonus — brief lean
       (recovery/acceleration) forfeits a few steps of bonus,
       lean-as-strategy bleeds continuously. torso_pitch deadband
       UNCHANGED at 15 deg (user: allow brief lean).
    5. feet_air_time 2->3, xy_vel 1->2: stride levers; feet-position
       group split deferred to v6 pending closeout stride numbers.
    """

    def __post_init__(self):
        super().__post_init__()
        import math as _math
        from isaaclab.managers import RewardTermCfg as _RewT
        from robot_rl.tasks.manager_based.robot_rl import mdp as _mdp

        self.rewards.arms_vel_track = _RewT(
            func=_mdp.joint_vel_group_reward, weight=0.4,
            params={"command_name": "traj_ref",
                    "sigma": 0.5 * _math.sqrt(8), "group": "arms"})
        self.rewards.arms_deviation_cap = _RewT(
            func=_mdp.joint_group_deviation_cap, weight=-3.0,
            params={"command_name": "traj_ref", "group": "arms",
                    "theta_max": 0.4})
        self.rewards.arms_track_linear.weight = -0.2
        self.rewards.upright = _RewT(
            func=_mdp.upright_bonus, weight=1.0,
            params={"sigma_deg": 9.0})
        self.rewards.feet_air_time.weight = 3.0
        self.rewards.xy_vel.weight = 2.0


@configclass
class LpaWalkingCLFRoughV6EnvCfg(LpaWalkingCLFRoughV5EnvCfg):
    """V5 + the arm-swing overlay (user direction 2026-08-26).

    rough24 proved v5 killed the lean strategy (mean tilt 3.59 deg)
    but the arms still never swung: the collision bank showed the
    carriage yaw (-0.70) leaves 0.9 mm of forearm-hip clearance over
    any pitch swing -- the reference itself made swinging impossible.
    v6 keeps every v5 reward and moves the TARGET: the overlay swings
    the arm reference on the ellipse the user styled (fists in at the
    extremes, out at hip passage), so arms_vel_track / the deviation
    cap / arms_track_linear all aim at a swinging, collision-free
    reference instead of a parked, colliding one.
    """

    def __post_init__(self):
        super().__post_init__()
        from robot_rl.tasks.manager_based.robot_rl.mdp.commands.\
            traj_tracking.trajectory_cmd_cfg import ArmSwingOverlayCfg

        self.commands.traj_ref.arm_swing = ArmSwingOverlayCfg()


@configclass
class LpaWalkingCLFArmSwingPlayEnvCfg(LpaWalkingCLFEnvCfg_PLAY):
    """Flat PLAY env matching the v6 training distribution.

    The base flat play env serves the ORIGINAL walk reference (static
    yaw-in arm carriage + stomp punch orbit). A v6-trained policy is
    out-of-distribution against it: the CLF arm error grows over each
    cycle until the policy fights it at the actuator limits — the
    deterministic once-per-cycle right-arm slam + waist spin seen in
    the rough26 flat renders (rollout_cmd058 curves, 2026-08-27).
    Every flat render or rollout of an arm-swing policy must use THIS
    env so the reference carries the same overlay it trained against.
    """

    def __post_init__(self):
        super().__post_init__()
        from robot_rl.tasks.manager_based.robot_rl.mdp.commands.\
            traj_tracking.trajectory_cmd_cfg import ArmSwingOverlayCfg

        self.commands.traj_ref.arm_swing = ArmSwingOverlayCfg()


@configclass
class LpaWalkingCLFRoughV7EnvCfg(LpaWalkingCLFRoughV6EnvCfg):
    """V6 + contact pricing + asymmetric bands (rough26 diagnosis).

    The in-distribution rough26 curves showed a once-per-cycle arm
    whip at the actuator limits: elbow parked folded (eta -1.7,
    fists high — NOT collision avoidance, the bank refutes that),
    snapping at the reference's punch flourish, waist counter-
    rotating, contact forces shoving the torso. Three prices v6
    never charged:

    1. arm_contact: forearm/hand contact force is the punch
       MECHANISM — price it directly (sensor already covers every
       body; self-collisions are physically on).
    2. asymmetric bands: outward freedom kept at 0.4; inward
       (yaw/roll toward the body) tight at 0.1 past the reference;
       elbow band +-0.3 so the fold is priced from the first rad.
    3. arms_track_linear back to -0.5: constant retrieval pressure
       against the fossilized pendulum9b-lineage arm habit.
    """

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.managers import RewardTermCfg as _RewT
        from isaaclab.managers import SceneEntityCfg as _SceneCfg
        from robot_rl.tasks.manager_based.robot_rl import mdp as _mdp

        self.rewards.arms_deviation_cap = _RewT(
            func=_mdp.joint_group_deviation_cap_asym, weight=-3.0,
            params={"command_name": "traj_ref", "group": "arms",
                    "theta_max": 0.4,
                    "bands": {
                        "L_SHOULDER_YAW": (-0.10, 0.40),
                        "R_SHOULDER_YAW": (-0.40, 0.10),
                        "L_SHOULDER_ROLL": (-0.10, 0.40),
                        "R_SHOULDER_ROLL": (-0.40, 0.10),
                        "L_ELBOW": (-0.30, 0.30),
                        "R_ELBOW": (-0.30, 0.30),
                    }})
        self.rewards.arm_contact = _RewT(
            func=_mdp.undesired_contacts, weight=-2.0,
            params={"sensor_cfg": _SceneCfg(
                        "contact_forces",
                        # WRIST links hang on fixed joints and
                        # merge into the ELBOW bodies in PhysX —
                        # .*_WRIST matches nothing in the sensor
                        # (rough27 first launch died resolving it).
                        # The ELBOW body IS forearm+hand.
                        body_names=[".*_ELBOW"]),
                    "threshold": 1.0})
        self.rewards.arms_track_linear.weight = -0.5


@configclass
class LpaWalkingCLFRoughV8EnvCfg(LpaWalkingCLFRoughV7EnvCfg):
    """V7 + elbow tuck (user direction 2026-08-27).

    rough27 halved the whip but the arms still flare: shoulder roll
    rides outward (0.5 rad observed) with the elbow folded, dropping
    the fist inboard over the hip (FK: fold pulls hand y from 0.35 m
    to 0.16 m — hip height and hip line). The outward roll direction
    was left cheap in v7 (0.40 free + weak quadratic = 0.04/step at
    the observed flare). Tuck = tighten outward roll to 0.15 and
    double the cap weight so both the flare and the fossilized elbow
    fold pay real rent. ROM audit (2026-08-27): limits correctly
    mirrored, FK mirror-consistent to 3.5 mm — ROM allows 120 deg
    abduction by design, so the tuck must come from pricing.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.arms_deviation_cap.weight = -6.0
        self.rewards.arms_deviation_cap.params["bands"] = {
            "L_SHOULDER_YAW": (-0.10, 0.40),
            "R_SHOULDER_YAW": (-0.40, 0.10),
            "L_SHOULDER_ROLL": (-0.10, 0.15),
            "R_SHOULDER_ROLL": (-0.15, 0.10),
            "L_ELBOW": (-0.30, 0.30),
            "R_ELBOW": (-0.30, 0.30),
        }


@configclass
class LpaWalkingCLFRoughV9EnvCfg(LpaWalkingCLFRoughV8EnvCfg):
    """V8 + absolute elbow-depth band (user: "continue fixing the
    elbows", 2026-08-28).

    rough28 fixed roll (tucked), waist (quiet) and the whip (gone),
    and turned the parked elbow fold into a gait-locked pump — but
    the pump troughs at the -2.2 joint STOP, ~1 rad past the
    reference band, and the ref-relative cap plateaued at that
    equilibrium. The trough is a fixed angle, so the rule becomes
    absolute: pump to -1.2 and no deeper (reference max depth is
    -0.69; the extra 0.5 rad is transition room), never past
    +0.1 toward hyperextension. Same lever class that broke the
    roll flare, aimed where the residual lives.
    """

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.managers import RewardTermCfg as _RewT
        from robot_rl.tasks.manager_based.robot_rl import mdp as _mdp

        self.rewards.elbow_depth = _RewT(
            func=_mdp.joint_abs_band, weight=-8.0,
            params={"joint_names": ["L_ELBOW", "R_ELBOW"],
                    "lo": -1.20, "hi": 0.10})


@configclass
class LpaWalkingCLFRoughV10EnvCfg(LpaWalkingCLFRoughV9EnvCfg):
    """V9 with the elbow band placed where TRAINING lives.

    v9 read ~1e-4 and did nothing. Diagnosis (2026-08-28): play uses
    act_inference (deterministic mean action) while training samples
    at noise std 2.56, and the elbow's -2.2 lower STOP clips that
    noise asymmetrically — so the deployed pose sits ~0.4 rad deeper
    than anything training visits. Measured: deterministic rollouts
    median -1.4/-1.6 with 66% of steps past -1.20, IDENTICAL on flat
    and rough (so it is not terrain); training's own logged elbow
    error (0.38 against a -0.6 reference) puts the training-time
    elbow near -1.0 — INSIDE v9's band, which is exactly why it
    never fired.

    A band can only shape states the training distribution visits.
    It moves to -0.95: inside that distribution, firing on the
    deeper half, dragging the distribution up — and with it the
    deterministic pose the videos actually show.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.elbow_depth.params["lo"] = -0.95



@configclass
class LpaWalkingCLFGraph1EnvCfg(LpaWalkingCLFGraphTurnEnvCfg):
    """graphturn19 (am-nai): the graph env on SEAM-EXACT references,
    with standing given real curriculum mass.

    Two changes against the graphturn variant, each answering a
    measured graphturn18 failure:

      library  chain1/graph1 instead of skill_smoke. Its locomotion
               refs are snipped from ONE 9-domain chain solve, so
               every stand->walk / cycle / walk->stand seam agrees to
               <=0.02 rad AND <=0.02 rad/s by construction, against
               the 0.36-0.4 m/s mismatch between the separately
               solved segments. gt18's own post-mortem blamed
               mid-gait reference swaps.

      standing rel_standing_envs 0.05 -> 0.25. gt18 gave frozen-phase
               standing 5% of envs and could not hold a pinned zero
               command for two seconds. Standing here means FEET
               TOGETHER (the certified stop pose, +-1 mm stagger) —
               a different state from the staggered double support
               the phase-hold law freezes at.

    A SEPARATE env id: graphturn18 trained under
    LpaWalkingCLFGraphTurnEnvCfg, and editing that in place would make
    its run declaration describe an env it never saw.
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.traj_ref.path = "lpa_lib/trajectories/graph1"
        # The four command fractions are validated as
        # open + closed + closed_yaw + standing == 1.0, by EXACT float
        # equality (velocity_commands.py). Raising standing to 0.25
        # without rebalancing left them at 1.20 and Isaac raised
        # "Relative envs ... don't sum to 1!" two minutes into boot,
        # after the 10.7GB env had already been cloned and staged.
        # These four sum to exactly 1.0 in float arithmetic; not every
        # set of round decimals does, so re-check when changing them.
        self.commands.base_velocity.rel_standing_envs = 0.25
        self.commands.base_velocity.rel_closed_loop = 0.40
        self.commands.base_velocity.rel_closed_loop_yaw = 0.20
        self.commands.base_velocity.rel_open_loop = 0.15


@configclass
class LpaWalkingCLFClad1EnvCfg(LpaWalkingCLFSkillEnvCfg):
    """graph1's layout against the CLAD behaviour library.

    clad_graph1 is the first library solved for the robot that exists:
    110.474 kg with cladding, and with the overhead traveller modelled
    as a world-up FORCE at WAIST_YAW (15.4 kg -- the same lift this env
    applies) rather than as removed mass. Every earlier library, graph1
    included, was solved either unclad or against the mass-removal
    uplift, so a policy trained on one imitated a reference its own
    body could not produce.

    Three behaviours, not graph1's full skill set: the stomp cycle plus
    both transitions. Seams to the cycle are 0.0800 rad / 0.3500 rad/s
    (stand->walk) and 0.1000 / 0.3500 (walk->stand), the velocity half
    at the certified tolerance.

    Based on the SKILL env, not Graph1. Graph1 inherits GraphTurn,
    whose graph_skills event does
    lib.ref_id_of("lpa_turn_left"/"lpa_turn_right"), so it HARD-REQUIRES
    turn references in its library -- measured: a smoke on the Graph1
    base died with KeyError: 'walk_to_stand' at that lookup. The clad
    library has no turns, because the only turn solves that exist are
    unclad, which is exactly what this env is here to stop using.

    A SEPARATE env id, for the same reason graph1 was: editing graph1
    in place would make graphturn21's run declaration describe an env
    it never saw.
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.traj_ref.path = "lpa_lib/trajectories/clad_graph1"
        # The skill env samples laser enter/exit on an interval, and
        # graph_skill_sampler resolves those BY NAME:
        #   ei = lib.ref_id_of("laser_enter")  -> KeyError
        # 90 s into a smoke, because this library has no laser refs.
        # The skill one-hot and params observations stay; what goes is
        # the sampler for skills that do not exist here.
        self.events.graph_skills = None


@configclass
class LpaWalkingCLFClad2EnvCfg(LpaWalkingCLFGraph1EnvCfg):
    """Graph1's layout against the clad library WITH turns.

    clad1 had to sit on the SKILL env because clad_graph1 has no turn
    references and graph_turn_sampler resolves them BY STRING --
    ref_id_of() for lpa_turn_left, lpa_turn_right, stand_to_walk,
    walk_to_stand and walk_to_stand_R -- so a library missing one
    raises KeyError at load. clad_graph2 supplies all five.

    The turn underneath is the first solved on the clad robot at all:
    every earlier one came from a probe reading a hand-built path into
    the retired TINH checkout, so it was fitted to 26 links / 71.9 kg
    against today's 44 / 110.5 (am-3v7).

    Keeps Graph1's laser-free sampler: Graph1 REPLACES graph_skills
    with graph_turn_sampler, which resolves no laser_enter/laser_exit,
    so this library legitimately has none and does not need clad1's
    `graph_skills = None`.
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.traj_ref.path = "lpa_lib/trajectories/clad_graph2"


@configclass
class LpaWalkingCLFClad3EnvCfg(LpaWalkingCLFRoughV10EnvCfg):
    """The ROUGH terminal, terrain forced FLAT, driving clad_graph2.

    clad1 fell over and its arms left the sagittal plane. Its reward
    set had ZERO arm terms: the skill env branches off the BASE, and
    every arm mitigation was built in the rough lineage V3..V10, which
    is not on that path.

    Composed this way round on Asa's suggestion (2026-08-31): "take
    the rough env but force the terrain to always be flat at first".
    Inheriting V10 takes ALL ten generations -- arms_home,
    arms_track_linear, arms_vel_track, the asymmetric
    arms_deviation_cap with V8's tight roll bands, arm_contact,
    elbow_depth, upright, torso_pitch, and whatever else those
    generations settled that a hand-picked port would miss. The first
    attempt at this class DID hand-pick seven terms -- the same
    mistake that produced clad1: choosing an ancestor without
    accounting for what the choice drops.

    What must be added back, because rough is "the plain walking env
    plus terrain -- no skill slots":

      terrain   forced to plane, generator None. Flat ground first.
      skills    the traj_ref library, slot vocabulary, param channels
                and the skill_onehot / skill_params observations that
                LpaWalkingCLFSkillEnvCfg would have supplied.
      sampler   graph_turn_sampler, NOT the skill env's laser sampler:
                clad_graph2 has turns and no laser references, and the
                laser sampler resolves laser_enter BY NAME.
    """

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.managers import ObservationTermCfg as _ObsTerm

        # FLAT FIRST. V10 inherits the rough terrain generator; take
        # the arm and stability work without the ground it was tuned on.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # ... and the curriculum that rides on it. terrain_levels
        # promotes a robot a ROW when it walks past half a patch; with
        # no generator there are no rows to promote into. The base LPA
        # env keeps this None for exactly that reason and RoughEnvCfg
        # re-enables it, so re-disabling is part of "force the terrain
        # flat" rather than an extra opinion of mine.
        self.curriculum.terrain_levels = None

        self.commands.traj_ref.path = "lpa_lib/trajectories/clad_graph2"
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
            func=mdp.graph_turn_sampler,
            mode="interval",
            interval_range_s=(0.5, 0.5),
            params={"command_name": "traj_ref", "p_turn": 0.10})

@configclass
class LpaWalkingCLFClad4EnvCfg(LpaWalkingCLFClad3EnvCfg):
    """clad3 with the command distribution narrowed to WALKING.

    WHY, measured off clad3's own config rather than guessed. Asa,
    watching the one-hour render: "There is a whole lot of holding the
    arms out behind, and standing around. Very little walking. Are we
    playing out stomp walk reference here?"

    We were -- 41.7% of the time. clad_graph2 has THREE periodic
    gaits, and the free state picks among them by nearest-gait under a
    weighted L2 with vel_yaw weighted 2.5x:

        lpa_stomp        vel_x  0.5732   vel_yaw  0.0
        lpa_turn_left    vel_x -0.01     vel_yaw +0.2646
        lpa_turn_right   vel_x -0.01     vel_yaw -0.2646

    The turns are PIVOTS -- vel_x -0.01, no forward travel. Against
    commands of vx ~ U(0, 0.58) and wz ~ U(-0.29, 0.29), any
    meaningful yaw pulls selection onto a pivot:

        turn gait   48.8 %      <- standing around
        stomp walk  41.7 %
        phase held   9.5 %      <- |cmd| < hold_phi_threshold 0.10,
                                   pose frozen, no gait cycle at all

    Two separate faults, both fixed here:

    1. HALF THE SAMPLES WERE PIVOTS. clad1 never had this -- its
       library, clad_graph1, has exactly one periodic gait, so its
       locomotion envs tracked the walk 100% of the time. Pointing
       clad3 at clad_graph2 for its turns swapped half of training
       into turning in place.

    2. THE WALKING HALF FOUGHT ITSELF. The only straight gait is
       0.573 m/s while commanded vx averaged 0.290, so the CLF term
       tracked a 0.57 reference while the velocity term asked for
       0.29. A compromise between those looks like marking time.

    Narrowing vx to a band AROUND the gait speed fixes both: mean
    commanded vx becomes 0.515 against the gait's 0.573, and the
    turns stop being selected at all.

        stomp 41.7% -> 100%,  turn 48.8% -> 0%,  held 9.5% -> 0%

    THE TURNS ARE NOT DELETED, only unreachable by nearest-gait. The
    graph sampler still traverses walk -> walk_to_stand -> turn ->
    stand_to_walk explicitly, which is how graph_turn_sampler's
    docstring says turning is supposed to be entered. Nearest-gait
    dropping an env straight onto a turn gait BYPASSED that traversal
    -- see am-kax. Widen these ranges back once flat walking is a
    keeper; Asa: "We need to get this all working on flat ground
    first."

    A NEW ID, not an edit to clad3: run 2026-08-31_clad3 trained 1000
    iterations under clad3 and its results are on am-w51, so
    redefining clad3 would make that run's declaration describe an env
    it never saw.
    """

    def __post_init__(self):
        super().__post_init__()
        # A BAND around the gait, not a range from zero. This keeps
        # the tracked reference and the velocity reward agreeing on a
        # speed; what makes turns unreachable by command is the
        # mixable: false on the turn trajectories themselves.
        self.commands.base_velocity.ranges.lin_vel_x = (0.45, 0.58)
        # Enough yaw to hold a heading, far too little to select a
        # pivot: at wz = 0.03 the stomp is nearer by 27x.
        self.commands.base_velocity.ranges.ang_vel_z = (-0.03, 0.03)
        # Standing envs command (0,0,0), which is below
        # hold_phi_threshold and freezes the phase mid-stomp -- that
        # is the held pose in the render, arms wherever the cycle left
        # them. No standing while we are fixing walking.
        #
        # ALL FOUR, because VelocityTrackingCommand asserts they sum
        # to 1 and dropping standing alone leaves 0.95 -- clad4's
        # first launch died on exactly that. The freed 0.05 goes to
        # closed_loop, whose envs walk to a y-target and heading. The
        # ramp env sets all four the same way for the same reason.
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_closed_loop = 0.55
        self.commands.base_velocity.rel_closed_loop_yaw = 0.25
        self.commands.base_velocity.rel_open_loop = 0.20
        #
        # The narrowed ang_vel_z binds the CLOSED-LOOP envs too, which
        # is why it is enough on its own: the heading controller's
        # output is clipped to ranges.ang_vel_z (velocity_commands.py
        # _update_command), so no env class can steer harder than the
        # band allows and reach a turn conditioner.

@configclass
class LpaWalkingCLFCladWalkEnvCfg(LpaWalkingCLFClad4EnvCfg):
    """THE WALKING BASELINE on the clad stomp reference. Nothing else.

    Asa: "turns appear to be breaking things, so I want to be sure
    that we have good walking policy we can control first" -- and,
    on what this is for, "I am trying to get the first trained model
    on the new clad stomp walk working as a baseline we can extend."

    Taking turns out of the nearest-gait mixer (mixable: false) was
    necessary and did not finish the job: graph_turn_sampler enters
    them by the OTHER route, and that route was the larger one.

        turn traversal = walk_to_stand 1.273
                       + turn 1.890 x turn_periods 2.0
                       + stand_to_walk 0.900          = 5.95 s
        p_turn 0.10 per 0.5 s tick -> a walking env draws one every
        ~5.0 s

        => 54.4% of the time in a traversal, 45.6% walking

    So clad4 walked 45.6% of the time, against clad3's 41.7%. Barely
    moved, for a different reason. p_turn = 0 is what actually makes
    this a walking run.

    WHAT THIS LEAVES. The stomp cycle, commanded in a band around its
    own speed, and nothing else: with p_turn = 0 no env ever leaves
    _FREE, so the transitions are never entered either. That is
    correct here rather than a loss -- walk_to_stand and
    stand_to_walk exist to start and stop a commanded walk, and a
    command band of (0.45, 0.58) never stops.

    THE SAMPLER STAYS INSTALLED, at p_turn 0. Its skill_onehot and
    skill_params observations are what make this env 79 wide, so
    removing it would break checkpoint compatibility with everything
    in this lineage. Re-enabling turns is then one number, on a
    policy that can already walk -- which is the "baseline we can
    extend".

    NOT a rename of clad4: run 2026-08-31_clad4b is training under
    clad4 as this is written.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.graph_skills.params["p_turn"] = 0.0


@configclass
class LpaWalkingCLFCladWalk2EnvCfg(LpaWalkingCLFCladWalkEnvCfg):
    """cladwalk, tracking the arms trajopt ACTUALLY SOLVED.

    Asa on the cladwalk1 hour render: "I am not seeing enough
    reference following, as the arms are way way more active than in
    the reference."

    They were, and not by a tuning margin -- the policy was tracking a
    target that is not the reference. V6's ArmSwingOverlayCfg rewrites
    traj_ref before the policy sees it:

        pitch   +0.40 rad ADDED to the reference
        yaw     REPLACED by an ellipse, [-0.55, -0.10] L-signed

    Measured on clad_graph2's lpa_stomp, the reference already swings
    the shoulders +-0.25 rad about a +0.22 mean, anti-phase
    (corr(L,R) = -0.887), with solved yaw in [-0.672, -0.270]. So the
    overlay roughly 2.6x'd the commanded pitch swing and threw the
    solved carriage away.

    WHY THE OVERLAY EXISTED, and why it no longer should. V6's own
    docstring: "the carriage yaw (-0.70) leaves 0.9 mm of forearm-hip
    clearance over any pitch swing -- the reference itself made
    swinging impossible, which is why rough21-24 never swung
    regardless of reward shaping." That was the UNCLAD walk_forward
    reference. The clad stomp was re-solved WITH wrist-thigh clearance
    (tinh-lpa-clfrl.7.10) and swings on its own. The overlay is curing
    a disease this reference does not have, and the cure became the
    symptom.

    Same failure shape as clad1 one level down: inheriting the rough
    lineage wholesale brings terms tuned against a reference that no
    longer looks like the one being tracked.

    STILL OPEN, deliberately not changed here so the result is
    attributable:

      - arms_deviation_cap's asymmetric bands and arms_vel_track were
        tuned against the OVERLAID target. They are deviation-from-
        traj_ref terms so they remain meaningful, but the band widths
        want a second look once the arms track.
      - reference tracking is outweighed by progress 9.66 to 2.61 per
        step, which is why the hips and CoM move less than the
        reference. That is a separate change (am-m7l).
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.traj_ref.arm_swing = None


@configclass
class LpaWalkingCLFCladWalkP9EnvCfg(LpaWalkingCLFCladWalk2EnvCfg):
    """cladwalk with PENDULUM9B's arm treatment: track, do not shape.

    Asa, 2026-09-01: "Pendulum 9b trained the arms fine so I wonder
    what so much changed from how we trained that run? I know that was
    the lighter version of the bot, but we used a similar character
    for the reference walk and had a good looking walk policy come
    out."

    Diffed. cladwalk3 is pendulum9b's reward set PLUS eleven
    rough-lineage terms, and pendulum9b has nothing it lacks:

        arm_contact  arms_deviation_cap  arms_track_linear
        arms_vel_track  elbow_depth  upright  torso_pitch
        waist_quiet  joint_pos_{arms,legs,torso}

    And the split cost the arms most of their tracking weight:

        term              pendulum9b   cladwalk3
        joint_pos            1.0         0.0   <- zeroed
        joint_pos_arms        -          0.381
        joint_pos_legs        -          0.476
        joint_pos_torso       -          0.143

    So the arms went from weight 1.0 to 0.381 -- a 2.6x cut in how
    hard the reference pulls them -- and then eight band/cap terms
    were layered on that push toward BOUNDS rather than toward the
    reference. Measured consequence on cladwalk3: arm ranges 2.6-7.1x
    the reference's, elbow 1.0 rad deeper than the reference and
    riding its -2.2 stop.

    The stack was built over V3..V10 to fix arms on ROUGH TERRAIN with
    the UNCLAD walk_forward reference, whose carriage left 0.9 mm of
    forearm-hip clearance and made swinging mechanically impossible.
    The clad stomp swings +-0.25 rad on its own with real clearance.
    The stack is solving a problem this reference does not have.

    THIS ENV: pendulum9b's arm treatment, everything else cladwalk2.
    Restores joint_pos to 1.0 over all joints, zeroes the three
    splits, and removes the six ARM-shaping terms. torso_pitch and
    waist_quiet stay -- they are not arm terms, and waist_quiet is
    part of the waist guard trio added after the ramp env found a
    heading exploit.

    NOT A CLAIM THAT THE STACK IS WRONG IN GENERAL. It is a claim that
    it is wrong HERE: flat ground, a reference with good arms, at
    100% walking. Rung 4 (rough terrain) may well want it back.
    """

    def __post_init__(self):
        super().__post_init__()
        # Track the whole body with one term at full weight, as
        # pendulum9b did, instead of three splits summing to 1.0 that
        # give the arms 0.381.
        self.rewards.joint_pos.weight = 1.0
        self.rewards.joint_pos_legs = None
        self.rewards.joint_pos_arms = None
        self.rewards.joint_pos_torso = None
        # Shape nothing. Let the reference do the work.
        self.rewards.arms_track_linear = None
        self.rewards.arms_vel_track = None
        self.rewards.arms_deviation_cap = None
        self.rewards.arm_contact = None
        self.rewards.elbow_depth = None
        self.rewards.upright = None


@configclass
class LpaWalkingCLFCladWalkClearEnvCfg(LpaWalkingCLFCladWalk2EnvCfg):
    """Put the FOREARM CLEARANCE back. It was never about style.

    Measured 2026-09-01 with lpa_reset_cause_probe on cladwalkp9,
    pinned at 0.57 m/s:

        torso_contact     91   100.0%
        time_out / base_height / waist_twist / bad_orientation   0

    NOT ONE FALL. Every reset is contact above 1 N on TORSO_ROLL,
    which carries the torso and head spheres -- the forearms hitting
    the chest. Asa called it from the video: "Likely resetting from
    self collisions?"

    So "pinned survival" was never measuring driveability in this
    family; it was counting arm-to-torso strikes. And the count tracks
    exactly the two mechanisms that kept the forearms clear, removed
    one after the other:

        run          overlay   arm_contact   resets
        cladwalk1      yes         yes          16
        cladwalk2      NO          yes          73
        cladwalk3      NO          yes          74
        cladwalkp9     NO          NO          130

    ArmSwingOverlayCfg is a CLEARANCE DEVICE, not decoration. Its own
    docstring: "at the pitch extremes the forearm is displaced
    fore/aft so yaw-in is safe; at hip passage yaw must open out. Bank
    clearance at the defaults below: 57-92 mm over the full cycle." I
    removed it (am-m7l.5) reading that as history about the UNCLAD
    reference, on the grounds that it distorted arm STYLE -- which it
    did. It was also the only geometry keeping the arms off the body.

    THIS ENV restores both, on top of everything learned since:
    the overlay, arm_contact, the entropy cut (cladwalk3), turns out
    of the mixer, p_turn 0, and the walking command band.

    THE STYLE COST IS REAL AND UNRESOLVED. The overlay adds 0.40 rad
    of pitch to a reference that already swings +-0.25 and REPLACES
    the solved yaw [-0.672, -0.270] with an ellipse [-0.55, -0.10].
    Asa's original complaint -- "the arms are way way more active than
    in the reference" -- is a complaint about this overlay. Restoring
    it trades that back for clearance.

    The right answer is probably neither: a clearance CONSTRAINT
    solved into the clad reference itself, so the trajopt arms are
    both correct and safe. That is am-m7l.11. This run buys a working
    baseline while that is built.
    """

    def __post_init__(self):
        super().__post_init__()
        # ABSOLUTE, matching V6 six hundred lines above. The relative
        # form I first wrote (...mdp) resolves to manager_based.mdp,
        # not robot_rl.mdp, and died inside Isaac with
        # ModuleNotFoundError after the env was staged.
        from robot_rl.tasks.manager_based.robot_rl.mdp.commands.\
            traj_tracking.trajectory_cmd_cfg import ArmSwingOverlayCfg

        self.commands.traj_ref.arm_swing = ArmSwingOverlayCfg()


@configclass
class LpaWalkingCLFCladWalkArmEnvCfg(LpaWalkingCLFCladWalk2EnvCfg):
    """Track the SOLVED arms hard enough that the reference's own
    clearance suffices.

    Asa: 'make the policy track well enough that 3.5 mm suffices.'

    THE PROBLEM, measured rather than argued. Every reset in this
    family is torso_contact -- forearm-to-chest self-collision, zero
    falls, on three checkpoints (am-m7l.5). The stomp reference runs
    at 2.9 mm worst clearance and the collision bank can buy it about
    3.5 mm; asking 4.1 mm makes the solve infeasible with
    L_SHOULDER_PITCH at 62/62 Nm. So no plausible margin survives a
    policy that is a RADIAN off at the elbow. The margin is not the
    lever -- the tracking is.

    WHY THE ARMS ARE FREE. arms_track_linear is already the
    right-shaped term: joint_pos_group_error_l2 is LINEAR in tracking
    error, so unlike the exp form it has constant gradient at any
    distance and can retrieve a limb that has left the basin (its
    docstring: rough21 walked 3400 iters with both arms wrapped and
    the exp term flat at ~0.13). It simply has no authority --
    measured -0.266 per step against progress at +9.66, under 3%.
    Every other arm term is smaller still: arms_deviation_cap
    -0.022, elbow_depth -0.0002.

    That is why six reward configurations all gave 2.6-7.2x the
    reference's arm range -- with the shaping stack and without it,
    split tracking and unsplit, noise 2.2 and 1.0. Nothing was ever
    asking the arms to be anywhere.

    THIS ENV: arms_track_linear -0.5 -> -4.0, on cladwalk2's
    overlay-off base so the target is the SOLVED carriage rather than
    V6's ellipse. Tracking the overlay hard would only make the punch
    precise.

    8x is deliberately decisive rather than a nudge: at -0.5 the term
    is 3% of progress, and a 2x step would still be inside the noise
    of what six previous configurations failed to move. If it
    overshoots, the failure is legible -- arms pinned rigidly on
    reference, legs degraded because effort moved to the arms -- and
    that is a more useful result than another null.

    RISK, stated: progress is load-bearing for not falling, and this
    does not touch it. If the legs regress, the ratio is wrong rather
    than the direction.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.arms_track_linear.weight = -4.0


@configclass
class LpaWalkingCLFCladWalkClrEnvCfg(LpaWalkingCLFCladWalkArmEnvCfg):
    """Price the arm-to-torso MARGIN, not just the contact cliff.

    Asa 2026-09-01: 'lets try the clearance shaping reward.' (am-m7l.14)

    THE MEASURED SITUATION. Every reset in this family is
    torso_contact -- forearm/hand into the chest, zero falls -- on
    four checkpoints. The only term that saw it was the illegal-contact
    termination at >1 N on TORSO_ROLL: a cliff with no gradient
    leading up to it, so the policy learns nothing about the chest
    until it touches. Both other levers were tried and measured today:

      joint tracking x8   cladwalkarm: 73 resets vs cladwalk3's 74,
                          elbow range 2.15 vs 2.14 rad. Inert.
      banked reference    3.5 mm bank in the stomp solve stalled in
                          stage 2 (MUMPS, 36.7 GB) and was killed.

    THIS ENV: one new reward, mdp.arm_torso_clearance -- the trajopt's
    own collision-bank rows between the arm links (shoulder roll/yaw,
    elbow: 224 sphere rows) and TORSO_ROLL, on the policy's live body
    poses, each row paying ((m - d)/m)^2 below its margin m =
    min(20 mm, rest - 2 mm), the bank's clamp. Parity of these rows on
    Isaac poses was measured (lpa_sphere_parity_probe: 2.8 mm on the
    binding shoulder-roll pair vs the bank's 6.2 mm rest). Weight -5 so
    that at 10 mm on the closest row the cost is -1.25/step and at
    contact -5, against progress at +9.66: decisive, the way
    cladwalkarm's 8x was, so a null result is a null and not a nudge
    inside the noise.

    Inherits cladwalkarm (tracking 8x stays on: it is the lineage and
    it measured as inert). Obs layout unchanged, so cladwalkarm's
    checkpoint resumes directly.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.arm_torso_clearance = RewTerm(
            func=mdp.arm_torso_clearance,
            weight=-5.0,
            params={"margin": 0.02})


@configclass
class LpaWalkingCLFCladWalkSymEnvCfg(LpaWalkingCLFCladWalkClrEnvCfg):
    """The left arm is the right arm, half a cycle later.

    Asa on the cladwalkclr render (2026-09-01): 'its much better than
    the last passes, and the right arm looks good, but the left arm is
    very different from it. can we have a term that penalizes
    differences from left to right, but only when shifted by the
    counterswing phase?'

    MEASURED on cladwalkclr (arm-symmetry probe, mean actions, vx
    0.515): shoulder pitch range L 1.09 vs R 1.21, roll 0.58 vs 0.27,
    yaw 1.02 vs 0.85, elbow mean -1.55 vs -1.30; worst mirror gap
    0.30 rad. The reference is mirror-symmetric with a half-cycle
    delay, so a gap that large is the policy's.

    THIS ENV: one new reward, mdp.arm_phase_mirror -- IN SPACE: the
    elbow origin and hand point of each arm, in the base frame,
    against the y-mirrored points of the other arm 0.745 s earlier
    (the stomp step period; the library cycle is 1.489 s), mean
    squared distance, zero until an episode is older than the shift.
    Cartesian because the reference is an exact physical mirror with
    a half-cycle delay (hands to 1 mm) while its joint-space mirror is
    not the axis-string sign table -- a joint-space draft with those
    signs (cladwalksym, killed at iter 30) pushed the left roll and
    elbow the wrong way. Weight -40: the smoke on cladwalkclr's
    checkpoint opened at 0.26 m^2 (51 cm rms, the arms in phase, not
    counterphase), which at -200 was ~52/step -- several times
    progress, enough to wreck the walk before fixing the arms. At -40
    the opening cost is ~10/step, about progress, and a 5 cm residual
    asymmetry still costs -0.1/step, the clearance term's scale.
    Inherits cladwalkclr (clearance term
    stays: it is what stopped the resets), obs unchanged, resumes its
    checkpoint -- one variable.

    FAIL-with-information: the shifted error does not fall below the
    unshifted one in the first-call print -> the shift is wrong for
    this gait (measure the period, do not guess it); the arms become
    symmetric and stop swinging -> the term is fighting the tracking
    term and the weight is too high, halve it.
    """

    def __post_init__(self):
        super().__post_init__()
        # unshifted_cap: cladwalksym3 satisfied the shifted mirror by
        # swinging both arms together at twice the gait frequency (Asa:
        # 'exactly in sync'); the capped unshifted term forfeits 0.04 m^2
        # x 40 = 1.6/step for that and pays it back for a counterswing.
        self.rewards.arm_phase_mirror = RewTerm(
            func=mdp.arm_phase_mirror,
            weight=-40.0,
            params={"shift_s": 0.745, "unshifted_cap": 0.04})
