# LPA articulation config for robot_rl / Isaac Lab (bead tinh-7a2m).
#
# OVERLAY FILE: canonical copy lives in the tinh repo; deploy by
# copying into the robot_rl fork on the training box as
#   source/robot_rl/robot_rl/assets/robots/lpa_21j.py
# (see modules/wbc/lpa_rl/README.md). Modeled on g1_21j.py.
#
# Numbers:
#   - effort/velocity: lpa_walk.urdf limits (body HEJ 140 Nm /
#     10.4 rad/s, arm HEJ 62 Nm / 17.8 rad/s)
#   - armature/damping: build_lpa_mjcf.py _JOINT_DYN first guesses —
#     SYSID PLACEHOLDERS until tinh-lpa-sysid fits real rollouts
#   - PD gains: robot_rl's convention (stiffness = armature * w^2,
#     damping = 2 * zeta * armature * w, w = 2*pi*10 Hz, zeta = 2)
#   - init state: the stand behavior pose (walk_start ds node 0 of
#     the committed fixtures) so episodes start on the library's
#     entry state
#   - USD: generated on the training box from lpa_walk.urdf via
#     modules/wbc/lpa_sim/lpa_isaac_load_probe.py into the fork's
#     robot_assets/lpa/ (regenerate after any URDF change)

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# HEJ first guesses (armature, viscous damping) keyed by class.
ARMATURE_HEJ_BODY = 0.12   # 140 Nm joints (legs, waist, torso)
ARMATURE_HEJ_ARM = 0.08    # 62 Nm joints (shoulders, elbows)

# 5.5 Hz (not G1's 10): LPA's HEJ armature first-guess (0.12) is ~5x
# G1's biggest actuator, so 10 Hz gives stiffness ~474 Nm/rad and an
# action scale of 0.037 rad — the first policy tracked the gait's
# rhythm but couldn't command the stride amplitude (knee/hip swings
# compressed ~2x, reward plateau 3.3, every episode ending in a fall).
# 5.5 Hz lands K~143, action scale ~0.12 — the G1 recipe's ballpark.
NATURAL_FREQ = 5.5 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

STIFFNESS_BODY = ARMATURE_HEJ_BODY * NATURAL_FREQ**2
STIFFNESS_ARM = ARMATURE_HEJ_ARM * NATURAL_FREQ**2
DAMPING_BODY = 2.0 * DAMPING_RATIO * ARMATURE_HEJ_BODY * NATURAL_FREQ
DAMPING_ARM = 2.0 * DAMPING_RATIO * ARMATURE_HEJ_ARM * NATURAL_FREQ

ROBOT_ASSETS = "robot_assets/lpa"

LPA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ROBOT_ASSETS}/lpa_21j/lpa_visual.usd",  # tinh-wmor: + render meshes
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.01, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # The stand behavior pose: walk_start ds node 0 (committed
        # fixture), CORE at 1.06 m. Exact per-joint values — L/R are
        # NOT symmetric (the solved stand leans slightly).
        pos=(0.0, 0.0, 1.06),
        joint_pos={
            "L_HIP_PITCH": -0.5588,
            "L_HIP_ROLL": -0.0308,
            "L_HIP_YAW": -0.0685,
            "L_KNEE": 0.6699,
            "L_ANKLE": -0.2563,
            "R_HIP_PITCH": -0.5297,
            "R_HIP_ROLL": -0.0284,
            "R_HIP_YAW": -0.0688,
            "R_KNEE": 0.6677,
            "R_ANKLE": -0.2848,
            "WAIST_YAW": 0.0425,
            "TORSO_PITCH": -0.0387,
            "TORSO_ROLL": -0.05,
            "L_SHOULDER_PITCH": -0.0067,
            "L_SHOULDER_ROLL": 0.15,
            "L_SHOULDER_YAW": 0.0,
            "L_ELBOW": -0.15,
            "R_SHOULDER_PITCH": -0.0033,
            "R_SHOULDER_ROLL": -0.15,
            "R_SHOULDER_YAW": 0.0,
            "R_ELBOW": -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # Single HEJ family per class. Body = HEJ 90 with the
        # MAXON-PROVIDED torque-speed envelope (vendor 2026-08-09,
        # tinh-lpa-clfrl.6.4): DCMotor four-quadrant curve —
        # saturation 294 Nm at zero speed (140 rpm / 0.4762 rpm/Nm),
        # no-load 14.661 rad/s (140 rpm), continuous clip 128 Nm.
        # The vendor's mild 0.0405 Nm/rpm derate of the flat region
        # is not representable in the 3-param model (<=2.5%
        # optimistic mid-band; the MuJoCo gate carries the exact
        # envelope). Old flat 140/10.4 was peak-torque + conservative
        # speed.
        "body": DCMotorCfg(
            joint_names_expr=[
                ".*_HIP_PITCH", ".*_HIP_ROLL", ".*_HIP_YAW",
                ".*_KNEE", ".*_ANKLE",
                "WAIST_YAW",
            ],
            effort_limit=128.0,
            saturation_effort=294.0,
            velocity_limit=14.661,
            stiffness=STIFFNESS_BODY,
            damping=DAMPING_BODY,
            armature=ARMATURE_HEJ_BODY,
        ),
        # SONAX 4-bar torso (tinh-lpa-clfrl.5.6): TORSO_PITCH/ROLL are
        # driven by two HEJ90 cams through crank+rod push-rods onto the
        # U-joint plate. Joint-space capacity is pose-dependent and
        # ASYMMETRIC (closure-solved from CAD, sonax_linkage.py):
        # pitch 233-269 Nm over the ROM (rods in phase), roll
        # 96-138 Nm (differential). Flat MIN-over-ROM bounds here.
        "sonax_pitch": ImplicitActuatorCfg(
            joint_names_expr=["TORSO_PITCH"],
            effort_limit_sim=233.0,
            velocity_limit_sim=10.4,
            stiffness=STIFFNESS_BODY,
            damping=DAMPING_BODY,
            armature=ARMATURE_HEJ_BODY,
        ),
        "sonax_roll": ImplicitActuatorCfg(
            joint_names_expr=["TORSO_ROLL"],
            effort_limit_sim=96.0,
            velocity_limit_sim=10.4,
            stiffness=STIFFNESS_BODY,
            damping=DAMPING_BODY,
            armature=ARMATURE_HEJ_BODY,
        ),
        # Arms = HEJ 70: DERIVED placeholder envelope (same shape
        # ratios as the vendor HEJ 90 — saturation 2.30x continuous;
        # UNVERIFIED, awaiting maxon numbers, tinh-lpa-clfrl.6.1).
        "arms": DCMotorCfg(
            joint_names_expr=[
                ".*_SHOULDER_PITCH", ".*_SHOULDER_ROLL",
                ".*_SHOULDER_YAW", ".*_ELBOW",
            ],
            effort_limit=62.0,
            saturation_effort=142.6,
            velocity_limit=17.8,
            stiffness=STIFFNESS_ARM,
            damping=DAMPING_ARM,
            armature=ARMATURE_HEJ_ARM,
        ),
    },
)
"""Configuration for the TINH LPA humanoid (21 actuated joints)."""

LPA_ACTION_SCALE = {}
for _a in LPA_CFG.actuators.values():
    _e = _a.effort_limit_sim
    _s = _a.stiffness
    for _n in _a.joint_names_expr:
        LPA_ACTION_SCALE[_n] = 0.125 * _e / _s

LPA_MINIMAL_CFG = LPA_CFG.copy()
