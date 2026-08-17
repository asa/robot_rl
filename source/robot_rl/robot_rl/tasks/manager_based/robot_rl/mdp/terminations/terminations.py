# Copyright 2026 Zachary Olkin. All rights reserved.

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

def no_progress(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Terminates the episode early if the robot is not making enough progress
    compared to expected distance at current time step.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")

    # Distance traveled from starting point
    root_pos = asset.data.root_pos_w[:, :2]
    origin = env.scene.env_origins[:, :2]
    distance = torch.norm(root_pos - origin, dim=1)

    # Expected distance so far = commanded_speed * time_elapsed
    commanded_speed = torch.norm(command[:, :2], dim=1)
    elapsed_time = env.episode_length_buf * env.step_dt  # [num_envs]
    expected_distance = commanded_speed * elapsed_time

    # Flag for insufficient progress
    behind_schedule = distance < (0.5 * expected_distance)

    # Optional: only trigger after a minimum time has passed (e.g., 30% of episode)
    enough_time_passed = env.episode_length_buf > (0.5 * env.max_episode_length)
    no_progress_flag = behind_schedule & enough_time_passed

    return no_progress_flag

def base_orientation(env, cmd_name: str, roll_limit_deg: float = 30.0, pitch_limit_deg: float = 30.0,
                     base_link: str = "pelvis_link",
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Terminates the episode if the robot's base orientation exceeds certain limits.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(cmd_name)
    ref_traj = cmd.y_des
    output_names = cmd.ordered_output_names

    pitch_idx = output_names.index(f"{base_link}:ori_y")
    roll_idx = output_names.index(f"{base_link}:ori_x")

    # Get base orientation in Euler angles
    root_quat = asset.data.root_quat_w  # [num_envs, 4]
    root_euler = euler_xyz_from_quat(root_quat, wrap_to_2pi=False)  # [num_envs, 3]

    roll_error = root_euler[0][:] - ref_traj[:, roll_idx]
    pitch_error = root_euler[1][:] - ref_traj[:, pitch_idx]

    # Define orientation limits (in radians)
    roll_limit = torch.deg2rad(torch.tensor(roll_limit_deg))  # ±30 degrees
    pitch_limit = torch.deg2rad(torch.tensor(pitch_limit_deg))  # ±30 degrees

    # Check if limits are exceeded
    roll_exceeded = (roll_error.abs() > roll_limit)
    pitch_exceeded = (pitch_error.abs() > pitch_limit)

    orientation_flag = roll_exceeded | pitch_exceeded

    return orientation_flag

def runaway_dynamics(env, base_vel_limit: float = 10.0,
                     joint_vel_limit: float = 60.0):
    """Terminate envs whose simulation runs away with HUGE-BUT-FINITE
    dynamics (tinh graphturn forensics 2026-08-12: PhysX explosions
    that never reach NaN — joint velocities ~1e8 — lived out whole
    episodes because nothing terminated them; the quadratic CLF
    turned those states into -1e32 rewards and froze learning).
    Limits sit far above the hardware envelope (HEJ no-load 17.8
    rad/s), so healthy motion never trips."""
    import torch
    robot = env.scene["robot"]
    bad = robot.data.root_lin_vel_w.norm(dim=1) > base_vel_limit
    bad |= robot.data.joint_vel.abs().max(dim=1).values > joint_vel_limit
    bad |= ~torch.isfinite(robot.data.root_pos_w).all(dim=1)
    bad |= ~torch.isfinite(robot.data.joint_vel).all(dim=1)
    return bad


def waist_twist(env, limit: float = 2.2):
    """Terminate envs riding WAIST_YAW toward its +-pi joint limit
    (graphturn14 twist probe 2026-08-14: 64/64 envs walked with the
    torso twisted 90-180 deg — the chest is not a tracked body, so
    a pi-twist was nearly free). Every reference keeps the waist
    within ~0.3 rad; 2.2 rad is unreachable by honest tracking."""
    import torch
    robot = env.scene["robot"]
    idx = robot.data.joint_names.index("WAIST_YAW")
    return robot.data.joint_pos[:, idx].abs() > limit


def base_height_above_feet(env, minimum_height: float = 0.6,
                           feet_names: tuple = ("L_ANKLE", "R_ANKLE")):
    """Fall detector measured against the robot's OWN FEET, not world
    zero (tinh-lpa-ramp.5).

    IsaacLab's `root_height_below_minimum` documents itself as flat-
    terrain-only ("the minimum height is in the world frame"). On the
    ramp terrain the uphill columns spawn at the bottom of an inverted
    pyramid — absolute root z 0.30-0.39 m against a 0.6 m threshold —
    so a perfectly upright robot was terminated at every reset (spawn
    audit: exactly the up10+up12 envs, 21 of them, every step).

    Base-above-feet is terrain-independent and is what "fallen"
    actually means: standing clearance is ~1.0 m on any slope.
    """
    import torch
    robot = env.scene["robot"]
    if not hasattr(env, "_feet_idx_cache"):
        env._feet_idx_cache = [robot.body_names.index(n)
                               for n in feet_names]
    feet_z = robot.data.body_pos_w[:, env._feet_idx_cache, 2]
    return (robot.data.root_pos_w[:, 2] - feet_z.min(dim=1).values
            ) < minimum_height
