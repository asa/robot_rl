def overhead_traveller(
    env,
    env_ids,
    asset_cfg,
    lift_kg: float,
):
    """Hold the robot on the overhead traveller it never runs without.

    The LPA is always on the rig for safety (user, 2026-08-28), so a
    sim without it trains against a machine that does not exist — and
    a HARDER one, since the rig carries part of the weight. On the
    clad 110.5 kg robot that is not academic: the reference gait
    saturates knees, hips, ankle, waist yaw and both shoulders at 100%
    of limit.

    NOT isaaclab's apply_external_force_torque. That helper never
    passes is_global, so its wrench lands in the BODY frame and would
    tilt with the torso — a rope does not roll with the robot. This
    applies world +z.

    The rope is CONTINUOUS through both waist attachments, so tension
    equalises along it and the two legs cannot pull unequally: a
    single vertical force, no couple. (Modelling them as independent
    tensions manufactures roll stability the rig does not have, and
    the error is invisible until the robot leans.)

    Set as an interval event, not reset: the wrench composer holds a
    value once set, but re-asserting keeps it correct across resets
    and makes the lift visible in the event list rather than implied.
    """
    import torch

    asset = env.scene[asset_cfg.name]
    n = len(env_ids)
    ids = asset_cfg.body_ids
    num_bodies = len(ids) if isinstance(ids, list) else asset.num_bodies

    forces = torch.zeros((n, num_bodies, 3), device=asset.device)
    forces[..., 2] = lift_kg * 9.81 / max(num_bodies, 1)
    torques = torch.zeros_like(forces)

    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=forces,
        torques=torques,
        body_ids=ids,
        env_ids=env_ids,
        is_global=True,        # a rope pulls world-up, not body-up
    )
