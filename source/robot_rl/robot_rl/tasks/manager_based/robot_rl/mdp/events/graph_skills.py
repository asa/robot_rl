# TINH behavior-graph skill curricula (tinh-lpa-clfrl.8.5d / 7.7
# graph retrain).

from __future__ import annotations

import torch


def graph_skill_sampler(env, env_ids, command_name: str,
                        enter_name: str, exit_name: str,
                        p_enter: float = 0.5,
                        loco_entry_time: float = 0.45):
    """8.5d smoke curriculum: stand -> ENTER segment -> hold -> EXIT
    -> locomotion (episodic-only; kept for the smoke's env cfg)."""
    cmd = env.command_manager.get_term(command_name)
    lib = cmd.manager
    ei = lib.ref_id_of(enter_name)
    xi = lib.ref_id_of(exit_name)
    phi = cmd.get_phasing_var()
    no_pending = cmd.pending_ref_id <= cmd._NO_PENDING
    epi = cmd._active_episodic_mask()
    sat = epi & (phi >= 1.0 - 1e-6)

    loco_hold = ((cmd.active_ref_id < 0) & (cmd.hold_phi_value >= 0)
                 & no_pending)
    if loco_hold.any():
        draw = torch.rand(int(loco_hold.sum()),
                          device=env.device) < p_enter
        ids = torch.nonzero(loco_hold).flatten()[draw]
        if len(ids):
            cmd.set_next_ref(ids, ei)

    on_enter = sat & (cmd.active_ref_id == ei) & no_pending
    if on_enter.any():
        cmd.set_next_ref(torch.nonzero(on_enter).flatten(), xi)

    on_exit = sat & (cmd.active_ref_id == xi) & no_pending
    if on_exit.any():
        cmd.set_next_ref(torch.nonzero(on_exit).flatten(), -1,
                         entry_time=loco_entry_time)


# Graph-turn sequence states.
_FREE, _STOPPING, _TURNING, _STARTING = 0, 1, 2, 3
# The certified stop's dwell splice phases in the stomp cycle
# (L-forward dwell / mirrored R-forward dwell).
_DWELL_L, _DWELL_R = 0.397, 0.897


def graph_turn_sampler(env, env_ids, command_name: str,
                       p_turn: float = 0.02,
                       turn_periods: float = 2.0,
                       loco_entry_time: float = 0.45):
    """Graph-turn curriculum (pendulum9b post-mortem, 2026-08-12):
    turning is a GRAPH TRAVERSAL — walk -> walk_to_stand (handedness
    matched to the nearest dwell) -> turn cycle entered at ITS stand
    (phase 0) -> stand_to_walk -> locomotion. 9b swapped references
    instantly at arbitrary phase and never learned to turn.

    While an env runs the sequence, its VELOCITY COMMAND is pinned to
    the skill's own rate (turn: (0,0,+-0.269); stop/start: forward
    taper) and the closed-loop heading controller is disabled — the
    9b run had the vel-tracking rewards fighting the turn reference.

    p_turn is per SAMPLER TICK (0.5 s): ~4%/s of walking envs enter
    the sequence.
    """
    cmd = env.command_manager.get_term(command_name)
    vel = env.command_manager.get_term("base_velocity")
    lib = cmd.manager
    stopL = lib.ref_id_of("walk_to_stand")
    stopR = lib.ref_id_of("walk_to_stand_R")
    s2w = lib.ref_id_of("stand_to_walk")
    turnL = lib.ref_id_of("lpa_turn_left")
    turnR = lib.ref_id_of("lpa_turn_right")

    if not hasattr(cmd, "_graph_state"):
        cmd._graph_state = torch.zeros(env.num_envs, dtype=torch.long,
                                       device=env.device)
        cmd._turn_until = torch.zeros(env.num_envs, device=env.device)
        cmd._turn_dir = torch.zeros(env.num_envs, dtype=torch.long,
                                    device=env.device)
    st = cmd._graph_state
    # Episode resets drop graph state (the cmd term already clears
    # its own active/pending refs on true resets).
    st[env.episode_length_buf == 0] = _FREE

    t_now = env.episode_length_buf.float() * env.step_dt
    phi = cmd.get_phasing_var()
    uphi = cmd.unmasked_phasing_var
    no_pending = cmd.pending_ref_id <= cmd._NO_PENDING
    epi = cmd._active_episodic_mask()
    sat = epi & (phi >= 1.0 - 1e-6)

    # 1) walking envs draw the sequence: stop at the NEAREST dwell
    # (handedness-matched splice).
    walking = ((cmd.active_ref_id < 0) & (st == _FREE) & no_pending
               & (cmd.hold_phi_value < 0))
    if walking.any():
        draw = torch.rand(int(walking.sum()), device=env.device) < p_turn
        ids = torch.nonzero(walking).flatten()[draw]
        if len(ids):
            # nearest dwell in phase-future
            u = uphi[ids]
            dl = torch.where(u <= _DWELL_L, _DWELL_L - u,
                             1.0 - u + _DWELL_L)
            dr = torch.where(u <= _DWELL_R, _DWELL_R - u,
                             1.0 - u + _DWELL_R)
            use_l = dl <= dr
            for mask, ref, ph in ((use_l, stopL, _DWELL_L),
                                  (~use_l, stopR, _DWELL_R)):
                sub = ids[mask]
                if len(sub):
                    cmd.set_next_ref(sub, ref, exit_phase=ph)
            st[ids] = _STOPPING
            cmd._turn_dir[ids] = (torch.rand(len(ids), device=env.device)
                                  < 0.5).long()

    # 2) stop saturated -> enter the turn cycle at ITS stand (phase 0).
    stopped = (sat & ((cmd.active_ref_id == stopL)
                      | (cmd.active_ref_id == stopR))
               & (st == _STOPPING) & no_pending)
    if stopped.any():
        ids = torch.nonzero(stopped).flatten()
        for d, ref in ((0, turnL), (1, turnR)):
            sub = ids[cmd._turn_dir[ids] == d]
            if len(sub):
                cmd.set_next_ref(sub, ref)
                cmd._turn_until[sub] = (t_now[sub] + turn_periods
                                        * float(lib.total_times[ref]))
        st[ids] = _TURNING

    # 3) turn ran its periods -> leave at the loop's stand crossing.
    turning = ((st == _TURNING) & (cmd.active_ref_id >= 0)
               & (t_now >= cmd._turn_until) & no_pending)
    if turning.any():
        ids = torch.nonzero(turning).flatten()
        cmd.set_next_ref(ids, s2w, exit_phase=0.02)
        st[ids] = _STARTING

    # 4) start saturated -> locomotion at the touchdown phase;
    # the velocity term resumes its own sampling shortly.
    started = (sat & (cmd.active_ref_id == s2w) & (st == _STARTING)
               & no_pending)
    if started.any():
        ids = torch.nonzero(started).flatten()
        cmd.set_next_ref(ids, -1, entry_time=loco_entry_time)
        vel.time_left[ids] = 2.0
        st[ids] = _FREE

    # 5) pin the velocity command to the ACTIVE skill (the 9b run had
    # vel-tracking rewards fighting the turn reference) and disable
    # the closed-loop heading override for sequence envs.
    seq = st != _FREE
    if seq.any():
        ids = torch.nonzero(seq).flatten()
        tgt = torch.zeros(len(ids), 3, device=env.device)
        a = cmd.active_ref_id[ids]
        tgt[(a == stopL) | (a == stopR), 0] = 0.2   # decelerating
        tgt[a == s2w, 0] = 0.45                      # accelerating
        tl = a == turnL
        tr = a == turnR
        tgt[tl, 2] = float(lib._conditioning_vars_sorted[turnL][2])
        tgt[tr, 2] = float(lib._conditioning_vars_sorted[turnR][2])
        vel.vel_target_b[ids] = tgt
        vel.time_left[ids] = 100.0  # no resample mid-sequence
        vel.is_closed_loop_yaw_env[ids] = False
        vel.is_closed_loop_env[ids] = False
        if hasattr(vel, "is_standing_env"):
            vel.is_standing_env[ids] = False


def graph_nan_tripwire(env, env_ids, command_name: str):
    """Forensic guard (graphturn1 crashed on a sudden NaN at iter
    ~101.2k with healthy rewards — a rare discrete event): every
    tick, verify the tracking pipeline is finite; on trigger, DUMP
    the offending envs' graph state and clamp so training survives
    long enough to log more events."""
    cmd = env.command_manager.get_term(command_name)
    bad = ~torch.isfinite(cmd.y_des).all(dim=1)
    bad |= ~torch.isfinite(cmd.dy_des).all(dim=1)
    bad |= ~torch.isfinite(cmd.phasing_var)
    vel = env.command_manager.get_term("base_velocity")
    bad |= ~torch.isfinite(vel.command).all(dim=1)
    # PHYSICS blowup detector (graphturn2 forensics: the ref-side
    # tensors stayed finite through both crashes — the classic Isaac
    # hole is a sim explosion whose NaN root state fails every
    # termination COMPARISON and never resets, poisoning the batch).
    robot = env.scene["robot"]
    phys_bad = (~torch.isfinite(robot.data.root_pos_w).all(dim=1)
                | ~torch.isfinite(robot.data.root_quat_w).all(dim=1)
                | ~torch.isfinite(robot.data.joint_pos).all(dim=1)
                | ~torch.isfinite(robot.data.joint_vel).all(dim=1))
    # Also flag merely ESCAPING envs (|xy| or z insane but finite).
    phys_bad |= robot.data.root_pos_w[:, 2].abs() > 5.0
    phys_bad |= robot.data.root_pos_w[:, :2].abs().max(dim=1).values > 500.0
    if bad.any() or phys_bad.any():
        ids = torch.nonzero(bad | phys_bad).flatten()[:8]
        st = getattr(cmd, "_graph_state", None)
        for i in ids.tolist():
            print(f"[NAN-TRIPWIRE] env {i}: phys={bool(phys_bad[i])} "
                  f"ref={bool(bad[i])} active "
                  f"{int(cmd.active_ref_id[i])} pending "
                  f"{int(cmd.pending_ref_id[i])} phi "
                  f"{float(cmd.phasing_var[i]):.4f} ref_t0 "
                  f"{float(cmd.ref_start_time[i]):.3f} state "
                  f"{int(st[i]) if st is not None else -9} eplen "
                  f"{int(env.episode_length_buf[i])} vel "
                  f"{vel.command[i].tolist()} rootz "
                  f"{float(robot.data.root_pos_w[i, 2]):.2f}")
        cmd.y_des = torch.nan_to_num(cmd.y_des)
        cmd.dy_des = torch.nan_to_num(cmd.dy_des)
        cmd.phasing_var = torch.nan_to_num(cmd.phasing_var)
        if phys_bad.any():
            pids = torch.nonzero(phys_bad).flatten()
            # Scrub the NaN state in-sim FIRST (a timeout reset alone
            # still feeds one poisoned obs batch to PPO), then force
            # the natural timeout reset path.
            default = robot.data.default_root_state[pids].clone()
            default[:, :3] += env.scene.env_origins[pids]
            robot.write_root_pose_to_sim(default[:, :7], env_ids=pids)
            robot.write_root_velocity_to_sim(default[:, 7:13],
                                             env_ids=pids)
            robot.write_joint_state_to_sim(
                robot.data.default_joint_pos[pids],
                robot.data.default_joint_vel[pids], env_ids=pids)
            env.episode_length_buf[pids] = env.max_episode_length
