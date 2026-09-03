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


_HOLDING = 4


def graph_stop_sampler(env, env_ids, command_name: str,
                       loco_entry_time: float = 0.45):
    """Rung 2 (am-m7l.2): a COMMANDED stop and start, as a graph
    traversal driven by the velocity command instead of a turn draw.

    walk --(standing command)--> walk_to_stand spliced at the nearest
    dwell (handedness-matched) --> saturated = a held STAND, tracked
    closed-loop by the policy as the turn sequence already does at its
    stand --(command rises)--> stand_to_walk --> locomotion.

    WHY NOT the phase hold: a standing env commands (0,0,0), which is
    below hold_phi_threshold and freezes the periodic reference
    mid-cycle -- the pose held with the arms wherever the cycle left
    them (clad3's render). A commanded stop should land in walk_to_stand
    and hold a stand, not freeze the walk; the env that uses this
    sampler sets hold_phi_threshold 0 so the freeze never fires and
    the traversal owns the stop.

    The standing draw IS the stop command: VelocityTrackingCommand
    marks rel_standing_envs of resamples as is_standing_env and zeroes
    their command until the next resample (7-10 s), which is the hold
    length training sees; a resample that draws walking again is the
    start command.

    While an env runs the sequence its velocity command is pinned to
    the skill's own rate (stop: forward taper 0.2; hold: 0; start:
    0.45) and the closed-loop heading controller is disabled, as the
    turn sampler does. The hold does NOT freeze resampling, so the
    start command can arrive.
    """
    cmd = env.command_manager.get_term(command_name)
    vel = env.command_manager.get_term("base_velocity")
    lib = cmd.manager
    stopL = lib.ref_id_of("walk_to_stand")
    stopR = lib.ref_id_of("walk_to_stand_R")
    s2w = lib.ref_id_of("stand_to_walk")
    if not hasattr(cmd, "_graph_state"):
        cmd._graph_state = torch.zeros(env.num_envs, dtype=torch.long,
                                       device=env.device)
    st = cmd._graph_state
    st[env.episode_length_buf == 0] = _FREE
    phi = cmd.get_phasing_var()
    uphi = cmd.unmasked_phasing_var
    no_pending = cmd.pending_ref_id <= cmd._NO_PENDING
    epi = cmd._active_episodic_mask()
    sat = epi & (phi >= 1.0 - 1e-6)
    standing = vel.is_standing_env
    # 1) walking envs that were commanded to stand: stop at the
    #    nearest dwell in phase-future (the certified splice).
    walking = ((cmd.active_ref_id < 0) & (st == _FREE) & no_pending
               & (cmd.hold_phi_value < 0))
    go_stop = walking & standing
    if go_stop.any():
        ids = torch.nonzero(go_stop).flatten()
        u = uphi[ids]
        dl = torch.where(u <= _DWELL_L, _DWELL_L - u, 1.0 - u + _DWELL_L)
        dr = torch.where(u <= _DWELL_R, _DWELL_R - u, 1.0 - u + _DWELL_R)
        use_l = dl <= dr
        for mask, ref, ph in ((use_l, stopL, _DWELL_L),
                              (~use_l, stopR, _DWELL_R)):
            sub = ids[mask]
            if len(sub):
                cmd.set_next_ref(sub, ref, exit_phase=ph)
        st[ids] = _STOPPING
    # 2) stop saturated -> HOLD the stand (the episodic holds its last
    #    pose at phi = 1; the policy tracks it).
    stopped = (sat & ((cmd.active_ref_id == stopL)
                      | (cmd.active_ref_id == stopR))
               & (st == _STOPPING) & no_pending)
    if stopped.any():
        st[torch.nonzero(stopped).flatten()] = _HOLDING
    # 3) held envs whose command came back -> stand_to_walk from the
    #    stand (a saturated source hands off at once).
    go_start = (st == _HOLDING) & (~standing) & no_pending
    if go_start.any():
        ids = torch.nonzero(go_start).flatten()
        cmd.set_next_ref(ids, s2w)
        st[ids] = _STARTING
    # 4) start saturated -> locomotion at the touchdown phase.
    started = (sat & (cmd.active_ref_id == s2w) & (st == _STARTING)
               & no_pending)
    if started.any():
        ids = torch.nonzero(started).flatten()
        cmd.set_next_ref(ids, -1, entry_time=loco_entry_time)
        vel.time_left[ids] = 2.0
        st[ids] = _FREE
    # 5) pin the velocity command to the active skill; the HOLD keeps
    #    resampling so the start command can arrive.
    seq = st != _FREE
    if seq.any():
        ids = torch.nonzero(seq).flatten()
        tgt = torch.zeros(len(ids), 3, device=env.device)
        a = cmd.active_ref_id[ids]
        tgt[(a == stopL) | (a == stopR), 0] = 0.2
        tgt[a == s2w, 0] = 0.45
        vel.vel_target_b[ids] = tgt
        moving = st[ids] != _HOLDING
        vel.time_left[ids[moving]] = 100.0
        vel.is_closed_loop_yaw_env[ids] = False
        vel.is_closed_loop_env[ids] = False
        # a stopping/starting env is not a "standing" env for the
        # velocity term (its command is the taper, not zero); a held
        # env stays standing so its command is exactly zero
        vel.is_standing_env[ids[moving]] = False


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
    # CLF-magnitude forensics: WHICH channel explodes (rewards
    # hit -1e32 with all finiteness guards silent and the runaway
    # termination live — so the blowup is huge-but-finite in a
    # channel none of the guards cover).
    v = getattr(cmd, "v", None)
    if v is not None and torch.isfinite(v).any() and float(
            torch.nan_to_num(v, nan=0.0, posinf=0.0).max()) > 1e6:
        i = int(torch.nan_to_num(v, nan=0.0, posinf=0.0).argmax())
        st_dbg = getattr(cmd, "_graph_state", None)
        print(f"[V-BLOWUP] env {i} v={float(v[i]):.3e} "
              f"|linvel|={float(robot.data.root_lin_vel_w[i].norm()):.2e} "
              f"|angvel|={float(robot.data.root_ang_vel_w[i].norm()):.2e} "
              f"|jvel|max={float(robot.data.joint_vel[i].abs().max()):.2e} "
              f"|y_act|max={float(cmd.y_act[i].abs().max()):.2e} "
              f"|y_des|max={float(cmd.y_des[i].abs().max()):.2e} "
              f"|dy_act|max={float(cmd.dy_act[i].abs().max()):.2e} "
              f"|dy_des|max={float(cmd.dy_des[i].abs().max()):.2e} "
              f"active={int(cmd.active_ref_id[i])} "
              f"state={int(st_dbg[i]) if st_dbg is not None else -9} "
              f"phi={float(cmd.phasing_var[i]):.3f} "
              f"eplen={int(env.episode_length_buf[i])}")
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


def ramp_ref_sampler(env, env_ids, command_name: str,
                     ref_names: list, slope_degs: list):
    """Pin each env to the reference gait matching ITS terrain slope
    (tinh-lpa-ramp.5).

    Slope is TERRAIN, not a command: the velocity conditioner cannot
    tell a +12 deg climb from a -4 deg descent (all the ramp gaits
    sit at vel_x 0.27-0.35, vel_yaw 0), so the reference is selected
    EXPLICITLY from `terrain_types` — which, with the ramp generator's
    one-sub-terrain-per-column layout, is an exact column index.

    Runs on an interval and (re)assigns any env sitting on locomotion
    (-1), which is the post-reset state — this avoids depending on
    reset ordering between the event and command managers.
    """
    import torch

    cmd = env.command_manager.get_term(command_name)
    lib = cmd.manager
    terrain = getattr(env.scene, "terrain", None)
    types = getattr(terrain, "terrain_types", None)
    if types is None:
        return

    if not hasattr(cmd, "_ramp_ref_ids"):
        cmd._ramp_ref_ids = torch.tensor(
            [lib.ref_id_of(n) for n in ref_names],
            dtype=torch.long, device=env.device)
        cmd._ramp_slope = torch.tensor(
            [float(d) for d in slope_degs], device=env.device)

    # Per-env slope, for the observation term below.
    col = types.clamp(max=len(ref_names) - 1).long()
    cmd._env_slope_deg = cmd._ramp_slope[col]

    need = cmd.active_ref_id < 0
    if not need.any():
        return
    ids = need.nonzero().flatten()
    want = cmd._ramp_ref_ids[col[ids]]

    # Assign DIRECTLY, not via set_next_ref: that stages a PENDING
    # ref which only fires at a hold point, and the ramp gaits are
    # periodic — their only hold is episode start, exactly when the
    # command term clears pending. Staged mid-episode it would never
    # fire (probe: 0/64 assigned). A ramp is a static per-episode
    # condition, not a spliced traversal, so the handoff machinery
    # buys nothing here. Mirrors _process_handoffs' bookkeeping.
    t_now = env.episode_length_buf.float() * env.step_dt
    cmd.active_ref_id[ids] = want
    cmd.pending_ref_id[ids] = cmd._NO_PENDING
    cmd.pending_exit_phase[ids] = -1.0
    cmd.pending_entry_time[ids] = 0.0
    cmd.ref_start_time[ids] = t_now[ids]
    cmd.hold_phi_value[ids] = -1.0
    cmd.boundaries_crossed[ids] = 0
    if cmd.manager_type == "library":
        cmd.manager.invalidate_cache()
