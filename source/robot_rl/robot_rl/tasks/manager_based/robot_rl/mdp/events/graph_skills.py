# TINH behavior-graph skill sampler (tinh-lpa-clfrl.8.5d).

from __future__ import annotations

import torch


def graph_skill_sampler(env, env_ids, command_name: str,
                        enter_name: str, exit_name: str,
                        p_enter: float = 0.5,
                        loco_entry_time: float = 0.45):
    """Training curriculum for behavior-graph sequencing: drive envs
    through stand -> ENTER segment -> saturated hold -> EXIT segment
    -> locomotion, via the 8.5c handoff API.

    Interval-mode event (env_ids = all). Per call:
      - locomotion envs LOCKED in a phase hold (velocity command
        small) draw p_enter to queue the enter segment;
      - envs whose enter segment saturated queue the exit;
      - envs whose exit saturated return to velocity-conditioned
        locomotion, entering the gait at its touchdown phase.
    """
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
