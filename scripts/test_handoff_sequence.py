"""Gate for the behavior-graph handoff (tinh-lpa-clfrl.8.5c).

Drives the REAL TrajectoryCommand handoff/phase logic (no Isaac env —
stubbed command manager + clock) through a full graph walk on the
mixed library:

  locomotion (0.56) -> vel 0 -> phase-hold locks -> laser_enter fires
  -> phi saturates 1 (hold) -> laser_exit fires -> phi saturates
  -> return to locomotion fires

and asserts: pendings only fire at hold points, segment clocks rebase
(phi starts at 0), episodic saturation holds, and the reference
never TELEPORTS — max per-step joint-reference delta stays walking-
scale except at sanctioned handoff steps, where the seam must still
be small (the segments were solved from the shared stand).

  .venv/bin/python scripts/test_handoff_sequence.py <library_dir>
"""

import importlib.util
import sys
import types

import torch

_PKG = "robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking"
_DIR = "source/robot_rl/robot_rl/tasks/manager_based/robot_rl/mdp/commands/traj_tracking"

parts = _PKG.split(".")
for i in range(len(parts)):
    name = ".".join(parts[: i + 1])
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = []
        sys.modules[name] = mod
sys.modules[_PKG].__path__ = [_DIR]

def _load(mod_name: str):
    full = f"{_PKG}.{mod_name}"
    spec = importlib.util.spec_from_file_location(
        full, f"{_DIR}/{mod_name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod

_load("manager_base")
tm = _load("trajectory_manager")
LibraryManager = _load("library_manager").LibraryManager

# trajectory_cmd imports isaaclab at module level — stub the two
# symbols it needs (CommandTerm base + math utils it doesn't touch
# on this code path).
isaaclab_managers = types.ModuleType("isaaclab.managers")
isaaclab_managers.CommandTerm = object
isaaclab_math = types.ModuleType("isaaclab.utils.math")
for fn in ("wrap_to_pi", "quat_apply", "quat_mul", "quat_from_euler_xyz",
           "euler_xyz_from_quat", "yaw_quat", "quat_inv"):
    setattr(isaaclab_math, fn, lambda *a, **k: None)
isaaclab_pkg = types.ModuleType("isaaclab")
isaaclab_utils = types.ModuleType("isaaclab.utils")
sys.modules.setdefault("isaaclab", isaaclab_pkg)
sys.modules.setdefault("isaaclab.managers", isaaclab_managers)
sys.modules.setdefault("isaaclab.utils", isaaclab_utils)
sys.modules.setdefault("isaaclab.utils.math", isaaclab_math)
sys.modules.setdefault("isaaclab.utils.math", isaaclab_math)
_load("clf")
TrajectoryCommand = _load("trajectory_cmd").TrajectoryCommand
TrajectoryType = tm.TrajectoryType

N = 1
DT = 0.02


class VelTerm:
    def __init__(self):
        self.command = torch.tensor([[0.56, 0.0, 0.0]])


class CmdMgr:
    def __init__(self, term):
        self._term = term
    def get_term(self, name):
        return self._term
    def get_command(self, name):
        return self._term.command


class EnvStub:
    def __init__(self, term):
        self.command_manager = CmdMgr(term)
        self.episode_length_buf = torch.zeros(N, dtype=torch.long)


def main() -> int:
    folder = sys.argv[1]
    vel = VelTerm()
    env = EnvStub(vel)

    cmd = TrajectoryCommand.__new__(TrajectoryCommand)
    cmd.num_envs = N
    cmd.device = "cpu"
    cmd.env = env
    cmd.manager_type = "library"
    cmd.manager = LibraryManager(folder, None, "cpu", env=env,
                                 conditioner_generator_name="base_velocity")
    cmd.manager.owner = cmd
    cmd.trajectory_type = cmd.manager.trajectory_type
    cmd.cfg = types.SimpleNamespace(hold_phi_threshold=0.1,
                                    phasing_boundaries=2)
    cmd.phasing_var = torch.zeros(N)
    cmd.unmasked_phasing_var = torch.zeros(N)
    cmd.prev_unmasked_phasing_var = torch.zeros(N)
    cmd.should_hold = torch.zeros(N, dtype=torch.bool)
    cmd.boundaries_crossed = torch.zeros(N, dtype=torch.int)
    cmd.hold_phi_value = -1.0 * torch.ones(N)
    cmd._NO_PENDING = -2
    cmd.active_ref_id = torch.full((N,), -1, dtype=torch.long)
    cmd.pending_ref_id = torch.full((N,), -2, dtype=torch.long)
    cmd.ref_start_time = torch.zeros(N)

    lib = cmd.manager
    names = lib.ref_names
    joint_cols = [i for i, n in enumerate(lib.pos_output_names)
                  if "pos_" not in n and "ori_" not in n]
    ei, xi = lib.ref_id_of("laser_enter"), lib.ref_id_of("laser_exit")

    events = []          # (step, what)
    max_step_jump = 0.0  # outside handoff steps
    handoff_jumps = []
    prev_y = None
    prev_active = cmd.active_ref_id.clone()

    def script(k, t):
        # The supervisory sequence a game/curriculum layer would run.
        if k == 150:                    # 3 s of walking -> stop
            vel.command[:] = torch.tensor([[0.0, 0.0, 0.0]])
        if k == 400:                    # queued while holding
            cmd.set_next_ref(torch.tensor([0]), ei)
        if k == 650:                    # queued while enter still playing
            cmd.set_next_ref(torch.tensor([0]), xi)
        if k == 900:
            cmd.set_next_ref(torch.tensor([0]), -1)
        if k == 950:
            vel.command[:] = torch.tensor([[0.56, 0.0, 0.0]])

    phi_log = []
    for k in range(1200):
        env.episode_length_buf += 1     # never 0: no episode resets
        t = torch.tensor([k * DT])
        script(k, t)
        lib.invalidate_cache()
        cmd._process_handoffs(t)
        t_ref = t - cmd.ref_start_time
        phi = cmd.update_phasing_var(t_ref)
        y_pos, y_vel = lib.get_output(t_ref)
        # per-env episodic saturation (the get_desired_outputs path)
        epi = cmd._active_episodic_mask()
        sat = epi & (phi >= 1.0 - 1e-6)

        if bool((cmd.active_ref_id != prev_active).any()):
            events.append((k, f"handoff -> "
                           f"{'locomotion' if cmd.active_ref_id[0] < 0 else names[cmd.active_ref_id[0]]}"))
            if prev_y is not None:
                handoff_jumps.append(float(
                    (y_pos[0, joint_cols] - prev_y[0, joint_cols])
                    .abs().max()))
        elif prev_y is not None:
            max_step_jump = max(max_step_jump, float(
                (y_pos[0, joint_cols] - prev_y[0, joint_cols])
                .abs().max()))
        prev_y = y_pos
        prev_active = cmd.active_ref_id.clone()
        phi_log.append((k, float(phi[0]), int(cmd.active_ref_id[0]),
                        bool(sat[0])))

    for e in events:
        print("event:", e)
    print(f"max per-step joint jump (non-handoff): {max_step_jump:.4f}")
    print(f"handoff seams: {[round(j, 4) for j in handoff_jumps]}")

    # --- assertions ---
    # 1. exactly 3 handoffs, in order enter -> exit -> locomotion
    kinds = [w for _, w in events]
    assert kinds == [f"handoff -> laser_enter", "handoff -> laser_exit",
                     "handoff -> locomotion"], kinds
    # 2. enter fired only after the phase hold locked (>= step 150)
    assert events[0][0] >= 400, events[0]
    # 3. exit fired only after enter saturated (1.95 s after enter)
    enter_k, exit_k = events[0][0], events[1][0]
    assert (exit_k - enter_k) * DT >= 1.90, (enter_k, exit_k)
    # 4. episodic segments saturated and held (phi == 1 with sat flag)
    sat_steps = [s for s in phi_log
                 if s[2] >= 0 and s[3]]
    assert len(sat_steps) > 50, len(sat_steps)
    # 5. phi rebased at each handoff (starts < 0.1 right after)
    for hk, _ in events[:2]:
        phi_after = [p for kk, p, a, _ in phi_log if kk == hk][0]
        assert phi_after < 0.1, (hk, phi_after)
    # 6. no teleports: continuous steps stay walking-scale; handoff
    #    seams small (segments share the stand fixture)
    assert max_step_jump < 0.12, max_step_jump
    assert all(j < 0.30 for j in handoff_jumps), handoff_jumps

    print("HANDOFF SEQUENCE GATE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
