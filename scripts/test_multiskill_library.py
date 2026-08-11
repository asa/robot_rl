"""Gate for the multi-skill LibraryManager (tinh-lpa-clfrl.8.5b).

Loads a MIXED library folder (velocity-conditioned periodic gaits +
episodic behavior segments from export_clfrl_library --episodic) and
verifies the selection semantics:

  - locomotion (ref_id -1): weighted nearest-gait over the PERIODIC
    pool only — an episodic segment's zeroed conditioner must never
    win a velocity command (not even (0,0,0));
  - explicit ref_id: returns exactly that trajectory;
  - registry: names, episodic mask, per-traj total times.

  .venv/bin/python scripts/test_multiskill_library.py <library_dir>
"""

import sys

import torch

sys.path.insert(0, "source/robot_rl")
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.library_manager import (  # noqa: E402
    LibraryManager,
)

def main() -> int:
    folder = sys.argv[1]
    lib = LibraryManager(folder, hf_repo=None, device="cpu")

    names = lib.ref_names
    epi = lib.episodic_mask
    print("library:", list(zip(names, epi.tolist(),
                               lib.total_times.tolist())))
    assert "laser_enter" in names and "laser_exit" in names, names
    n_epi = int(epi.sum())
    assert n_epi == 2, f"expected 2 episodic segments, got {n_epi}"
    assert len(lib.periodic_indices) == len(names) - 2

    ei = lib.ref_id_of("laser_enter")
    xi = lib.ref_id_of("laser_exit")
    assert bool(epi[ei]) and bool(epi[xi])
    assert abs(float(lib.total_times[ei]) - 1.95) < 0.01
    assert abs(float(lib.total_times[xi]) - 1.10) < 0.01

    # Locomotion selection never lands on an episodic segment — probe
    # the velocity envelope incl. the standing command (0,0,0), which
    # ties an episodic zero conditioner without the pool restriction.
    probes = torch.tensor([
        [0.56, 0.0, 0.0],
        [0.0, 0.0, 0.289],
        [0.0, 0.0, -0.289],
        [0.0, 0.0, 0.0],
        [0.2, 0.0, 0.1],
    ])
    idx = lib.get_traj_indices(probes)
    assert not epi[idx].any(), (
        f"locomotion lookup landed on an episodic segment: "
        f"{[names[i] for i in idx.tolist()]}")
    print("locomotion probes ->", [names[i] for i in idx.tolist()])

    # Explicit selection wins regardless of the velocity command.
    ref_ids = torch.tensor([ei, xi, -1])
    idx = lib.get_traj_indices(probes[:3], ref_ids)
    assert int(idx[0]) == ei and int(idx[1]) == xi
    assert not epi[idx[2]]
    print("explicit selection ->", [names[i] for i in idx.tolist()])

    print("MULTISKILL LIBRARY GATE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
