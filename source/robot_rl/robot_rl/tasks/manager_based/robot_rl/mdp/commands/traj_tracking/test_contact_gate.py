"""Pin the contact gate's semantics — especially the two bugs the
first versions shipped with.

Runs without Isaac, in milliseconds — but ONLY from this directory
with conftest discovery cut, because the repo's parent conftests pull
in Isaac fixtures that error at setup:

    cd .../traj_tracking && python -m pytest test_contact_gate.py --confcutdir=.
"""

import os
import sys

import torch

# Import the sibling module directly rather than through the package:
# the package __init__ chain imports isaaclab, and the whole point of
# this test is running without Isaac.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact_gate import contact_gate_step  # noqa: E402


def run_sequence(expected_seq, measured_seq, dt=0.02, cap=0.20):
    """Drive the gate through [T, C] bool sequences for one env.
    Returns per-step (holding, alpha, offset)."""
    C = len(expected_seq[0])
    prev = torch.tensor([expected_seq[0]], dtype=torch.bool)
    awaiting = torch.zeros(1, C, dtype=torch.bool)
    elapsed = torch.zeros(1)
    offset = torch.zeros(1)
    out = []
    for k, (e, m) in enumerate(zip(expected_seq, measured_seq)):
        fresh = torch.tensor([k == 0])
        exp = torch.tensor([e], dtype=torch.bool)
        mea = torch.tensor([m], dtype=torch.bool)
        prev, awaiting, elapsed, offset, holding, alpha = contact_gate_step(
            exp, prev, awaiting, elapsed, offset, mea, fresh, dt, cap)
        out.append((bool(holding[0]), float(alpha[0]), float(offset[0])))
    return out


def test_late_ground_holds_until_landing():
    """The case the gate exists for: reference enters stance, ground
    arrives 3 steps late. Hold for those 3 steps, then release."""
    E = [[0], [0], [1], [1], [1], [1], [1]]
    M = [[0], [0], [0], [0], [0], [1], [1]]
    out = run_sequence(E, M)
    assert [h for h, _, _ in out] == [False, False, True, True, True, False, False]
    assert abs(out[-1][2] - 0.06) < 1e-6  # 3 held steps * 20 ms (float32)
    # alpha is 0 exactly while holding, 1 otherwise
    assert [a for _, a, _ in out] == [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]


def test_early_toeoff_never_holds():
    """THE regression test for gate v1: the robot lifts a foot a
    moment before the reference ends its stance. expected=1,
    measured=0 — but with no rising edge this must NOT hold, or the
    gate fires on ordinary gait variation at nearly every step."""
    E = [[1], [1], [1], [1], [0], [0]]
    M = [[1], [1], [0], [0], [0], [0]]   # lifts 2 steps early
    out = run_sequence(E, M)
    assert all(not h for h, _, _ in out)
    assert out[-1][2] == 0.0


def test_hold_cap_releases_and_stays_released():
    """A fallen robot never lands the awaited foot. The clock must
    resume after the cap — and a stale arm must not re-hold later."""
    E = [[0]] + [[1]] * 30
    M = [[0]] * 31
    out = run_sequence(E, M, dt=0.02, cap=0.10)
    held = [h for h, _, _ in out]
    assert held[1] is True                      # arms on the edge
    assert sum(held) == 5                       # 0.10 s / 0.02 s
    assert not any(held[7:])                    # never re-holds
    assert abs(out[-1][2] - 0.10) < 1e-6        # offset capped


def test_mid_stance_force_dropout_never_holds():
    """A sensor dropout mid-stance is a level mismatch, not an edge."""
    E = [[1], [1], [1], [1], [1]]
    M = [[1], [1], [0], [1], [1]]
    out = run_sequence(E, M)
    assert all(not h for h, _, _ in out)


def test_per_foot_independence():
    """Foot A awaits ground; foot B's toe-off slop must not extend or
    trigger anything."""
    #        A  B
    E = [[0, 1], [1, 1], [1, 1], [1, 0], [1, 0]]
    M = [[0, 1], [0, 1], [1, 0], [1, 0], [1, 0]]
    out = run_sequence(E, M)
    # holds only at k=1 (A's rising edge, no ground); A lands at k=2
    assert [h for h, _, _ in out] == [False, True, False, False, False]


def test_fresh_episode_cannot_fabricate_an_edge():
    """Reset into double-support stance: expected jumps 0->1 across
    the reset boundary, but a fresh episode must not read that as a
    touchdown edge."""
    E = [[0], [0], [0]]
    M = [[0], [0], [0]]
    out1 = run_sequence(E, M)
    assert all(not h for h, _, _ in out1)
    # simulate reset directly: fresh step whose expected is stance
    prev = torch.tensor([[False]])
    awaiting = torch.tensor([[True]])   # stale arm from previous episode
    elapsed = torch.tensor([0.18])
    offset = torch.tensor([0.14])
    prev, awaiting, elapsed, offset, holding, alpha = contact_gate_step(
        torch.tensor([[True]]), prev, awaiting, elapsed, offset,
        torch.tensor([[False]]), torch.tensor([True]), 0.02, 0.20)
    assert not bool(holding[0])
    assert float(offset[0]) == 0.0
    assert not bool(awaiting[0, 0])


