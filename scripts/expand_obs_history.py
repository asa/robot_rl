"""Widen an actor's first layer for per-term observation HISTORY.

Adding history to an observation term does NOT append to the end of
the observation vector -- it expands that term IN PLACE, from d to
history_length*d, shifting every later term along. So
pad_checkpoint_obs.py (which appends zeros) cannot be used: every
weight after the first expanded term would read the wrong input.

IsaacLab's CircularBuffer documents "most recent entry at the end,
oldest at the beginning", so within each expanded block the CURRENT
frame is the LAST d columns. This script places the old weights
there and zeros the older frames, which reproduces the original
policy exactly on the first step after a reset -- when the history
buffer is filled by repeating the current observation, every frame
is identical, and the zeroed columns contribute nothing either way.

Only the ACTOR is touched. The critic keeps its privileged
observations unchanged (asymmetric actor-critic), so it needs no
remap.

  python scripts/expand_obs_history.py in.pt out.pt \
      --terms base_ang_vel:3,projected_gravity:3,velocity_commands:3,\
joint_pos:21,joint_vel:21,actions:21,sin_phase:1,cos_phase:1 \
      --history 5 \
      --history-terms base_ang_vel,projected_gravity,joint_pos,joint_vel,actions

Verify the term order against the "Active Observation Terms in Group:
'policy'" table the env prints at startup. Getting the order wrong
silently scrambles the policy.
"""

import argparse

import torch


def parse_terms(spec: str):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, dim = part.partition(":")
        out.append((name.strip(), int(dim)))
    return out


def column_map(terms, history: int, hist_names: set):
    """old column index -> new column index, for every old column.

    Also returns the new width. Old columns of an expanded term land
    in the LAST frame of its new block (the most recent entry).
    """
    mapping = {}
    old_off = new_off = 0
    for name, dim in terms:
        if name in hist_names:
            block = history * dim
            # most recent frame sits at the end of the block
            base = new_off + block - dim
        else:
            block = dim
            base = new_off
        for i in range(dim):
            mapping[old_off + i] = base + i
        old_off += dim
        new_off += block
    return mapping, new_off, old_off


def remap_first_layer(w: torch.Tensor, mapping: dict, new_width: int):
    out = torch.zeros(w.shape[0], new_width, dtype=w.dtype)
    for old_i, new_i in mapping.items():
        out[:, new_i] = w[:, old_i]
    return out


def verify(w_old, w_new, terms, history, hist_names, tol=1e-5):
    """Behavioural proof, not an index-arithmetic argument.

    Build a random observation, replicate it across every history
    frame exactly as the buffer does immediately after a reset, and
    require the two layers to produce the same output.
    """
    torch.manual_seed(0)
    old_obs, new_parts = [], []
    for name, dim in terms:
        v = torch.randn(1, dim, dtype=w_old.dtype)
        old_obs.append(v)
        if name in hist_names:
            new_parts.append(v.repeat(1, history))
        else:
            new_parts.append(v)
    old_vec = torch.cat(old_obs, dim=1)
    new_vec = torch.cat(new_parts, dim=1)
    a = old_vec @ w_old.T
    b = new_vec @ w_new.T
    err = (a - b).abs().max().item()
    return err, err <= tol


def main() -> int:
    ap = argparse.ArgumentParser(prog="expand_obs_history")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--terms", required=True,
                    help="ordered name:dim list for the POLICY group")
    ap.add_argument("--history", type=int, required=True)
    ap.add_argument("--history-terms", required=True,
                    help="comma-separated terms that gain history")
    args = ap.parse_args()

    terms = parse_terms(args.terms)
    hist_names = {t.strip() for t in args.history_terms.split(",") if t.strip()}
    unknown = hist_names - {n for n, _ in terms}
    if unknown:
        print(f"FAIL: --history-terms names not in --terms: {sorted(unknown)}")
        return 2

    mapping, new_width, old_width = column_map(terms, args.history, hist_names)
    print(f"policy obs {old_width} -> {new_width} "
          f"(history={args.history} on {len(hist_names)} terms)")

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    key = next((k for k in sd if k.endswith("actor.0.weight")), None)
    if key is None:
        print("FAIL: no actor.0.weight in the checkpoint")
        return 2
    w_old = sd[key]
    if w_old.shape[1] != old_width:
        print(f"FAIL: actor.0.weight has {w_old.shape[1]} input columns "
              f"but --terms sums to {old_width}. The term list does not "
              "match this checkpoint -- check the startup table.")
        return 2

    w_new = remap_first_layer(w_old, mapping, new_width)
    err, ok = verify(w_old, w_new, terms, args.history, hist_names)
    print(f"equivalence check (all history frames = current): "
          f"max|old-new| = {err:.3e} -> {'OK' if ok else 'FAILED'}")
    if not ok:
        print("FAIL: refusing to write a checkpoint that changes the policy")
        return 1

    sd[key] = w_new
    torch.save(ckpt, args.dst)
    print(f"wrote {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
