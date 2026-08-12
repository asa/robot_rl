"""Pad an rsl_rl checkpoint's obs input for new trailing channels
(tinh-lpa-clfrl.8.5d).

Adding skill_onehot + skill_params observation terms at the END of
the policy/critic obs groups grows the input dim; resuming a
pendulum-era checkpoint needs the first-layer weight matrices padded
with ZERO columns for the new channels (zero weights = the policy is
initially blind to them — behaviorally identical to the source
checkpoint, then learns to read them).

  .venv/bin/python scripts/pad_checkpoint_obs.py \
      <in.pt> <out.pt> --extra 10 [--critic-extra 10]

Pads every '*.0.weight' matrix under actor/critic (rsl_rl MLP naming)
whose second dim matches the detected obs dim.
"""

import argparse

import torch


def pad_first_layers(sd: dict, extra: int, critic_extra: int) -> list[str]:
    padded = []
    for key, w in sd.items():
        if not key.endswith(".weight") or w.ndim != 2:
            continue
        is_actor = "actor" in key
        is_critic = "critic" in key
        if not (is_actor or is_critic):
            continue
        # First layer only: rsl_rl names MLP layers actor.0, actor.2,
        # ... — the .0 matrix is the one reading the obs vector.
        if not (".0.weight" in key or key.endswith("actor.0.weight")
                or key.endswith("critic.0.weight")):
            continue
        add = extra if is_actor else critic_extra
        if add <= 0:
            continue
        pad = torch.zeros(w.shape[0], add, dtype=w.dtype)
        sd[key] = torch.cat([w, pad], dim=1)
        padded.append(f"{key}: {tuple(w.shape)} -> {tuple(sd[key].shape)}")
    return padded


def main() -> int:
    ap = argparse.ArgumentParser(prog="pad_checkpoint_obs")
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--extra", type=int, required=True,
                    help="new trailing obs channels (actor)")
    ap.add_argument("--critic-extra", type=int, default=None,
                    help="critic group channels (default: same)")
    args = ap.parse_args()
    ce = args.extra if args.critic_extra is None else args.critic_extra

    ckpt = torch.load(args.inp, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    padded = pad_first_layers(sd, args.extra, ce)
    assert padded, "no first-layer matrices found — naming mismatch?"
    for line in padded:
        print("padded", line)
    # The optimizer state no longer matches the grown parameters —
    # drop it (fresh optimizer on resume; standard for surgery).
    ckpt.pop("optimizer_state_dict", None)
    torch.save(ckpt, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
