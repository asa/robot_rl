"""Fold a skill checkpoint's trailing obs channels into the bias
(inverse of pad_checkpoint_obs.py, tinh-lpa-clfrl.7.7 v2 close-out).

The graphturn/skill checkpoints read obs = [base(74) | skill_onehot(3)
| skill_params(2)].  The walk/yaw gate harnesses build the plain
walking env (74 obs).  In pure-locomotion mode the skill channels are
CONSTANT: onehot = [1, 0, 0] (slot 0 = locomotion), params = [0, 0] —
so the exact 74-obs equivalent policy is

    W' = W[:, :74],   b' = b + W[:, 74] * 1.0

applied to the first actor/critic layers.  No approximation.

  .venv/bin/python scripts/fold_checkpoint_obs.py \
      <in.pt> <out.pt> --extra 5 [--critic-extra 5] [--hot 0]
"""

import argparse

import torch


def fold_first_layers(sd: dict, extra: int, critic_extra: int,
                      hot: int) -> list[str]:
    folded = []
    for key, w in list(sd.items()):
        if not key.endswith(".0.weight") or w.ndim != 2:
            continue
        is_actor = "actor" in key
        is_critic = "critic" in key
        if not (is_actor or is_critic):
            continue
        cut = extra if is_actor else critic_extra
        if cut <= 0:
            continue
        base = w.shape[1] - cut
        bkey = key[: -len("weight")] + "bias"
        sd[bkey] = sd[bkey] + w[:, base + hot]
        sd[key] = w[:, :base].clone()
        folded.append(f"{key}: {tuple(w.shape)} -> {tuple(sd[key].shape)}"
                      f" (col {base + hot} -> bias)")
    return folded


def main() -> int:
    ap = argparse.ArgumentParser(prog="fold_checkpoint_obs")
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--extra", type=int, required=True,
                    help="trailing skill obs channels (actor)")
    ap.add_argument("--critic-extra", type=int, default=None)
    ap.add_argument("--hot", type=int, default=0,
                    help="index within the trailing block whose onehot"
                         " is 1.0 (default 0 = locomotion)")
    args = ap.parse_args()
    ce = args.extra if args.critic_extra is None else args.critic_extra

    ckpt = torch.load(args.inp, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    folded = fold_first_layers(sd, args.extra, ce, args.hot)
    assert folded, "no first-layer matrices found — naming mismatch?"
    for line in folded:
        print("folded", line)
    ckpt.pop("optimizer_state_dict", None)
    torch.save(ckpt, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
