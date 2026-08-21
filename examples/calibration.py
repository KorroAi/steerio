"""Calibration empirique : pourquoi les directions sont des contrastes bruts.

Mesure la norme résiduelle d'une couche, la norme de la direction contrastive,
et l'effet du steering sur des prompts neutres à différentes amplitudes.

Lance :  python examples/calibration.py [--model openai-community/gpt2]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from steerio import Instrument, prompts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration des amplitudes de steering")
    parser.add_argument("--model", default="openai-community/gpt2")
    args = parser.parse_args()

    inst = Instrument(args.model)
    layer = inst.default_layer
    print(f"couche par défaut : {layer}, n_layers {inst.model.n_layers}")

    # norme résiduelle moyenne à cette couche
    pos_acts = inst.model.extract_activations(prompts.SENTIMENT_POSITIVE, layer=layer)
    neg_acts = inst.model.extract_activations(prompts.SENTIMENT_NEGATIVE, layer=layer)
    all_acts = torch.cat([pos_acts, neg_acts])
    print(f"norme moyenne des activations : {all_acts.norm(dim=1).mean().item():.3f}")

    raw = pos_acts.mean(0) - neg_acts.mean(0)
    print(f"norme direction brute (pos-mean diff) : {raw.norm().item():.3f}")
    print(f"ratio direction brute / norme activations : {(raw.norm() / all_acts.norm(dim=1).mean()).item():.4f}")
    print("=> le contraste brut a une amplitude naturelle (~20% du résiduel) ;")
    print("   une direction unitaire x 3 serait 20x trop faible.")

    # essai d'amplitudes avec direction BRUTE (le défaut de l'instrument)
    def score_out(text: str) -> float:
        return prompts.lexicon_sentiment(text)

    prompts_list = prompts.SENTIMENT_EVAL_PROMPTS[:6]
    inst.make_knob("pos", prompts.SENTIMENT_POSITIVE, prompts.SENTIMENT_NEGATIVE, layer=layer)
    for amp in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
        outs = inst.play_many(prompts_list, max_new_tokens=16, knobs={"pos": amp}, seed=0)
        avg = sum(score_out(o) for o in outs) / len(outs)
        print(f"amp {amp:>4.1f} -> score moyen {avg:+.4f} | ex: {outs[0][:70]!r}")


if __name__ == "__main__":
    main()

