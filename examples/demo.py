"""Démo de l'IA-instrument : jouer du gpt2 comme d'un instrument.

Lance :  python examples/demo.py [--model openai-community/gpt2]

Construit les knobs « positive » et « formal », puis joue :
  - un solo (knob seul, amplitude positive et négative) ;
  - une polyphonie (deux knobs ensemble) ;
  - une lecture RepE (mesurer le concept dans un texte).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from instrument import Instrument, prompts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Démo de l'IA-instrument")
    parser.add_argument("--model", default="openai-community/gpt2")
    parser.add_argument("--layer", type=int, default=None)
    args = parser.parse_args()

    inst = Instrument(args.model)
    layer = args.layer if args.layer is not None else inst.default_layer
    print(f"🎹 instrument chargé : {args.model} (couche de travail {layer})")

    inst.make_knob("positive", prompts.SENTIMENT_POSITIVE, prompts.SENTIMENT_NEGATIVE, layer=layer)
    inst.make_knob("formal", prompts.TONE_FORMAL, prompts.TONE_CASUAL, layer=layer)

    print("\n── Solo : knob « positive », amplitudes -1 → +1 (contraste brut) ──")
    for prompt in ("The movie was", "The food at the restaurant was"):
        print(f"\n· {prompt!r}")
        for amp, label in ((-1.0, "négatif"), (0.0, "neutre"), (1.0, "positif")):
            out = inst.play(prompt, max_new_tokens=20, knobs={"positive": amp}, seed=0)
            print(f"  [{label:>8} | amp {amp:+.1f}] {out}")

    print("\n── Polyphonie : positive + formal ensemble ──")
    out = inst.play(
        "Dear colleague, I am writing to",
        max_new_tokens=24,
        knobs={"positive": 0.5, "formal": 1.0},
        seed=0,
    )
    print(f"  [positif+formel] {out}")

    print("\n── Lecture RepE (présence du concept « positive ») ──")
    for text in (
        "I love this wonderful beautiful day.",
        "I hate this terrible awful day.",
        "The spreadsheet has been updated.",
    ):
        m = inst.read(text, name="positive")
        print(f"  cos={m['cosine']:+.3f} proj={m['projection']:+.3f}  ← {text!r}")


if __name__ == "__main__":
    main()
