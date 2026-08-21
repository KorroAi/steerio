"""steerio demo — play gpt2 like an instrument.

Run:  python examples/demo.py [--model openai-community/gpt2]

Builds the "positive" and "formal" knobs, then plays:
  - a solo (one knob, positive and negative amplitude);
  - a polyphony (two knobs together);
  - a RepE read (measure a concept inside a text).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from steerio import Instrument, prompts  # noqa: E402


def main() -> None:
    # Windows consoles (cp1252) choke on UTF-8/emoji by default — force UTF-8.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="steerio demo")
    parser.add_argument("--model", default="openai-community/gpt2")
    parser.add_argument("--layer", type=int, default=None)
    args = parser.parse_args()

    inst = Instrument(args.model)
    layer = args.layer if args.layer is not None else inst.default_layer
    print(f"▶ instrument loaded : {args.model} (working layer {layer})")

    inst.make_knob("positive", prompts.SENTIMENT_POSITIVE, prompts.SENTIMENT_NEGATIVE, layer=layer)
    inst.make_knob("formal", prompts.TONE_FORMAL, prompts.TONE_CASUAL, layer=layer)

    print("\n── Solo : knob 'positive', amplitudes -1 → +1 (raw contrast) ──")
    for prompt in ("The movie was", "The food at the restaurant was"):
        print(f"\n· {prompt!r}")
        for amp, label in ((-1.0, "negative"), (0.0, "neutral"), (1.0, "positive")):
            out = inst.play(prompt, max_new_tokens=20, knobs={"positive": amp}, seed=0)
            print(f"  [{label:>8} | amp {amp:+.1f}] {out}")

    print("\n── Polyphony : positive + formal together ──")
    out = inst.play(
        "Dear colleague, I am writing to",
        max_new_tokens=24,
        knobs={"positive": 0.5, "formal": 1.0},
        seed=0,
    )
    print(f"  [positive+formal] {out}")

    print("\n── RepE read (presence of the 'positive' concept) ──")
    for text in (
        "I love this wonderful beautiful day.",
        "I hate this terrible awful day.",
        "The spreadsheet has been updated.",
    ):
        m = inst.read(text, name="positive")
        print(f"  cos={m['cosine']:+.3f} proj={m['projection']:+.3f}  ← {text!r}")


if __name__ == "__main__":
    main()
