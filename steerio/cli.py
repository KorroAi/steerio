"""steerio command-line interface.

Examples:
    python -m steerio play --model openai-community/gpt2 --prompt "The movie was" --scale 1 \
        --positive "The weather is beautiful.; I love this!" --negative "The weather is gloomy.; I hate this!"
    python -m steerio build --model openai-community/gpt2 --name positive --layer 7 \
        --positive "The weather is beautiful." --negative "The weather is gloomy." --save knobs.pt
    python -m steerio benchmark --suite sentiment,tone --model openai-community/gpt2
    python -m steerio benchmark --suite refusal --model Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations

import argparse
import sys

from . import prompts
from .benchmarks import run_suite, print_report
from .instrument import Instrument


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="openai-community/gpt2", help="HF model name")
    parser.add_argument("--layer", type=int, default=None, help="steering layer (default: 60%% of depth)")
    parser.add_argument("--ratio", type=float, default=0.6, help="default layer = ratio * n_layers")


def cmd_play(args: argparse.Namespace) -> None:
    inst = Instrument(args.model)
    layer = args.layer if args.layer is not None else inst.default_layer
    texts_pos = [t.strip() for t in args.positive.split(";") if t.strip()]
    texts_neg = [t.strip() for t in args.negative.split(";") if t.strip()]
    if not texts_pos or not texts_neg:
        raise SystemExit("Need --positive and --negative (separated by ';') to build the knob.")
    inst.make_knob("knob", texts_pos, texts_neg, layer=layer, method=args.method)
    out = inst.play(args.prompt, max_new_tokens=args.tokens, knobs={"knob": args.scale}, seed=args.seed)
    print(f"--- prompt ---\n{args.prompt}\n--- continuation (scale={args.scale:+.2f}) ---\n{out}")


def cmd_build(args: argparse.Namespace) -> None:
    inst = Instrument(args.model)
    layer = args.layer if args.layer is not None else inst.default_layer
    texts_pos = [t.strip() for t in args.positive.split(";") if t.strip()]
    texts_neg = [t.strip() for t in args.negative.split(";") if t.strip()]
    if not texts_pos or not texts_neg:
        raise SystemExit("Need --positive and --negative (separated by ';') to build the direction.")
    vector = inst.build_direction(texts_pos, texts_neg, layer=layer, method=args.method, scale=args.scale, name=args.name)
    if args.save:
        inst.add_knob(vector)
        inst.save_knobs(args.save)
        print(f"knob '{args.name}' saved to {args.save}")
    else:
        print(f"direction '{args.name}' (layer {layer}, dim {vector.hidden_size}, norm {vector.direction.norm():.3f})")


def cmd_benchmark(args: argparse.Namespace) -> None:
    include = [s.strip().lower() for s in args.suite.split(",") if s.strip()]
    include = ["refusal" if s == "refus" else s for s in include]
    unknown = set(include) - {"sentiment", "tone", "refusal"}
    if unknown:
        raise SystemExit(f"unknown benchmarks: {sorted(unknown)} (choices: sentiment, tone, refus)")
    results = run_suite(
        args.model,
        include=include,
        scales=[float(s) for s in args.scales.split(",")] if args.scales else None,
        layer_ratio=args.ratio,
        seed=args.seed,
        max_new_tokens=args.tokens,
    )
    print_report(results)


def main(argv: list[str] | None = None) -> None:
    # Windows consoles (cp1252) don't handle UTF-8 by default
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(
        prog="steerio",
        description="steerio: play a LLM with steering vectors (RepE + CAA)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_play = sub.add_parser("play", help="generate a continuation with a knob")
    p_play.add_argument("--prompt", required=True)
    p_play.add_argument("--positive", required=True, help="positive texts, ';'-separated")
    p_play.add_argument("--negative", required=True, help="negative texts, ';'-separated")
    p_play.add_argument("--scale", type=float, default=1.0, help="amplitude (1 ≈ one full contrast)")
    p_play.add_argument("--tokens", type=int, default=32)
    p_play.add_argument("--seed", type=int, default=0)
    p_play.add_argument("--method", default="contrastive", choices=["contrastive", "pca", "probe"])
    _add_model_args(p_play)
    p_play.set_defaults(func=cmd_play)

    p_build = sub.add_parser("build", help="build a steering direction")
    p_build.add_argument("--name", default="concept")
    p_build.add_argument("--positive", required=True, help="positive texts, ';'-separated")
    p_build.add_argument("--negative", required=True, help="negative texts, ';'-separated")
    p_build.add_argument("--scale", type=float, default=1.0)
    p_build.add_argument("--method", default="contrastive", choices=["contrastive", "pca", "probe"])
    p_build.add_argument("--save", default=None, help=".pt path to save the knob")
    _add_model_args(p_build)
    p_build.set_defaults(func=cmd_build)

    p_bench = sub.add_parser("benchmark", help="run steerability benchmarks")
    p_bench.add_argument("--suite", default="sentiment,tone", help="sentiment,tone,refus (or a combo)")
    p_bench.add_argument("--scales", default=None, help="comma-separated scales (e.g. -3,-1.5,0,1.5,3)")
    p_bench.add_argument("--tokens", type=int, default=32)
    p_bench.add_argument("--seed", type=int, default=0)
    _add_model_args(p_bench)
    p_bench.set_defaults(func=cmd_benchmark)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
