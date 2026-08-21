"""Benchmarks réels de steerabilité pour l'instrument.

Principe commun :
    1. construire une direction (CAA/RepE) depuis des paires contrastées ;
    2. balayer une amplitude (échelle) de steering ;
    3. générer des continuations sur des prompts neutres ;
    4. mesurer la sortie avec un score lexicographique INDÉPENDANT du modèle ;
    5. rapporter corrélation échelle→score (steerabilité) + exemples.

Benchmarks fournis :
    sentiment  — positive/negative (GPT2 & co)          [met: lexicon_sentiment]
    tone       — formel/familier (GPT2 & co)            [met: tone_formality]
    refus      — renforcer le refus, usage défensif     [met: refusal_score]
                 (à exécuter sur un modèle instruct, e.g. Qwen2.5-0.5B-Instruct)
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Mapping, Sequence

import numpy as np

from . import prompts
from .instrument import Instrument
from .model import set_seed


@dataclass
class BenchmarkResult:
    name: str
    model_name: str
    knob: str
    scales: list[float]
    metrics: list[float]                  # moyenne du score par échelle
    per_prompt: dict[str, list[float]]    # échelle -> scores individuels
    examples: dict[str, list[str]]        # échelle -> exemples générés
    correlation: float                    # pearson(scale, metric)
    slope: float                          # pente de la régression linéaire
    baseline: float                       # métrique à échelle 0
    max_new_tokens: int = 32
    report_path: str | None = None

    def table(self) -> str:
        rows = [f"| scale | score |", "|---:|---:|"]
        for s, m in zip(self.scales, self.metrics):
            rows.append(f"| {s:+.2f} | {m:+.4f} |")
        return "\n".join(rows)


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2:
        return 0.0
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0  # variance nulle => corrélation non définie
    r = np.corrcoef(x, y)[0, 1]
    return float(r) if r == r else 0.0


def _slope(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


class SteeringBenchmark:
    """Balayage d'amplitude d'un knob et mesure de la métrique associée."""

    def __init__(
        self,
        instrument: Instrument,
        knob: str,
        prompts_list: Sequence[str],
        metric_fn: Callable[[str], float],
        name: str = "benchmark",
        max_new_tokens: int = 32,
        report_dir: str = "reports/benchmarks",
        apply_chat: bool = False,
        n_examples: int = 4,
    ) -> None:
        self.instrument = instrument
        self.knob = knob
        self.prompts_list = list(prompts_list)
        self.metric_fn = metric_fn
        self.name = name
        self.max_new_tokens = max_new_tokens
        self.report_dir = report_dir
        self.apply_chat = apply_chat
        self.n_examples = n_examples

    def run(
        self,
        scales: Sequence[float] | None = None,
        seed: int = 0,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> BenchmarkResult:
        if scales is None:
            # directions brutes : amplitude 1 ≈ un contraste complet
            scales = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
        scales = [float(s) for s in scales]
        set_seed(seed)

        metrics: list[float] = []
        per_prompt: dict[str, list[float]] = {}
        examples: dict[str, list[str]] = {}
        for scale in scales:
            outs = self.instrument.play_many(
                self.prompts_list,
                max_new_tokens=self.max_new_tokens,
                knobs={self.knob: scale},
                seed=seed,
                do_sample=do_sample,
                temperature=temperature,
                apply_chat=self.apply_chat,
            )
            scores = [self.metric_fn(o) for o in outs]
            metrics.append(float(np.mean(scores)))
            per_prompt[f"{scale:+.2f}"] = [float(s) for s in scores]
            examples[f"{scale:+.2f}"] = outs[: self.n_examples]

        result = BenchmarkResult(
            name=self.name,
            model_name=self.instrument.model.model_name,
            knob=self.knob,
            scales=scales,
            metrics=metrics,
            per_prompt=per_prompt,
            examples=examples,
            correlation=_pearson(scales, metrics),
            slope=_slope(scales, metrics),
            baseline=metrics[scales.index(0.0)] if 0.0 in scales else metrics[0],
            max_new_tokens=self.max_new_tokens,
        )
        result.report_path = self._write_report(result)
        return result

    # ------------------------------------------------------------- rapports
    def _write_report(self, result: BenchmarkResult) -> str:
        os.makedirs(self.report_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        # slug sûr : remplace tout caractère non [a-zA-Z0-9._-] (chemins locaux,
        # séparateurs Windows/UNIX, ':' , espaces) par '-'.
        model_slug = re.sub(r"[^\w.\-]+", "-", result.model_name)
        path = os.path.join(self.report_dir, f"{result.name}_{model_slug}_{ts}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "benchmark": result.name,
                    "model": result.model_name,
                    "knob": result.knob,
                    "max_new_tokens": result.max_new_tokens,
                    "scales": result.scales,
                    "metrics": result.metrics,
                    "per_prompt": result.per_prompt,
                    "examples": result.examples,
                    "correlation": result.correlation,
                    "slope": result.slope,
                    "baseline": result.baseline,
                },
                fh,
                indent=2,
                ensure_ascii=False,
            )
        return path


# ------------------------------------------------------- benchmarks concrets
def sentiment_benchmark(instrument: Instrument, **kw) -> SteeringBenchmark:
    return SteeringBenchmark(
        instrument,
        knob="positive",
        prompts_list=prompts.SENTIMENT_EVAL_PROMPTS,
        metric_fn=prompts.lexicon_sentiment,
        name="sentiment",
        **kw,
    )


def tone_benchmark(instrument: Instrument, **kw) -> SteeringBenchmark:
    kw.setdefault("max_new_tokens", 24)
    return SteeringBenchmark(
        instrument,
        knob="formal",
        prompts_list=prompts.TONE_EVAL_PROMPTS,
        metric_fn=prompts.tone_formality,
        name="tone",
        **kw,
    )


def refusal_benchmark(instrument: Instrument, **kw) -> SteeringBenchmark:
    kw.setdefault("max_new_tokens", 48)
    kw.setdefault("apply_chat", True)
    return SteeringBenchmark(
        instrument,
        knob="refusal",
        prompts_list=prompts.REFUSAL_EVAL_PROMPTS,
        metric_fn=prompts.refusal_score,
        name="refusal",
        **kw,
    )


# ------------------------------------------------------- suite orchestrée
def run_suite(
    model_name: str,
    include: Sequence[str] = ("sentiment", "tone", "refusal"),
    scales: Sequence[float] | None = None,
    layer_ratio: float = 0.6,
    seed: int = 0,
    max_new_tokens: int = 32,
    **model_kwargs,
) -> dict[str, BenchmarkResult]:
    """Construit l'instrument, fabrique les knobs, lance les benchmarks demandés."""
    from .instrument import Instrument

    inst = Instrument(model_name, **model_kwargs)
    inst.default_layer = inst.model.pick_default_layer(layer_ratio)
    layer = inst.default_layer
    results: dict[str, BenchmarkResult] = {}

    if "sentiment" in include:
        inst.make_knob("positive", prompts.SENTIMENT_POSITIVE, prompts.SENTIMENT_NEGATIVE, layer=layer, scale=1.0)
        results["sentiment"] = sentiment_benchmark(inst, max_new_tokens=max_new_tokens).run(scales, seed=seed)
    if "tone" in include:
        inst.make_knob("formal", prompts.TONE_FORMAL, prompts.TONE_CASUAL, layer=layer, scale=1.0)
        results["tone"] = tone_benchmark(inst, max_new_tokens=max_new_tokens).run(scales, seed=seed)
    if "refusal" in include:
        inst.make_knob(
            "refusal",
            prompts.REFUSAL_PROMPTS,
            prompts.COMPLIANCE_PROMPTS,
            layer=layer,
            scale=1.0,
            apply_chat=True,  # direction du refus sur le chat-template (modèle instruct)
        )
        results["refusal"] = refusal_benchmark(inst, max_new_tokens=max_new_tokens).run(scales, seed=seed)

    return results


def print_report(results: Mapping[str, BenchmarkResult]) -> None:
    """Print a readable summary of the benchmark results."""
    # ASCII-safe: the Windows console (cp1252) can't render ▶ / →
    print("\n" + "=" * 70)
    print("STEERABILITY REPORT - steerio (RepE + CAA)")
    print("=" * 70)
    for name, r in results.items():
        print(f"\n- {name}  [model: {r.model_name}, knob: {r.knob}]")
        print(f"  scale->score correlation : {r.correlation:+.3f}")
        print(f"  slope (score/scale)      : {r.slope:+.4f}")
        print(f"  baseline (scale 0)       : {r.baseline:+.4f}")
        print(r.table())
        print(f"  report: {r.report_path}")
