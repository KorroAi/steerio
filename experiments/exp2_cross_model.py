"""Expérience 2 — Benchmark systématique de steerabilité cross-modèles.

Protocole IDENTIQUE sur 5 modèles (4 familles) :
  - sentiment : knob « positive », 16 prompts neutres, balayage 5 amplitudes ;
  - tone      : knob « formal », 8 prompts neutres, balayage 5 amplitudes ;
  - refus (défensif, chat) : knob « refusal » sur les 4 modèles instruct.

Chaque benchmark produit : corrélation échelle→score, pente, baseline,
score par amplitude, exemples. Rapports JSON par (modèle, benchmark) +
tableau de synthèse écrit dans reports/experiments/.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
for p in (_HERE, _PKG):
    if p not in sys.path:
        sys.path.insert(0, p)

from instrument import Instrument, prompts
from instrument.benchmarks import SteeringBenchmark

import config as C


def run_model(model_id: str, include: tuple = ("sentiment", "tone", "refusal")) -> dict:
    t0 = time.time()
    inst = Instrument(C.resolve_model_path(model_id))
    layer = inst.model.pick_default_layer(C.LAYER_RATIO)
    print(f"[{model_id}] loaded ({round(time.time()-t0,1)}s) layers={inst.model.n_layers} hidden={inst.model.hidden_size}")
    results: dict = {}

    # sentiment/tone : mode natif (chat si instruct, base sinon) pour tous les
    # modèles — gpt2 n'a pas de chat_template, apply_chat y est sans effet.
    if "sentiment" in include:
        inst.make_knob("positive", prompts.SENTIMENT_POSITIVE, prompts.SENTIMENT_NEGATIVE, layer=layer, apply_chat=True)
        b = SteeringBenchmark(inst, "positive", C.sent_prompts(), prompts.lexicon_sentiment,
                              name="sentiment", max_new_tokens=C.MAX_NEW_TOKENS, report_dir=C.REPORT_DIR, apply_chat=True)
        r = b.run(C.SCALES, seed=C.SEED)
        results["sentiment"] = {"correlation": r.correlation, "slope": r.slope,
                                "baseline": r.baseline, "metrics": r.metrics, "scales": r.scales,
                                "report": r.report_path}
    if "tone" in include:
        inst.make_knob("formal", prompts.TONE_FORMAL, prompts.TONE_CASUAL, layer=layer, apply_chat=True)
        b = SteeringBenchmark(inst, "formal", C.tone_prompts(), prompts.tone_formality,
                              name="tone", max_new_tokens=24, report_dir=C.REPORT_DIR, apply_chat=True)
        r = b.run(C.SCALES, seed=C.SEED)
        results["tone"] = {"correlation": r.correlation, "slope": r.slope,
                           "baseline": r.baseline, "metrics": r.metrics, "scales": r.scales,
                           "report": r.report_path}
    if "refusal" in include and model_id in C.INSTRUCT_MODELS:
        inst.make_knob("refusal", prompts.REFUSAL_PROMPTS, prompts.COMPLIANCE_PROMPTS,
                       layer=layer, apply_chat=True)
        b = SteeringBenchmark(inst, "refusal", C.refusal_prompts(), prompts.refusal_score,
                              name="refusal", max_new_tokens=48, report_dir=C.REPORT_DIR, apply_chat=True)
        r = b.run(C.SCALES, seed=C.SEED)
        results["refusal"] = {"correlation": r.correlation, "slope": r.slope,
                              "baseline": r.baseline, "metrics": r.metrics, "scales": r.scales,
                              "report": r.report_path}
    results["wall_s"] = round(time.time() - t0, 1)
    return results


def main(models: list[str] | None = None) -> None:
    models = models or C.MODELS
    os.makedirs(C.REPORT_DIR, exist_ok=True)
    summary: dict = {"models": [], "rows": []}
    for m in models:
        r = run_model(m)
        row = {"model": m, "family": C.FAMILY.get(m, "?"), **r}
        summary["models"].append(row)
        print(f"[{m}] OK {r.get('wall_s')}s")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(C.REPORT_DIR, f"exp2_cross_model_summary_{ts}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print("EXP2 DONE ->", path)


if __name__ == "__main__":
    main()
