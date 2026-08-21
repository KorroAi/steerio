"""Expérience 1 — Contrôle par steering vs contrôle par instruction (prompt).

Même modèle, mêmes prompts d'évaluation, même scoreur externe :
  - baseline        : prompt nu, aucun contrôle ;
  - steer_+1/-1     : knob CAA à amplitude +1 / -1 ;
  - prompt_pos/neg  : instruction textuelle (2 formulations) ajoutée au prompt,
                      SANS steering.

On rapporte, par dimension (sentiment, ton) et par modèle :
  - le score moyen de chaque condition ;
  - l'écart à la baseline (effect size) de chaque contrôle ;
  - les scores par prompt (appariés) pour un test de signe.
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

from steerio import Instrument, prompts

import config as C


def _mean(xs):
    return float(np.mean(xs)) if xs else 0.0


def _std(xs):
    return float(np.std(xs)) if len(xs) > 1 else 0.0


def _paired_sign_test(control, baseline):
    """Test de signe apparié : fraction de prompts où control > baseline."""
    diffs = [c - b for c, b in zip(control, baseline) if c != b]
    if not diffs:
        return {"n": 0, "p_pos": 0.0}
    return {"n": len(diffs), "p_pos": sum(1 for d in diffs if d > 0) / len(diffs)}


def run_model(model_id: str) -> dict:
    t0 = time.time()
    inst = Instrument(C.resolve_model_path(model_id))
    layer = inst.model.pick_default_layer(C.LAYER_RATIO)
    inst.make_knob("positive", prompts.SENTIMENT_POSITIVE, prompts.SENTIMENT_NEGATIVE, layer=layer, apply_chat=True)
    inst.make_knob("formal", prompts.TONE_FORMAL, prompts.TONE_CASUAL, layer=layer, apply_chat=True)
    print(f"[{model_id}] loaded ({round(time.time()-t0,1)}s), layers={inst.model.n_layers}, hidden={inst.model.hidden_size}")

    result: dict = {"model": model_id, "family": C.FAMILY.get(model_id, "?"), "dimensions": {}}

    for dim, knob, plist, metric in [
        ("sentiment", "positive", C.sent_prompts(), prompts.lexicon_sentiment),
        ("tone", "formal", C.tone_prompts(), prompts.tone_formality),
    ]:
        prompts_list = list(plist)
        # --- conditions (mode natif : chat-template si instruct, base sinon)
        # knobs={} = AUCUN knob (baseline vraie) : knobs=None applique au contraire
        # tous les knobs à amplitude 1.0 (sémantique « jouer tout l'instrument »).
        base = inst.play_many(prompts_list, max_new_tokens=C.MAX_NEW_TOKENS, seed=C.SEED, apply_chat=True, knobs={})
        s_plus = inst.play_many(prompts_list, max_new_tokens=C.MAX_NEW_TOKENS, knobs={knob: +1.0}, seed=C.SEED, apply_chat=True)
        s_minus = inst.play_many(prompts_list, max_new_tokens=C.MAX_NEW_TOKENS, knobs={knob: -1.0}, seed=C.SEED, apply_chat=True)
        pos_instrs = C.POS_INSTRUCTIONS if dim == "sentiment" else [
            "Continue in a formal tone.",
            "Write in an official, professional register.",
        ]
        neg_instrs = C.NEG_INSTRUCTIONS if dim == "sentiment" else [
            "Continue in a casual, friendly tone.",
            "Write in a relaxed, informal register.",
        ]
        p_pos = [inst.play_many([f"{p} {i}" for p in prompts_list], max_new_tokens=C.MAX_NEW_TOKENS, seed=C.SEED, apply_chat=True, knobs={})
                 for i in pos_instrs]
        p_neg = [inst.play_many([f"{p} {i}" for p in prompts_list], max_new_tokens=C.MAX_NEW_TOKENS, seed=C.SEED, apply_chat=True, knobs={})
                 for i in neg_instrs]

        # --- scores par prompt (appariés)
        def scores(outs):
            return [float(metric(o)) for o in outs]

        base_sc = scores(base)
        conds = {
            "baseline": base_sc,
            "steer_+1": scores(s_plus),
            "steer_-1": scores(s_minus),
        }
        for k, i in enumerate(pos_instrs):
            conds[f"prompt_pos_{k+1}"] = scores(p_pos[k])
        for k, i in enumerate(neg_instrs):
            conds[f"prompt_neg_{k+1}"] = scores(p_neg[k])

        # --- agrégats
        agg = {}
        for name, sc in conds.items():
            agg[name] = {
                "mean": _mean(sc),
                "std": _std(sc),
                "effect_vs_baseline": _mean(sc) - _mean(base_sc),
                "sign_test_vs_baseline": _paired_sign_test(sc, base_sc),
            }
        # steering control power = écart max |steer +/-| ; prompting = écart max |prompt pos/neg|
        steer_power = max(abs(agg["steer_+1"]["mean"] - agg["baseline"]["mean"]),
                          abs(agg["steer_-1"]["mean"] - agg["baseline"]["mean"]))
        prompt_means = {n: agg[n]["mean"] for n in agg if n.startswith("prompt")}
        prompt_power = max(abs(m - agg["baseline"]["mean"]) for m in prompt_means.values())

        result["dimensions"][dim] = {
            "conditions": agg,
            "per_prompt": conds,
            "control_power_steer": steer_power,
            "control_power_prompt": prompt_power,
            "best_prompt_formulation": max(prompt_means, key=lambda n: abs(prompt_means[n] - agg["baseline"]["mean"])),
        }
        print(f"[{model_id}] {dim}: steer_power={steer_power:+.3f} prompt_power={prompt_power:+.3f}")

    result["wall_s"] = round(time.time() - t0, 1)
    return result


def main(models: list[str] | None = None) -> None:
    models = models or ["openai-community/gpt2", "Qwen/Qwen2.5-0.5B-Instruct"]
    os.makedirs(C.REPORT_DIR, exist_ok=True)
    all_results = {}
    for m in models:
        r = run_model(m)
        all_results[m] = r
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(C.REPORT_DIR, f"exp1_steering_vs_prompting_{ts}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, ensure_ascii=False)
    print("EXP1 DONE ->", path)


if __name__ == "__main__":
    main()
