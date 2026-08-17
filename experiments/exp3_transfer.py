"""Expérience 3 — Transfert de directions entre modèles.

Deux designs :
  A. Hidden_size identique (même architecture) : knob « positive » construit
     sur Qwen2.5-1.5B-Instruct → appliqué tel quel à
     DeepSeek-R1-Distill-Qwen-1.5B (1536 dims, 28 couches).
  B. Hidden_size différent : knob « positive » construit sur gpt2 (768) →
     projeté par zero-pad sur Qwen2.5-0.5B-Instruct (896) à la couche
     proportionnelle. Projection naïve documentée.

Pour chaque design on mesure, sur le modèle cible :
  - effet NATIF   : knob construit sur le cible ;
  - effet TRANSFÉRÉ : vecteur du modèle source appliqué sur le cible ;
  - coefficient de transfert = effet_transféré / effet_natif.
L'effet = delta du score de sentiment moyen (baseline → amplitude +1).
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
for p in (_HERE, _PKG):
    if p not in sys.path:
        sys.path.insert(0, p)

from instrument import Instrument, prompts
from instrument.vectors import SteeringVector

import config as C

# Designs : (source, cible, projection)
DESIGNS = [
    {
        "name": "A_matched_hidden",
        "source": "Qwen/Qwen2.5-1.5B-Instruct",
        "target": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "projection": "none",
        "note": "même architecture (Qwen), hidden 1536, 28 couches",
    },
    {
        "name": "B_zero_pad_cross_family",
        "source": "openai-community/gpt2",
        "target": "Qwen/Qwen2.5-0.5B-Instruct",
        "projection": "zero_pad",
        "note": "hidden 768 -> 896, projection naïve zero-pad, couche proportionnelle",
    },
]


def _effect(inst: Instrument, knob: str, eval_prompts, apply_chat: bool = False) -> tuple[float, float]:
    """Delta moyen du score de sentiment entre baseline et amplitude +1.

    apply_chat : mode natif — obligatoire pour les modèles instruct (sinon
    l'effet est annulé, voir découverte Exp. 1).
    """
    # knobs={} = AUCUN knob (baseline vraie) ; knobs=None appliquerait tous les knobs.
    base = inst.play_many(eval_prompts, max_new_tokens=C.MAX_NEW_TOKENS, seed=C.SEED, apply_chat=apply_chat, knobs={})
    steered = inst.play_many(eval_prompts, max_new_tokens=C.MAX_NEW_TOKENS, knobs={knob: 1.0}, seed=C.SEED, apply_chat=apply_chat)
    base_m = sum(prompts.lexicon_sentiment(o) for o in base) / len(base)
    steer_m = sum(prompts.lexicon_sentiment(o) for o in steered) / len(steered)
    return steer_m - base_m, base_m


def run_design(d: dict) -> dict:
    src_id, tgt_id = d["source"], d["target"]
    out: dict = {"design": d["name"], "source": src_id, "target": tgt_id, "note": d["note"]}
    t0 = time.time()

    # --- construire le knob sur la source (mode natif : chat si instruct)
    src = Instrument(C.resolve_model_path(src_id))
    src_layer = src.model.pick_default_layer(C.LAYER_RATIO)
    src_vec = src.build_direction(prompts.SENTIMENT_POSITIVE, prompts.SENTIMENT_NEGATIVE,
                                  layer=src_layer, name="positive", apply_chat=True)
    out["source_layers"] = src.model.n_layers
    out["source_hidden"] = src.model.hidden_size
    out["source_layer"] = src_layer
    out["source_build_s"] = round(time.time() - t0, 1)

    # --- cible : chargement
    tgt = Instrument(C.resolve_model_path(tgt_id))
    tgt_layer = tgt.model.pick_default_layer(C.LAYER_RATIO)
    out["target_layers"] = tgt.model.n_layers
    out["target_hidden"] = tgt.model.hidden_size
    out["target_layer_ratio"] = tgt_layer / max(1, tgt.model.n_layers - 1)

    # --- effet natif (knob construit sur le cible, mode natif)
    native = tgt.build_direction(prompts.SENTIMENT_POSITIVE, prompts.SENTIMENT_NEGATIVE,
                                 layer=tgt_layer, name="native", apply_chat=True)
    tgt.add_knob(native)
    native_delta, native_base = _effect(tgt, "native", C.sent_prompts(), apply_chat=True)
    out["native"] = {"delta": native_delta, "baseline": native_base}

    # --- vecteur transféré
    direction = src_vec.direction
    if d["projection"] == "zero_pad":
        if direction.numel() != tgt.model.hidden_size:
            direction = torch.nn.functional.pad(direction, (0, tgt.model.hidden_size - direction.numel()))
    elif d["projection"] != "none":
        raise ValueError(f"projection inconnue: {d['projection']}")
    transferred = SteeringVector(direction.clone(), tgt_layer, 1.0, "transferred")
    # retirer le knob natif pour mesurer uniquement le transféré
    del tgt.knobs["native"]
    tgt.add_knob(transferred)
    trans_delta, trans_base = _effect(tgt, "transferred", C.sent_prompts(), apply_chat=True)
    out["transferred"] = {"delta": trans_delta, "baseline": trans_base}

    # --- coefficient de transfert
    out["transfer_coefficient"] = (trans_delta / native_delta) if abs(native_delta) > 1e-8 else None
    out["wall_s"] = round(time.time() - t0, 1)
    print(f"[{d['name']}] native_delta={native_delta:+.3f} transferred_delta={trans_delta:+.3f} "
          f"coef={out['transfer_coefficient']}")
    return out


def main() -> None:
    os.makedirs(C.REPORT_DIR, exist_ok=True)
    results = [run_design(d) for d in DESIGNS]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(C.REPORT_DIR, f"exp3_transfer_{ts}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print("EXP3 DONE ->", path)


if __name__ == "__main__":
    main()
