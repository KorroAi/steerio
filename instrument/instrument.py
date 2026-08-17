"""L'instrument — l'interface « jouable » d'un LLM par représentations internes.

Métaphore musicale :
    - chaque :class:`Instrument` est un instrument (un modèle).
    - chaque « knob » est une direction de steering (un concept) sur une couche.
    - « jouer » = générer en tournant les knobs (amplitudes), en solo ou en
      polyphonie (plusieurs knobs simultanément).
    - « lire » (read) = mesurer la présence d'un concept dans une activation
      (lecture RepE).
"""
from __future__ import annotations

from typing import Mapping, Sequence

import torch

from .model import InstrumentModel
from .steering import SteeringEntry, SteeringFilter, attach_steering, detach_steering
from .vectors import SteeringVector, contrastive_direction, mean_difference, pca_direction, probe_direction


class Instrument:
    """Un LLM jouable : knobs (directions) + play() + read()."""

    def __init__(self, model_name: str | None = None, model: InstrumentModel | None = None, **model_kwargs) -> None:
        if model is None and model_name is None:
            raise ValueError("Il faut un model_name ou un InstrumentModel")
        self.model = model if model is not None else InstrumentModel(model_name, **model_kwargs)
        self.knobs: dict[str, SteeringVector] = {}
        self.default_layer = self.model.pick_default_layer()

    # ------------------------------------------------------------- directions
    def build_direction(
        self,
        positive: Sequence[str],
        negative: Sequence[str],
        layer: int | None = None,
        method: str = "contrastive",
        scale: float = 1.0,
        name: str = "concept",
        normalize_direction: bool = False,
        apply_chat: bool = False,
    ) -> SteeringVector:
        """Construit une direction depuis des exemples contrastés (texte).

        method : "contrastive" (CAA), "pca" (RepE non supervisé), "probe" (sonde).
        normalize_direction : False (défaut) = contraste brut, amplitude 1 ≈ un
            contraste complet ; True = direction unitaire (amplitudes à calibrer).
        apply_chat : formate les prompts en tour utilisateur (modèles instruct).
        """
        layer = self.default_layer if layer is None else layer
        pos_acts = self.model.extract_activations(list(positive), layer=layer, apply_chat=apply_chat)
        neg_acts = self.model.extract_activations(list(negative), layer=layer, apply_chat=apply_chat)
        if method == "contrastive":
            direction = contrastive_direction(pos_acts, neg_acts, normalize_direction)
        elif method == "pca":
            direction = pca_direction(torch.cat([pos_acts, neg_acts]))
        elif method == "probe":
            direction = probe_direction(pos_acts, neg_acts)
        else:
            raise ValueError(f"method inconnue : {method!r}")
        return SteeringVector(direction, layer, scale, name)

    def add_knob(self, vector: SteeringVector) -> "Instrument":
        """Enregistre un knob (une direction) sur l'instrument."""
        if vector.hidden_size != self.model.hidden_size:
            raise ValueError(
                f"dimension de la direction ({vector.hidden_size}) != hidden_size "
                f"du modèle ({self.model.hidden_size})"
            )
        self.knobs[vector.name] = vector
        return self

    def make_knob(
        self,
        name: str,
        positive: Sequence[str],
        negative: Sequence[str],
        layer: int | None = None,
        method: str = "contrastive",
        scale: float = 1.0,
        apply_chat: bool = False,
    ) -> SteeringVector:
        """Raccourci : build_direction + add_knob en une étape."""
        vector = self.build_direction(
            positive, negative, layer=layer, method=method, scale=scale, name=name, apply_chat=apply_chat
        )
        self.add_knob(vector)
        return vector

    # ------------------------------------------------------- composition knobs
    def _compose_filters(self, amplitudes: Mapping[str, float] | None = None) -> dict[int, SteeringFilter]:
        """Groupe les knobs par couche ; échelle effective = amplitude * knob.scale.

        amplitudes : {nom_du_knob: amplitude}. Défaut : tous les knobs à 1.0
        (donc échelle = knob.scale).
        """
        if amplitudes is None:
            amplitudes = {name: 1.0 for name in self.knobs}
        by_layer: dict[int, list[SteeringEntry]] = {}
        for name, amp in amplitudes.items():
            if name not in self.knobs:
                raise KeyError(f"knob inconnu : {name!r} (disponibles : {sorted(self.knobs)})")
            knob = self.knobs[name]
            effective = float(amp) * knob.scale
            by_layer.setdefault(knob.layer, []).append(SteeringEntry(knob.direction, effective))
        return {layer: SteeringFilter(entries) for layer, entries in by_layer.items()}

    # ------------------------------------------------------------- play / read
    def play(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        knobs: Mapping[str, float] | None = None,
        **gen_kwargs,
    ) -> str:
        """Joue une note : génère une continuation en tournant les knobs."""
        steering = self._compose_filters(knobs)
        return self.model.generate(prompt, max_new_tokens=max_new_tokens, steering=steering, **gen_kwargs)

    def play_many(
        self,
        prompts: Sequence[str],
        max_new_tokens: int = 32,
        knobs: Mapping[str, float] | None = None,
        **gen_kwargs,
    ) -> list[str]:
        """Version multi-prompts de :meth:`play`."""
        steering = self._compose_filters(knobs)
        return self.model.generate(prompts, max_new_tokens=max_new_tokens, steering=steering, **gen_kwargs)

    def read(self, text: str, name: str | None = None, direction: torch.Tensor | None = None, layer: int | None = None) -> dict:
        """Lecture RepE : présence du concept dans l'activation de `text`."""
        if direction is None:
            if name is None or name not in self.knobs:
                raise ValueError("fournir `name` d'un knob ou une `direction`")
            direction = self.knobs[name].direction
            layer = layer if layer is not None else self.knobs[name].layer
        return self.model.read(text, direction, layer)

    # ------------------------------------------------------ réglage de couche
    def tune_layer(
        self,
        positive: Sequence[str],
        negative: Sequence[str],
        layer_range: Sequence[int] | None = None,
        eval_prompts: Sequence[str] | None = None,
        metric_fn=None,
        method: str = "contrastive",
        max_new_tokens: int = 16,
        seed: int = 0,
        name: str = "concept",
        apply_chat: bool = False,
    ) -> tuple[int, float]:
        """Cherche la couche où le steering est le plus efficace (mécanique CAA).

        Pour chaque couche candidate : on construit la direction par contraste,
        on génère des continuations sur des prompts neutres à amplitude ±A, et
        on mesure |score(+A) − score(−A)| — l'écart produit par le knob.
        Retourne (meilleure_couche, écart_max).
        """
        import numpy as np

        from . import prompts as _prompts

        if eval_prompts is None:
            eval_prompts = ["The movie was", "The food was", "The meeting was"]
        if metric_fn is None:
            metric_fn = _prompts.lexicon_sentiment
        if layer_range is None:
            step = max(1, self.model.n_layers // 5)
            layer_range = range(0, self.model.n_layers, step)
        amplitudes = (1.0, -1.0)  # un contraste complet, dans chaque sens

        best_layer, best_gap = -1, -float("inf")
        for layer in layer_range:
            vector = self.build_direction(
                positive, negative, layer=layer, method=method, name=name, apply_chat=apply_chat
            )
            self.add_knob(vector)
            scores = []
            for amp in amplitudes:
                outs = self.play_many(list(eval_prompts), max_new_tokens=max_new_tokens, knobs={name: amp}, seed=seed)
                scores.append(float(np.mean([metric_fn(o) for o in outs])))
            gap = abs(scores[0] - scores[1])
            print(f"  couche {layer:>3} : écart {gap:+.4f}")
            if gap > best_gap:
                best_gap, best_layer = gap, layer
            del self.knobs[name]
        print(f"→ meilleure couche : {best_layer} (écart {best_gap:+.4f})")
        return best_layer, best_gap

    # ----------------------------------------------------------- persistance
    def save_knobs(self, path: str) -> None:
        """Sauvegarde les knobs (torch.save)."""
        payload = {
            "knobs": {name: {"layer": v.layer, "scale": v.scale, "direction": v.direction} for name, v in self.knobs.items()},
            "model_name": self.model.model_name,
            "hidden_size": self.model.hidden_size,
        }
        torch.save(payload, path)

    def load_knobs(self, path: str) -> "Instrument":
        """Recharge des knobs sauvegardés (même modèle attendu)."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["hidden_size"] != self.model.hidden_size:
            raise ValueError("knobs incompatibles avec ce modèle (hidden_size différent)")
        self.knobs = {
            name: SteeringVector(torch.as_tensor(data["direction"]), data["layer"], data["scale"], name)
            for name, data in payload["knobs"].items()
        }
        return self
