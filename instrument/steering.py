"""Injection de steering à l'inférence via hooks forward (contrôle CAA).

Un :class:`SteeringFilter` est accroché à une couche du transformers ;
à chaque forward il ajoute somme(scale * direction) au flux résiduel.
Pendant `model.generate()`, le filtre s'applique au préfill (positions du
prompt) et aux positions générées — le comportement standard des
implémentations CAA.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


@dataclass
class SteeringEntry:
    """Une contribution (direction, échelle) appliquée à une couche."""

    vector: torch.Tensor
    scale: float = 1.0

    def __post_init__(self) -> None:
        self.vector = self.vector.detach().cpu().float().contiguous()
        self.scale = float(self.scale)


class SteeringFilter:
    """Hook forward qui ajoute sum(scale * direction) au flux résiduel.

    Gère les sorties tenseur (GPT2) et tuple (Llama/Qwen : (hidden, cache)).
    Le hook renvoie `None` quand désactivé (sortie inchangée) et quand aucune
    entrée n'est enregistrée.
    """

    def __init__(self, entries: Sequence[SteeringEntry] | None = None) -> None:
        self.entries: list[SteeringEntry] = list(entries or [])
        self.enabled: bool = True

    def add(self, vector: torch.Tensor, scale: float = 1.0) -> "SteeringFilter":
        self.entries.append(SteeringEntry(vector, scale))
        return self

    def clear(self) -> None:
        self.entries.clear()

    def __len__(self) -> int:
        return len(self.entries)

    def __call__(self, module, args, output):
        if not self.enabled or not self.entries:
            return None
        is_tuple = isinstance(output, (tuple, list))
        h = output[0] if is_tuple else output
        if h is None:
            return None
        device, dtype = h.device, h.dtype
        for entry in self.entries:
            v = entry.vector.to(device=device, dtype=dtype)
            h = h + v * entry.scale
        if is_tuple:
            return (h,) + tuple(output[1:])
        return h


def make_steering(
    entries: Mapping[int, Sequence[SteeringEntry]],
) -> dict[int, SteeringFilter]:
    """Construit un dict {couche: SteeringFilter} depuis un mapping couche->entrées."""
    return {
        layer: SteeringFilter(entries)
        for layer, entries in entries.items()
    }


def attach_steering(model, layers, steering: Mapping[int, SteeringFilter]):
    """Attache les filtres aux couches ; retourne les handles de hook.

    Usage :
        handles = attach_steering(model, model.layers, steering)
        try: ... generation ...
        finally: detach_steering(handles)
    """
    handles = []
    for layer, filt in steering.items():
        handles.append(layers[layer].register_forward_hook(filt))
    return handles


def detach_steering(handles) -> None:
    for handle in handles:
        handle.remove()
