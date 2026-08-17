"""Math des directions de steering — le cœur RepE/CAA.

Toutes les opérations sont pures (tenseurs torch, pas de dépendance modèle).
Une « direction » est un vecteur dans le flux résiduel d'une couche du
transformers ; l'ajouter (multiplicativement pondéré) à l'inférence déplace
la distribution de sortie le long d'un concept.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


@dataclass
class SteeringVector:
    """Direction de steering : vecteur + couche + échelle + nom (knob).

    direction : tenseur de forme [hidden_size] — le vecteur qui sera AJOUTÉ au
        flux résiduel. Par défaut c'est le contraste brut (moyenne(pos) −
        moyenne(neg)) : son amplitude naturelle fait que `scale` 1 ≈ un
        contraste complet. `normalize_direction=True` est disponible pour une
        direction unitaire (amplitudes à calibrer alors plus grandes).
    layer     : index de la couche transformateur où l'ajouter
    scale     : amplitude par défaut (multiplie la direction)
    name      : libellé optionnel — le nom du « bouton » sur l'instrument
    """

    direction: torch.Tensor
    layer: int
    scale: float = 1.0
    name: str | None = None

    def __post_init__(self) -> None:
        self.direction = self.direction.detach().cpu().float().contiguous()

    @property
    def hidden_size(self) -> int:
        return self.direction.shape[0]

    def to(self, device: torch.device, dtype: torch.dtype) -> "SteeringVector":
        # object.__new__ pour court-circuiter __post_init__ (qui force float32/CPU)
        new = object.__new__(SteeringVector)
        new.direction = self.direction.to(device=device, dtype=dtype).contiguous()
        new.layer = self.layer
        new.scale = self.scale
        new.name = self.name
        return new


def normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalise un vecteur à la norme unité."""
    return v / (v.norm() + eps)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Similarité cosinus entre deux vecteurs/activations."""
    return F.cosine_similarity(a, b, dim=dim)


def mean_difference(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
    normalize_direction: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direction contrastive (CAA) : moyenne(pos) - moyenne(neg).

    pos_acts : tenseur [n_pos, hidden]
    neg_acts : tenseur [n_neg, hidden]
    normalize_direction : si True, ramène la direction à la norme unité.
        Par défaut False : la direction est le **contraste brut** (sa norme est
        la distance entre les centroïdes), ce qui rend l'amplitude 1
        interprétable : « un contraste complet ».
    Retourne (direction, pos_mean, neg_mean).
    """
    pos_mean = pos_acts.mean(dim=0)
    neg_mean = neg_acts.mean(dim=0)
    direction = pos_mean - neg_mean
    if normalize_direction:
        direction = normalize(direction)
    return direction, pos_mean, neg_mean


def contrastive_direction(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
    normalize_direction: bool = False,
) -> torch.Tensor:
    """Alias pratique de :func:`mean_difference` (ne renvoie que la direction)."""
    direction, _, _ = mean_difference(pos_acts, neg_acts, normalize_direction)
    return direction


def pca_direction(acts: torch.Tensor, component: int = 0) -> torch.Tensor:
    """Direction RepE par composante principale (SVD) sur des activations centrées.

    Pour un ensemble d'activations d'exemples du concept, l'axe de variance
    maximale capture un « facteur » du concept (lecture non supervisée).
    """
    acts = acts - acts.mean(dim=0, keepdim=True)
    _, _, vt = torch.linalg.svd(acts, full_matrices=False)
    return normalize(vt[component])


def probe_direction(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
    steps: int = 300,
    lr: float = 5e-2,
    seed: int = 0,
) -> torch.Tensor:
    """Direction par sonde linéaire (régression logistique, RepE « read »).

    Entraîne un classifieur linéaire pos vs neg sur les activations ;
    la normale du plan de séparation est la direction du concept.
    """
    torch.manual_seed(seed)
    X = torch.cat([pos_acts, neg_acts]).float()
    y = torch.cat([torch.ones(len(pos_acts)), torch.zeros(len(neg_acts))])
    # standardisation (numériquement stable)
    mu, sd = X.mean(dim=0), X.std(dim=0).clamp_min(1e-6)
    Xs = (X - mu) / sd
    w = torch.zeros(Xs.shape[1], dtype=torch.float32, requires_grad=True)
    b = torch.zeros((), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.SGD([w, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        logits = Xs @ w + b
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        opt.step()
    return normalize(w.detach())


def cosine_matrix(acts: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Matrice de similarité cosinus [n_acts, n_dirs]."""
    acts = acts / (acts.norm(dim=-1, keepdim=True) + 1e-8)
    dirs = directions / (directions.norm(dim=-1, keepdim=True) + 1e-8)
    return acts @ dirs.t()


def mean_center(vectors: Iterable[torch.Tensor]) -> torch.Tensor:
    """Centroïde moyen d'une liste de vecteurs."""
    stack = torch.stack([v.float() for v in vectors])
    return stack.mean(dim=0)


def compose_directions(
    vectors: Sequence[SteeringVector],
    scales: Sequence[float] | None = None,
) -> torch.Tensor:
    """Somme pondérée de directions (pour une polyphonie de knobs).

    scales : amplitudes effectives, une par vecteur (défaut : vector.scale).
    """
    if scales is None:
        scales = [v.scale for v in vectors]
    if len(vectors) != len(scales):
        raise ValueError("Il faut une échelle par vecteur")
    total = None
    for v, s in zip(vectors, scales):
        term = v.direction.float() * float(s)
        total = term if total is None else total + term
    return normalize(total)
