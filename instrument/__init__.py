"""L'IA-instrument — jouer d'un LLM comme d'un instrument de musique.

Contrôle par représentations internes / steering vectors
(RepE : Representation Engineering — Zou et al. 2023 ;
 CAA : Contrastive Activation Addition — Arditi et al. 2024).

API publique :
    Instrument        — l'instrument : knobs (directions de steering), play(), read()
    InstrumentModel   — wrapper modèle : activations, génération, lecture RepE
    SteeringVector    — une direction + couche + échelle
    contrastive_direction, pca_direction, probe_direction — constructions de directions
    benchmarks        — benchmarks réels de steerabilité
"""

from .instrument import Instrument
from .model import InstrumentModel, set_seed
from .steering import SteeringFilter, SteeringEntry
from .vectors import (
    SteeringVector,
    normalize,
    cosine_similarity,
    mean_difference,
    contrastive_direction,
    pca_direction,
    probe_direction,
)
from . import prompts
from . import benchmarks

__version__ = "1.0.0"
__all__ = [
    "Instrument",
    "InstrumentModel",
    "SteeringVector",
    "SteeringFilter",
    "SteeringEntry",
    "normalize",
    "cosine_similarity",
    "mean_difference",
    "contrastive_direction",
    "pca_direction",
    "probe_direction",
    "prompts",
    "benchmarks",
    "set_seed",
    "__version__",
]
