"""steerio — play a language model like a musical instrument.

Control a LLM through its internal representations / steering vectors
(RepE: Representation Engineering — Zou et al. 2023 ;
 CAA: Contrastive Activation Addition — Arditi et al. 2024).

Public API:
    Instrument        — the instrument: knobs (steering directions), play(), read()
    InstrumentModel   — model wrapper: activations, generation, RepE reading
    SteeringVector    — a direction + layer + scale
    contrastive_direction, pca_direction, probe_direction — direction builders
    benchmarks        — real steerability benchmarks
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
