import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pytest

from steerio import Instrument, InstrumentModel

TINY = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture(scope="session")
def tiny_model() -> InstrumentModel:
    return InstrumentModel(TINY)


@pytest.fixture
def tiny_instrument(tiny_model) -> Instrument:
    # function-scoped : chaque test démarre avec un instrument propre
    # (le modèle est partagé, l'état des knobs ne l'est pas)
    return Instrument(model=tiny_model)
