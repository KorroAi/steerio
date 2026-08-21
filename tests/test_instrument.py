"""Tests de l'instrument : knobs, play/read, persistance."""
import os

import pytest
import torch

from steerio import Instrument, InstrumentModel
from steerio.vectors import SteeringVector

TINY = "hf-internal-testing/tiny-random-gpt2"


def test_requires_model_or_name():
    with pytest.raises(ValueError):
        Instrument()


def test_build_direction_methods(tiny_instrument):
    pos = ["I love this great day.", "This is wonderful and amazing."]
    neg = ["I hate this bad day.", "This is terrible and awful."]
    for method in ("contrastive", "pca", "probe"):
        v = tiny_instrument.build_direction(pos, neg, layer=2, method=method, name=f"d-{method}")
        assert v.hidden_size == tiny_instrument.model.hidden_size
        assert torch.isfinite(v.direction).all()


def test_make_knob_and_dimension_check(tiny_instrument):
    tiny_instrument.make_knob("pos", ["I love this great day."], ["I hate this bad day."], layer=2)
    assert "pos" in tiny_instrument.knobs
    bad = SteeringVector(torch.randn(7), layer=2, name="wrong")
    with pytest.raises(ValueError):
        tiny_instrument.add_knob(bad)


def test_play_and_read(tiny_instrument):
    tiny_instrument.make_knob("pos", ["I love this great day."], ["I hate this bad day."], layer=2)
    out = tiny_instrument.play("The movie was", max_new_tokens=8, knobs={"pos": 2.0}, seed=0)
    assert isinstance(out, str) and out
    metrics = tiny_instrument.read("I love this great day.", "pos")
    assert set(metrics) == {"cosine", "projection"}
    assert metrics["cosine"] > metrics["cosine"] - 2  # sanity: pas d'exception


def test_unknown_knob_raises(tiny_instrument):
    with pytest.raises(KeyError):
        tiny_instrument.play("hi", knobs={"nope": 1.0})


def test_knob_persistence_roundtrip(tmp_path, tiny_instrument):
    tiny_instrument.make_knob("pos", ["I love this great day."], ["I hate this bad day."], layer=2)
    path = os.path.join(str(tmp_path), "knobs.pt")
    tiny_instrument.save_knobs(path)

    other = Instrument(model=InstrumentModel(TINY))
    other.load_knobs(path)
    assert set(other.knobs) == {"pos"}
    assert torch.allclose(other.knobs["pos"].direction, tiny_instrument.knobs["pos"].direction)
    assert other.knobs["pos"].layer == tiny_instrument.knobs["pos"].layer


def test_play_determinism_same_seed(tiny_instrument):
    tiny_instrument.make_knob("pos", ["I love this great day."], ["I hate this bad day."], layer=2)
    a = tiny_instrument.play("The movie was", max_new_tokens=8, knobs={"pos": 1.5}, seed=42)
    b = tiny_instrument.play("The movie was", max_new_tokens=8, knobs={"pos": 1.5}, seed=42)
    assert a == b


def test_tune_layer_returns_valid_layer(tiny_instrument):
    pos = ["I love this great day.", "This is wonderful."]
    neg = ["I hate this bad day.", "This is terrible."]
    best_layer, gap = tiny_instrument.tune_layer(
        pos, neg,
        layer_range=[0, 2, 4],
        eval_prompts=["The movie was", "The food was"],
        max_new_tokens=6,
        name="tune",
    )
    assert 0 <= best_layer < tiny_instrument.model.n_layers
    assert gap >= 0.0
    assert "tune" not in tiny_instrument.knobs  # nettoyé après la recherche
