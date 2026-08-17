"""Tests du wrapper modèle : découverte des couches, activations, génération, lecture."""
import torch

from instrument.model import discover_layers, set_seed
from instrument import InstrumentModel


def test_discover_layers(tiny_model):
    assert isinstance(tiny_model.layers, torch.nn.ModuleList)
    assert tiny_model.n_layers == len(tiny_model.layers) == 5
    assert tiny_model.hidden_size == 32
    assert tiny_model.vocab_size == 1000


def test_pick_default_layer_bounds(tiny_model):
    layer = tiny_model.pick_default_layer()
    assert 0 <= layer < tiny_model.n_layers
    assert tiny_model.pick_default_layer(0.0) == 0
    assert tiny_model.pick_default_layer(1.0) == tiny_model.n_layers - 1


def test_extract_activations_single_and_batch(tiny_model):
    a = tiny_model.extract_activations("Hello world", layer=2)
    assert a.shape == (tiny_model.hidden_size,)
    B = tiny_model.extract_activations(["Hello world", "Goodbye world"], layer=2)
    assert B.shape == (2, tiny_model.hidden_size)


def test_extract_activations_mean_position(tiny_model):
    B = tiny_model.extract_activations(["Hello world", "A longer prompt here today"], layer=2, position="mean")
    assert B.shape == (2, tiny_model.hidden_size)
    assert torch.isfinite(B).all()


def test_extract_activations_bad_layer(tiny_model):
    try:
        tiny_model.extract_activations("hi", layer=999)
        assert False, "devrait lever ValueError"
    except ValueError:
        pass


def test_generate_single_and_many(tiny_model):
    set_seed(0)
    single = tiny_model.generate("The movie was", max_new_tokens=8)
    assert isinstance(single, str) and len(single) > 0
    many = tiny_model.generate(["The movie was", "The food was"], max_new_tokens=8)
    assert isinstance(many, list) and len(many) == 2
    assert all(isinstance(m, str) and m for m in many)


def test_generate_deterministic_with_seed(tiny_model):
    set_seed(123)
    a = tiny_model.generate("Once upon a time", max_new_tokens=12, seed=123)
    b = tiny_model.generate("Once upon a time", max_new_tokens=12, seed=123)
    assert a == b


def test_generate_apply_chat_noop_without_template(tiny_model):
    out = tiny_model.generate("Hi there", max_new_tokens=4, apply_chat=True)
    assert isinstance(out, str)


def test_forward_logits_shape(tiny_model):
    logits = tiny_model.forward_logits("The movie was")
    assert logits.shape == (tiny_model.vocab_size,)


def test_word_logprob_single_token(tiny_model):
    probs = tiny_model.word_logprob("The movie was", ["great", "bad", "excellent"])
    assert set(probs) == {"great", "bad", "excellent"}  # chaque mot demandé est rendu
    for v in probs.values():
        assert v <= 0.0  # log-probabilités


def test_read_returns_metrics(tiny_model):
    import torch

    v = torch.randn(tiny_model.hidden_size)
    metrics = tiny_model.read("The movie was", v, layer=2)
    assert set(metrics) == {"cosine", "projection"}
    assert -1.0 <= metrics["cosine"] <= 1.0
    assert metrics["projection"] == metrics["projection"]  # pas NaN
