"""Tests du mécanisme d'injection : le filtre modifie la distribution, l'échelle 0 est neutre."""
import torch

from instrument.steering import SteeringEntry, SteeringFilter
from instrument.vectors import normalize


def _rand_dir(tiny_model):
    return normalize(torch.randn(tiny_model.hidden_size))


def test_filter_adds_to_tensor_output():
    filt = SteeringFilter([SteeringEntry(torch.tensor([1.0, 2.0]), 1.0)])
    out = torch.zeros(2)
    result = filt(None, None, out)
    assert result is not None
    assert torch.allclose(result, torch.tensor([1.0, 2.0]))


def test_filter_handles_tuple_output():
    filt = SteeringFilter([SteeringEntry(torch.tensor([1.0, 2.0]), 1.0)])
    h = torch.zeros(2)
    cache = (torch.ones(3), "payload")
    result = filt(None, None, (h, cache))
    assert isinstance(result, tuple)
    assert torch.allclose(result[0], torch.tensor([1.0, 2.0]))
    assert result[1] is cache


def test_filter_disabled_returns_none():
    filt = SteeringFilter([SteeringEntry(torch.tensor([1.0]), 5.0)])
    filt.enabled = False
    assert filt(None, None, torch.zeros(1)) is None


def test_empty_filter_returns_none():
    filt = SteeringFilter()
    assert filt(None, None, torch.zeros(1)) is None


def test_steering_changes_logits(tiny_model):
    layer = tiny_model.pick_default_layer()
    v = _rand_dir(tiny_model)
    filt = SteeringFilter([SteeringEntry(v, 8.0)])
    base = tiny_model.forward_logits("The movie was")
    handles = [tiny_model.layers[layer].register_forward_hook(filt)]
    try:
        steered = tiny_model.forward_logits("The movie was")
    finally:
        for h in handles:
            h.remove()
    assert not torch.allclose(base, steered, atol=1e-8), "le steering doit modifier les logits"


def test_steering_scale_zero_is_neutral(tiny_model, tiny_instrument):
    tiny_instrument.make_knob("pos", ["I love this great day."], ["I hate this bad day."], layer=tiny_model.pick_default_layer())
    import instrument.model as m

    m.set_seed(0)
    a = tiny_instrument.model.generate("The movie was", max_new_tokens=8, seed=0)
    m.set_seed(0)
    b = tiny_instrument.play("The movie was", max_new_tokens=8, knobs={"pos": 0.0}, seed=0)
    assert a == b
