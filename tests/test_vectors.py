"""Tests unitaires de la math des directions (pure, sans modèle)."""
import torch

from steerio.vectors import (
    SteeringVector,
    compose_directions,
    contrastive_direction,
    cosine_matrix,
    cosine_similarity,
    mean_difference,
    normalize,
    pca_direction,
    probe_direction,
)


def test_normalize_unit_norm():
    v = torch.tensor([3.0, 4.0])
    n = normalize(v)
    assert torch.allclose(n.norm(), torch.ones(()))


def test_normalize_zero_vector_safe():
    n = normalize(torch.zeros(4))
    assert torch.isfinite(n).all()


def test_mean_difference_direction():
    pos = torch.tensor([[1.0, 0.0], [1.0, 0.5]])
    neg = torch.tensor([[0.0, 0.0], [0.0, 0.5]])
    v, pos_mean, neg_mean = mean_difference(pos, neg, normalize_direction=False)
    assert torch.allclose(v, torch.tensor([1.0, 0.0]))
    assert torch.allclose(pos_mean, torch.tensor([1.0, 0.25]))
    assert torch.allclose(neg_mean, torch.tensor([0.0, 0.25]))


def test_contrastive_direction_raw_default():
    pos = torch.tensor([[2.0, 0.0]])
    neg = torch.tensor([[0.0, 0.0]])
    v = contrastive_direction(pos, neg)  # défaut : contraste brut, non normalisé
    assert torch.allclose(v, torch.tensor([2.0, 0.0]))


def test_contrastive_direction_normalized_option():
    pos = torch.tensor([[2.0, 0.0]])
    neg = torch.tensor([[0.0, 0.0]])
    v = contrastive_direction(pos, neg, normalize_direction=True)
    assert torch.allclose(v, torch.tensor([1.0, 0.0]))


def test_cosine_similarity_orthogonal():
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    assert abs(cosine_similarity(a, b).item()) < 1e-6


def test_pca_direction_captures_axis():
    # points along the x-axis (centered) => PC1 along x
    acts = torch.tensor([[3.0, 0.1], [-3.0, -0.1], [1.0, 0.0], [-1.0, 0.0]])
    v = pca_direction(acts)
    assert abs(v[0]) > 0.99  # ~[±1, ~0]


def test_probe_direction_separates():
    torch.manual_seed(0)
    pos = torch.randn(20, 8) + 1.0
    neg = torch.randn(20, 8) - 1.0
    w = probe_direction(pos, neg, steps=300)
    pos_proj = pos @ w
    neg_proj = neg @ w
    assert pos_proj.mean() > neg_proj.mean() + 1.0


def test_cosine_matrix_shape():
    acts = torch.randn(5, 3)
    dirs = torch.randn(2, 3)
    m = cosine_matrix(acts, dirs)
    assert m.shape == (5, 2)


def test_compose_directions():
    v1 = SteeringVector(torch.tensor([1.0, 0.0]), layer=2, scale=1.0)
    v2 = SteeringVector(torch.tensor([0.0, 1.0]), layer=2, scale=2.0)
    total = compose_directions([v1, v2])
    assert torch.allclose(total, torch.tensor([1.0, 2.0]) / torch.tensor(5.0).sqrt())


def test_steering_vector_moves_to_device_dtype():
    sv = SteeringVector(torch.tensor([1.0, 2.0]), layer=1, scale=0.5, name="x")
    assert sv.hidden_size == 2
    sv2 = sv.to("cpu", torch.float64)
    assert sv2.direction.dtype == torch.float64
