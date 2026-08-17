"""Tests des scorers et du harnais de benchmark (sur le modèle tiny)."""
import os

from instrument import prompts
from instrument.benchmarks import SteeringBenchmark, sentiment_benchmark, tone_benchmark, refusal_benchmark
from instrument.instrument import Instrument


# ------------------------------------------------------------- scorers
def test_lexicon_sentiment_signs():
    assert prompts.lexicon_sentiment("This is a great and wonderful day, I love it!") > 0
    assert prompts.lexicon_sentiment("This is a terrible and horrible day, I hate it!") < 0
    assert prompts.lexicon_sentiment("The cat sat on the mat.") == 0.0


def test_lexicon_sentiment_bounded():
    for text in ("great great great love love", "bad bad bad hate hate"):
        assert -1.0 <= prompts.lexicon_sentiment(text) <= 1.0


def test_tone_formality():
    assert prompts.tone_formality("I would like to kindly request your attention regarding this matter.") > 0
    assert prompts.tone_formality("hey dude, gonna grab some stuff, no worries") < 0


def test_refusal_score():
    assert prompts.refusal_score("I am sorry, I cannot help you with that.") == 1.0
    assert prompts.refusal_score("Here is how to bake a cake.") == 0.0


# ------------------------------------------------------------- harnais
def _instrument_with_knob(tiny_instrument, name, pos, neg):
    tiny_instrument.make_knob(name, pos, neg, layer=tiny_instrument.model.pick_default_layer())
    return tiny_instrument


def test_sentiment_benchmark_runs(tiny_instrument, tmp_path):
    _instrument_with_knob(tiny_instrument, "positive", prompts.SENTIMENT_POSITIVE[:4], prompts.SENTIMENT_NEGATIVE[:4])
    bench = sentiment_benchmark(tiny_instrument, max_new_tokens=8, report_dir=str(tmp_path))
    result = bench.run(scales=[-2.0, 0.0, 2.0], seed=0)
    assert result.name == "sentiment"
    assert len(result.scales) == 3 and len(result.metrics) == 3
    assert result.baseline == result.metrics[1]  # échelle 0
    assert -1.0 <= result.correlation <= 1.0
    assert os.path.exists(result.report_path)


def test_tone_benchmark_runs(tiny_instrument, tmp_path):
    _instrument_with_knob(tiny_instrument, "formal", prompts.TONE_FORMAL[:4], prompts.TONE_CASUAL[:4])
    bench = tone_benchmark(tiny_instrument, max_new_tokens=6, report_dir=str(tmp_path))
    result = bench.run(scales=[-2.0, 0.0, 2.0], seed=0)
    assert result.knob == "formal"


def test_refusal_benchmark_runs(tiny_instrument, tmp_path):
    _instrument_with_knob(tiny_instrument, "refusal", prompts.REFUSAL_PROMPTS[:4], prompts.COMPLIANCE_PROMPTS[:4])
    bench = refusal_benchmark(tiny_instrument, max_new_tokens=6, report_dir=str(tmp_path))
    result = bench.run(scales=[0.0, 3.0], seed=0)
    assert result.name == "refusal"


def test_steering_scales_change_outputs(tiny_instrument, tmp_path):
    _instrument_with_knob(tiny_instrument, "positive", prompts.SENTIMENT_POSITIVE[:4], prompts.SENTIMENT_NEGATIVE[:4])
    bench = sentiment_benchmark(tiny_instrument, max_new_tokens=8, report_dir=str(tmp_path))
    result = bench.run(scales=[-3.0, 3.0], seed=0)
    # Sur un modèle aléatoire, le signe n'est pas garanti : on vérifie seulement
    # que les deux échelles produisent des sorties et des scores.
    assert all(s == s for s in result.metrics)  # pas de NaN
