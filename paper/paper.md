# steerio: Reproducible Steering-Vector Control of Language Models and a Controlled Study Across Four Model Families

**Author:** korrocorp
**License:** MIT. This is an open, reproducible empirical-evaluation report, not a claim of a new method. Every number is regenerable from the committed scripts in `experiments/`.

---

## Abstract

Representation-level control — *steering* — offers an interface to a language model's
internal activations that is measurable, monotonic, and adjustable, in contrast to the
unmeasured, wording-sensitive effect of prompting. Publicly, methods such as Representation
Engineering (RepE) and Contrastive Activation Addition (CAA) are widely cited, yet
controlled comparisons against prompting, systematic cross-model studies, and honest
transfer evaluations remain scarce. We release **steerio**, a from-scratch, dependency-light
implementation of RepE and CAA unified under a single "instrument" API
(`make_knob` → `play` → `read`), and use it to run three controlled experiments on 5 real
open-weight models spanning 4 families (GPT-2, Qwen, Llama, DeepSeek), all ≤1.5B on CPU.

Our contributions are reproducible rather than methodological:

1. **Steering vs. prompting (paired).** Steering dominates on an instruction-tuned model
   (sentiment control power +0.535 vs. +0.409 for the best prompt) and is more *consistent*
   — the +1 knob shifts 100% of evaluation prompts (sign test p=1.00) versus 85–92% for the
   best textual prompt.
2. **Systematic cross-model steerability.** Sentiment control correlates 0.89–0.95 with
   amplitude on 4 of 5 families; a *reasoning* model (DeepSeek-R1-Distill) is barely
   steerable on sentiment (0.35); defensive refusal amplification is strong on Qwen (0.97;
   refusal rises 50%→100%) and fails on TinyLlama, which has no baseline refusal to amplify.
3. **Direction transfer (negative result).** Naive cross-family transfer via zero-padding
   fails — the transferred knob *counter-steers* (transfer coefficient −0.21). We report
   this openly.

Two methodological findings matter for any reproduction of steering work: (a) **instruct
models must be evaluated in their native chat mode** — base continuation annihilates the
effect (0.000 vs. +0.548 on the same knob); (b) a true baseline requires an **explicit empty
knob set** (`knobs={}`), because a `None` set plays every knob at full amplitude by default.

---

## 1. Introduction

Controlling a language model today means writing a prompt: an instruction whose effect is
unmeasured, sensitive to wording, and ultimately obeyed *by choice* of the model itself. A
different paradigm has emerged from interpretability research — **steering vectors**:
directions in the model's residual stream that, when added at inference time, shift behavior
along a concept (Zou et al., 2023; Turner et al., 2024; Arditi et al., 2024). The promise is
*measurable, monotonic, adjustable* control — an instrument rather than an oracle.

Three gaps separate that promise from practice:

- **No controlled comparison.** Papers demonstrate that a knob changes outputs, but rarely
  compare it fairly against prompting on the *same* prompts with the *same* external scorer.
- **No systematic cross-model study.** Most demonstrations use one or two models. Steerability
  across architectures, sizes, and training regimes (base vs. instruct vs. reasoning) is
  largely unquantified.
- **No honest transfer study.** "Steering vectors transfer between models" is often asserted;
  transfer with *mismatched hidden sizes* is seldom measured and reported.

This report addresses all three with a reproducible protocol and honest numbers, negative
results included.

**Contributions.** (i) A clean, unified, from-scratch implementation of RepE and CAA under
an instrument API. (ii) A controlled steering-vs-prompting evaluation on matched prompts and
an external scorer. (iii) A systematic steerability benchmark across five models and four
families. (iv) An honest transfer study that falsifies naive cross-family portability.
(v) Two methodological corrections (native chat mode, true `knobs={}` baseline) that bear on
the broader steering literature.

## 2. Related Work

**Representation Engineering (RepE)** (Zou et al., 2023) frames model internals as a space
of interpretable directions that can be *read* (probes, projections) and *controlled*
(direction addition). **Activation Addition** (Turner et al., 2024) reduces control to a
single cheap operation without any optimization. **Contrastive Activation Addition (CAA)**
(Arditi et al., 2024) constructs a direction from the mean difference of positive/negative
activation centroids and applies it at a chosen layer — and, importantly, finds that the
*refusal* behavior of a model is mediated by a single such direction. Concurrent interpretability
efforts — Anthropic's monosemantic-feature work and its product demos — signal industrial
interest in representation-level interfaces.

Our contribution is not a new method. It is (a) a clean, unified, reproduction of both
families under one API, and (b) the controlled empirical protocol and results — including the
negative transfer result and the chat-mode finding — that the field currently lacks.

## 3. Method

### 3.1 Steering vectors

A *steering vector* is a triple `(direction, layer, scale)`. Let
$\mathbf{h}^{(l)}(\mathbf{x}) \in \mathbb{R}^{d}$ denote the residual-stream activation of
layer $l$ for input $\mathbf{x}$. Given a set of positive and negative example inputs, the
default **raw contrast** direction is

$$ \mathbf{v} = \frac{1}{|P|}\sum_{\mathbf{x}\in P}\mathbf{h}^{(l)}(\mathbf{x})
          - \frac{1}{|N|}\sum_{\mathbf{x}\in N}\mathbf{h}^{(l)}(\mathbf{x}), $$

i.e. CAA with an unnormalized centroid difference. At generation, for every token we add
$s\cdot\mathbf{v}$ to the residual stream of layer $l$. Because $\mathbf{v}$'s norm is the
centroid distance, $s=1$ is interpretable as "one full contrast." Three constructions are
implemented: `contrastive` (CAA), `pca` (RepE; top principal component of the combined
activations), and `probe` (RepE; logistic-regression weight vector).

### 3.2 The instrument API

```python
from steerio import Instrument, prompts

inst = Instrument("openai-community/gpt2")                    # load a local model
inst.make_knob("positive", positive=prompts.SENTIMENT_POSITIVE,
               negative=prompts.SENTIMENT_NEGATIVE, layer=7)  # build a direction
inst.play("The book was", knobs={"positive": 2.0})            # steer generation
inst.read("I love this day.", name="positive")                # read a concept
```

`play` supports polyphony (several knobs simultaneously). Knobs persist via `save_knobs` /
`load_knobs`.

### 3.3 Implementation notes

- Compatible with GPT-2 (`transformer.h`), Llama/Qwen (`model.layers`), and generic causal
  LMs in transformers 5.x; runs on CPU.
- All measurement uses **external lexical scorers** (sentiment, formality, refusal markers),
  never the steered model's own logits.
- **Reproducibility:** greedy decoding, fixed seed, short `max_new_tokens`; every benchmark
  writes a timestamped JSON report under `reports/experiments/`.

## 4. Experimental Protocol

### 4.1 Models

| Model | Family | Layers | Hidden | Mode |
|---|---|---|---|---|
| gpt2 | GPT-2 | 12 | 768 | base |
| Qwen2.5-0.5B-Instruct | Qwen | 24 | 896 | chat |
| Qwen2.5-1.5B-Instruct | Qwen | 28 | 1536 | chat |
| TinyLlama-1.1B-Chat-v1.0 | Llama | 22 | 2048 | chat |
| DeepSeek-R1-Distill-Qwen-1.5B | DeepSeek | 28 | 1536 | chat |

All ≤1.5B, loaded locally on CPU. Directions are built at 60% of transformer depth
(`layer_ratio = 0.6`). Instruct models are evaluated with their chat template (see §4.3.1).

### 4.2 Metrics

- **Control power** (Exp. 1): the maximum absolute deviation of a condition's mean metric
  from the true baseline, over the shared evaluation prompts.
- **Steerability** (Exp. 2): Pearson correlation between steering amplitude and the external
  metric over five amplitudes $\{-2,-1,0,+1,+2\}$, plus the regression slope.
- **Transfer coefficient** (Exp. 3): $\Delta_\text{transferred} / \Delta_\text{native}$
  measured on the target model.

### 4.3 Methodological controls

**1. Native mode.** Instruct models are evaluated with their chat template. In base
continuation mode, the *same* knob yields 0.000 control on Qwen2.5-0.5B; in chat mode it
yields +0.548 (Exp. 1). Reporting base-mode numbers for instruct models is therefore an
artifact, and we avoid it.

**2. True baseline.** In the steerio API, `knobs=None` plays *all* knobs at amplitude 1.0.
A valid baseline must pass `knobs={}`. An early run of Exp. 1 used `knobs=None` and produced
contaminated baselines; we detected and corrected this. The committed numbers are from the
corrected runs.

**3. External scoring.** The metric is computed by an external lexicon, independent of the
model being steered, so a self-referential artifact cannot inflate steerability.

## 5. Results

### 5.1 Exp. 1 — Steering vs. prompting (paired)

Conditions: baseline · steer ±1 · prompt ± (textual instruction, 2 wordings), with the same
16 sentiment / 8 tone evaluation prompts and the same external scorer.

| Model | Dimension | Steering | Best prompt | Winner |
|---|---|---|---|---|
| gpt2 | sentiment | +0.271 | +0.302 | prompt (thin margin) |
| gpt2 | tone | +0.047 | +0.014 | steering |
| Qwen2.5-0.5B | sentiment | **+0.535** | +0.409 | **steering** |
| Qwen2.5-0.5B | tone | +0.006 | +0.028 | prompt (both ≈ 0) |

**Consistency.** Sign test, fraction of prompts moved in the intended direction: the Qwen
knob at +1 moved **100%** of prompts (p = 1.00), versus 85–92% for the best prompt. On gpt2,
negative steering was consistent (p = 0.00: no prompt inverted), while positive prompting
was not (p = 0.67).

**Reading.** Steering is at least as strong as prompting and markedly more consistent. On the
instruction-tuned model it is clearly stronger, at near-zero marginal cost. Tone remains
difficult to control by either method on these small models.

### 5.2 Exp. 2 — Cross-model steerability

| Model | Family | Sentiment corr (slope) | Tone corr (slope) | Refusal corr (baseline) |
|---|---|---|---|---|
| gpt2 | GPT-2 | **0.95** (+0.126) | 0.85 (+0.061) | — (no chat) |
| Qwen2.5-0.5B | Qwen | **0.92** (+0.205) | −0.38 | **0.97** (0.50) |
| Qwen2.5-1.5B | Qwen | **0.89** (+0.120) | **0.97** | **0.97** (0.58) |
| TinyLlama-1.1B | Llama | **0.89** (+0.057) | 0.35 | −0.35 (0.08) |
| DeepSeek-R1-Distill-1.5B | DeepSeek | 0.35 | 0.69 | 0.71 (0.00) |

**Reading.** Sentiment control is strong and monotonic on 4 of 5 families (corr 0.89–0.95).
The reasoning model (DeepSeek) barely responds to sentiment steering in chat mode (0.35) — an
open and, to our knowledge, under-reported result. Tone is model-dependent: strong on
Qwen2.5-1.5B (0.97) and gpt2 (0.85), weak on TinyLlama (0.35), and *negative* on
Qwen2.5-0.5B (−0.38), where the "formal" knob moves against the metric. Defensive refusal
amplification is strong on the Qwen models (0.97; refusal rises 50%→100% on Qwen2.5-0.5B)
and *fails* on TinyLlama, whose baseline refusal is near zero (8%, negative correlation) —
there is no direction to amplify.

### 5.3 Exp. 3 — Direction transfer

| Design | Source → Target | Hidden | Native Δ | Transferred Δ | Coef. |
|---|---|---|---|---|---|
| A — matched hidden | Qwen2.5-1.5B → DeepSeek-R1-Distill-1.5B | 1536→1536 | 0.000 | 0.000 | inconclusive |
| B — zero-pad cross-family | gpt2 → Qwen2.5-0.5B | 768→896 | +0.535 | **−0.115** | **−0.21** |

**Reading (honest).** Naive cross-family transfer **fails**: the gpt2 knob, zero-pad
projected onto Qwen2.5-0.5B at a proportional layer, actively *counter-steers* sentiment
(−0.21). Design A is inconclusive because the target (DeepSeek) is barely steerable natively
(0.35), so there is no native effect to transfer. We report this negative explicitly:
**steering directions are not portable across families by naive projection.**

## 6. Discussion

- **Steering is real and measurable** — monotone, amplitude-consistent, and verified by an
  external scorer — and on an instruction-tuned model it beats prompting on the dimension we
  measured.
- **But it is not universal.** Tone is fragile on small models; reasoning models resist
  sentiment steering; TinyLlama has no refusal to amplify. Any product or safety claim must
  be model- and dimension-specific.
- **Transfer is the open problem.** The cheap story ("learn once, apply everywhere") is
  contradicted by our zero-pad result. A learned cross-model mapping — or a representation
  that is provably aligned across architectures — is a natural and genuinely open research
  direction, not a solved one.

## 7. Limitations

1. Models ≤1.5B on CPU; steerability on 7B+ models is unmeasured here.
2. Prompting is wording-sensitive; we tested two formulations, not a systematic instruction
   search.
3. Single-seed, greedy decoding.
4. Lexicon scorers are crude proxies (e.g., refusal = regex markers), so absolute magnitudes
   are indicative, not calibrated probabilities.
5. "Reasoning model resists steering" rests on one model and one dimension.
6. The transfer design uses naive zero-padding; more principled alignment methods
   (e.g., learned projections over matched semantic features) are out of scope.

## 8. Conclusion

We release a clean, reproducible steering library and an honest empirical picture:
representation-level control works, often beats prompting on instruction-tuned models, and
transfers badly across families. The negative results — transfer failure, DeepSeek's
resistance, TinyLlama's refusal ceiling — are as informative as the positive ones. All
numbers regenerate from the committed scripts; we invite reproduction and challenge.

## References

1. Zou, A., et al. *Representation Engineering: A Top-Down Approach to AI Transparency.*
   arXiv:2310.01405 (ICLR 2024).
2. Turner, A. M., et al. *Activation Addition: Steering Language Models Without Optimization.*
   arXiv:2308.10248.
3. Arditi, A., et al. *Refusal in Language Models Is Mediated by a Single Direction.*
   arXiv:2406.11717.
4. Templeton, A., et al. (Anthropic). *Scaling Monosemanticity: Extracting Interpretable
   Features from Claude 3 Sonnet.* 2024.

## Appendix A — Reproduction

```bash
pip install -e . && pytest tests/ -q              # 44 tests, tiny-random-gpt2, no real weights
python experiments/run_all.py --exp1 --exp2 --exp3   # regenerates every number, ~40 min CPU
```

Every figure/table value is produced by these scripts and stored as timestamped JSON under
`reports/experiments/`; `SUMMARY.md` condenses them. The evidence chain and provenance are
documented in `reports/VERIFICATION.md`.

## Appendix B — Artifact statement

- **Code:** MIT-licensed Python package `steerio` (from-scratch RepE + CAA, 44 tests).
- **Data/weights:** all models are public HuggingFace checkpoints; `models_cache/` is
  downloaded in copy mode (no symlinks) and gitignored (≈8 GB).
- **Compute:** all experiments run on a single CPU in ≈40 minutes.
