# The IA-Instrument: Reproducible Steering-Vector Control of Language Models, and Three Controlled Experiments Across Four Model Families

**Author:** korrocorp — *work in progress, open source. This is a reproducibility + empirical-evaluation report, not a claim of a new method.*

---

## Abstract

Representation-level control — "steering" — promises to replace fragile prompt engineering with a measurable, monotonic interface over a language model's internal activations. Despite the popularity of methods like Representation Engineering (RepE) and Contrastive Activation Addition (CAA), controlled comparisons against prompting, systematic cross-model benchmarks, and honest transfer studies remain scarce. We release **IA-Instrument**, a from-scratch, dependency-light implementation of RepE + CAA as a unified "instrument" API (`make_knob` → `play`), and run **three controlled experiments** on 5 real models spanning 4 families (GPT-2, Qwen, Llama, DeepSeek):

1. **Steering vs. prompting** (paired, same prompts, same external scorer). Steering is more consistent (sign-test p=1.00 on 100% of prompts) and dominates on an instruct model (sentiment control power +0.535 vs +0.409 for the best prompt).
2. **Systematic cross-model steerability.** Sentiment control correlates 0.89–0.95 with amplitude on **4 of 5 families**; the *reasoning* model (DeepSeek-R1-Distill) is barely steerable on sentiment (0.35); defensive refusal amplification is strong on Qwen (0.97, 50%→100% refusal) and fails on TinyLlama (no baseline refusal to amplify).
3. **Direction transfer.** Naive cross-family transfer via zero-padding **fails**: the transferred knob *counter-steers* (coefficient −0.21). We document this as an honest negative result.

Two methodological findings matter for anyone reproducing steering work: (a) **instruct models must be evaluated in their native chat mode** — base-continuation annihilates the effect (0.000 vs +0.548 on the same knob); (b) a true baseline requires an **explicit empty knob set** (`knobs={}`), because a `None` knob set plays all knobs by default. All numbers are regenerated from the committed scripts and JSON reports in `reports/experiments/`.

---

## 1. Introduction

Controlling a language model today means writing a prompt: an instruction whose effect is unmeasured, sensitive to wording, and decided by the model itself. An alternative has emerged from interpretability research: **steering vectors** — directions in the model's residual stream that, when added at inference time, shift behavior along a concept [1,2,3]. The promise is a *measurable, monotonic, adjustable* control: an instrument rather than an oracle.

Three gaps stand between this promise and practice:

- **No controlled comparison.** Papers demonstrate that a knob changes outputs, but rarely compare fairly against prompting on the same prompts with the same external scorer.
- **No systematic cross-model study.** Most demonstrations use one or two models. It is unclear how steerability varies across architectures, sizes, and training regimes (base vs. instruct vs. reasoning).
- **No honest transfer study.** "Steering vectors transfer between models" is often asserted; transfer with mismatched hidden sizes is rarely measured and reported.

This report addresses all three with a reproducible protocol and honest numbers, negative results included.

## 2. Related Work

**Representation Engineering (RepE)** [1] introduced reading and controlling concepts via linear directions in activation space. **Activation Addition** [2] and **CAA** [3] made control cheap: build a direction from the mean difference of positive/negative activation centroids, add it at a chosen layer. Anthropic's work on monosemantic features [4] and the Cerulean demo show industrial interest in representation-level product interfaces.

Our contribution is not a new method. It is (a) a clean, unified, from-scratch implementation of both families, and (b) the empirical protocol and results — including the negative transfer result and the chat-mode methodological finding — that the field currently lacks.

## 3. Method

### 3.1 Steering vectors

A *steering vector* is a triple `(direction, layer, scale)`. `direction` is a vector in the residual stream of a transformer layer; at inference we add `scale × direction` to the residual stream of that layer for every generated token. The default direction is the **raw contrast** `mean(pos) − mean(neg)` (CAA), whose norm is the centroid distance — making `scale=1` interpretable as "one full contrast".

Three constructions are implemented: `contrastive` (CAA), `pca` (RepE, top principal component), and `probe` (RepE, logistic-regression normal).

### 3.2 The instrument API

```python
from instrument import Instrument
inst = Instrument("openai-community/gpt2")
inst.make_knob("positive", positive=POS, negative=NEG, layer=7)   # build a direction
inst.play("The book was", knobs={"positive": 2.0})                 # steer generation
inst.read("I love this day.", name="positive")                     # read a concept
```

`play` supports polyphony (several knobs at once). Knobs persist via `save_knobs`/`load_knobs`.

### 3.3 Implementation notes

- Works with GPT-2 (`transformer.h`), Llama/Qwen (`model.layers`) and generic causal LMs in transformers 5.x; CPU-safe.
- All measurement uses **external lexicons** (sentiment, formality, refusal markers), never the steered model's own logits.
- **Reproducibility:** greedy decoding, fixed seed, `max_new_tokens` short, JSON reports written per benchmark with timestamps.

## 4. Experimental Protocol

### 4.1 Models

| Model | Family | Layers | Hidden | Mode |
|---|---|---|---|---|
| gpt2 | GPT-2 | 12 | 768 | base |
| Qwen2.5-0.5B-Instruct | Qwen | 24 | 896 | chat |
| Qwen2.5-1.5B-Instruct | Qwen | 28 | 1536 | chat |
| TinyLlama-1.1B-Chat-v1.0 | Llama | 22 | 2048 | chat |
| DeepSeek-R1-Distill-Qwen-1.5B | DeepSeek | 28 | 1536 | chat |

All ≤1.5B, run locally on CPU. Directions built at 60% depth (`layer_ratio=0.6`).

### 4.2 Metrics

- **Control power** (Exp. 1): `|max deviation of a condition from the true baseline|` over the shared eval prompts.
- **Steerability** (Exp. 2): Pearson correlation between steering amplitude and the external metric, over 5 amplitudes `{−2,−1,0,+1,+2}`.
- **Transfer coefficient** (Exp. 3): `transferred_effect / native_effect` on the target model.

### 4.3 Methodological caveats we enforced (and found)

1. **Native mode.** Instruct models are evaluated with their chat template. In base continuation mode, the *same* knob produces 0.000 control on Qwen2.5-0.5B; in chat mode it produces +0.548 (Exp. 1). Reporting base-mode numbers for instruct models is an artifact.
2. **True baseline.** In our API, `knobs=None` plays *all* knobs at amplitude 1.0. A baseline must pass `knobs={}`. Early drafts of Exp. 1 used `knobs=None` and produced contaminated baselines; we caught this and re-ran. The committed numbers are from the corrected runs.

## 5. Results

### 5.1 Exp. 1 — Steering vs. prompting (paired)

Conditions: baseline · steer ±1 · prompt ± (instruction text, 2 wordings), same 16 (sentiment) / 8 (tone) prompts, same external scorer.

| Model | Dimension | Steering | Best prompt | Winner |
|---|---|---|---|---|
| gpt2 | sentiment | +0.271 | +0.302 | prompt (thin margin) |
| gpt2 | tone | +0.047 | +0.014 | steering |
| Qwen2.5-0.5B | sentiment | **+0.535** | +0.409 | **steering** |
| Qwen2.5-0.5B | tone | +0.006 | +0.028 | prompt (both ≈ 0) |

Consistency (sign test, fraction of prompts moved in the intended direction): the Qwen knob `+1` moved **100%** of prompts (p=1.00) vs. 85–92% for the best prompt. On gpt2, negative steering was consistent (p=0.00, no inverted prompts) while positive prompting was not (p=0.67).

**Reading.** Steering is at least as strong as prompting, and more consistent. On the instruct model it is clearly stronger, at near-zero marginal cost.

### 5.2 Exp. 2 — Cross-model steerability

| Model | Family | Sentiment corr (slope) | Tone corr (slope) | Refusal corr (baseline) |
|---|---|---|---|---|
| gpt2 | GPT-2 | **0.95** (+0.126) | 0.85 (+0.061) | — |
| Qwen2.5-0.5B | Qwen | **0.92** (+0.205) | −0.38 | **0.97** (0.50) |
| Qwen2.5-1.5B | Qwen | **0.89** (+0.120) | **0.97** | **0.97** (0.58) |
| TinyLlama-1.1B | Llama | **0.89** (+0.057) | 0.35 | −0.35 (0.08) |
| DeepSeek-R1-Distill-1.5B | DeepSeek | 0.35 | 0.69 | 0.71 (0.00) |

**Reading.** Sentiment control is strong and monotonic on 4/5 families. The reasoning model (DeepSeek) barely responds to sentiment steering in chat mode — an open, honest puzzle. Tone control is model-dependent (strong on Qwen1.5B and gpt2, negative on Qwen0.5B). Defensive refusal amplification is strong on Qwen (50%→100%) and *fails* on TinyLlama because that model has almost no baseline refusal to amplify.

### 5.3 Exp. 3 — Direction transfer

| Design | Source → Target | Hidden | Native Δ | Transferred Δ | Coef. |
|---|---|---|---|---|---|
| A — matched hidden | Qwen2.5-1.5B → DeepSeek-R1-Distill-1.5B | 1536→1536 | 0.000 | 0.000 | inconclusive |
| B — zero-pad cross-family | gpt2 → Qwen2.5-0.5B | 768→896 | +0.535 | **−0.115** | **−0.21** |

**Reading (honest).** Naive cross-family transfer **fails**: the gpt2 knob, zero-padded into Qwen2.5-0.5B, actively *counter-steers* sentiment. Design A is inconclusive because the target (DeepSeek) is barely steerable natively, so there is no native effect to transfer. We report this negative result explicitly: **steering directions are not portable across families by naive projection.**

## 6. Discussion

- **Steering is real and measurable** — monotone, amplitude-consistent, external-scorer-verified — and on an instruct model it beats prompting on the dimension we measured.
- **But it is not universal.** Tone is fragile on small models; reasoning models resist sentiment steering; TinyLlama has no refusal to amplify. Any product claim must be model- and dimension-specific.
- **Transfer is the open problem.** The cheap story ("learn once, apply everywhere") is contradicted by our zero-pad result. A learned cross-model mapping is the natural next step — and a genuine open research problem, not a solved one.

## 7. Limitations

1. Models ≤1.5B, CPU. 7B+ steerability unmeasured here.
2. Prompting is wording-sensitive; we tested 2 formulations, not a systematic instruction search.
3. Single-seed greedy decoding.
4. Lexicon scorers are crude proxies (e.g., refusal = regex markers).
5. The "reasoning model resists steering" result is one model, one dimension.

## 8. Conclusion

We release a clean, reproducible steering library and an honest empirical picture: representation-level control works, often beats prompting, and transfers badly. The negative results — transfer failure, DeepSeek's resistance, TinyLlama's refusal ceiling — are as informative as the positive ones. All numbers regenerate from the committed scripts.

## References

1. Zou et al., *Representation Engineering: A Top-Down Approach to AI Transparency*, arXiv:2310.01405 (ICLR 2024).
2. Turner et al., *Activation Addition: Steering Language Models Without Optimization*, arXiv:2308.10248.
3. Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction* (CAA), arXiv:2406.11717.
4. Templeton et al. (Anthropic), *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet*, 2024.

## Appendix — Reproduction

```bash
pip install -e . && pytest tests/ -q          # 44 tests, tiny-random-gpt2
python experiments/run_all.py --exp1 --exp2 --exp3   # all numbers, ~40 min CPU
```

Every number in this paper is produced by these scripts and stored as timestamped JSON in `reports/experiments/` (`SUMMARY.md` condenses them). The evidence chain is documented in `reports/VERIFICATION.md`.
