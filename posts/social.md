# Posts — IA-Instrument launch

> Ton : 100 % humain, zéro bullshit. Les chiffres sont RÉELS (regenerable via
> `experiments/run_all.py`). On ne prétend PAS « révolutionner l'IA » — on
> montre une lib propre + des résultats honnêtes, y compris négatifs. C'est ça
> qui génère le vrai buzz technique.
> ⚠️ URL repo : REMPLACER `<REPO_URL>` partout par l'URL réelle APRÈS le push
> (le compte GitHub authentifié est KorroAi → probablement
> `https://github.com/KorroAi/ia-instrument`).

---

## Thread X (14 tweets)

**1** — I built an open-source instrument that lets you TURN KNOBS inside a language model instead of writing prompts. The results are... honest. 🧵

**2** — What is steering? A LLM has directions inside its layers that correspond to concepts (positive, formal, refusal). A steering vector is one of those directions, added at inference time. Amplitude -2 → +2 = turning a volume knob. Measurable, monotonic.

**3** — The lib (Python, torch, ~1900 lines): `make_knob("positive", pos_examples, neg_examples)` then `play(prompt, knobs={"positive": 2.0})`. RepE + CAA, from scratch. 44 tests. MIT.

**4** — I then ran 3 controlled experiments on 5 real models, 4 families (GPT-2, Qwen, Llama, DeepSeek). Same prompts, same external scorer, greedy, fixed seed. Everything regenerates from the repo.

**5** — Exp 1: steering vs prompting, paired. On Qwen2.5-0.5B, steering moved 100% of the prompts (sign test p=1.00). The best prompt moved 85-92%. And steering won the effect size: +0.535 vs +0.409.

**6** — Prompting is word-sensitive. We tried 2 wordings. One "negative" instruction on gpt2... behaved unpredictably across formulations. Steering is monotone. That's the point.

**7** — Exp 2: cross-model steerability. Sentiment control correlated with amplitude 0.89-0.95 on 4/5 families. Strong, repeatable.

**8** — The interesting one: DeepSeek-R1-Distill (a REASONING model) barely responded to sentiment steering: 0.35. A reasoning model resists representation-level nudging on sentiment. That's a real, honest puzzle.

**9** — Defensive refusal steering worked great on Qwen: 50% → 100% refusal rate on borderline prompts. And it FAILED on TinyLlama, because TinyLlama barely refuses at baseline — nothing to amplify.

**10** — Exp 3: the transfer experiment. Does a knob learned on gpt2 work on Qwen? Short answer: NO. Zero-padded transfer gave coefficient -0.21. The knob actively counter-steered.

**11** — That's a negative result, and it's the most important one. "Learn once, steer everywhere" is not true with naive projection. Cross-model steering transfer is an OPEN research problem, not a solved one.

**12** — Two methodological findings for anyone reproducing steering work: (1) instruct models MUST be evaluated in chat mode — base continuation annihilates the effect (0.000 vs +0.548 on the same knob). (2) A true baseline is `knobs={}`, not "no knob arg" — a None knob set plays ALL knobs.

**13** — What this is NOT: a new method. It's a clean, reproducible implementation of RepE+CAA + an honest empirical study (negative results included). The paper and every JSON report are in the repo.

**14** — Repo + paper + verification chain: <REPO_URL>. If you reproduce steering experiments, the chat-mode and baseline findings alone are worth a look. Feedback welcome. 🙏

---

## LinkedIn

**Headline:** Open-source LLM steering: I measured it against prompting, across 4 model families, and found where it fails.

**Body:**

Prompt engineering has a problem nobody likes to talk about: the effect is unmeasured. You write "be more positive" and the model decides how much to comply — if at all.

Representation steering promises a different interface: a vector inside the model's layers that you add at inference time, like turning a knob. Amplitude +2 is measurably more positive than +1. It's monotone, it's cheap, and it's auditable.

I open-sourced an implementation (RepE + CAA, from scratch, ~1900 lines, MIT) and ran three controlled experiments on 5 models across 4 families — GPT-2, Qwen, Llama, and DeepSeek.

The honest results:

1. Steering beat prompting on an instruct model (0.535 vs 0.409 effect size) and was more consistent (100% of prompts moved, vs 85-92% for the best prompt).

2. Sentiment steering worked on 4 of 5 families (correlation 0.89-0.95 with amplitude). The exception was DeepSeek-R1-Distill, a reasoning model, which barely responded (0.35). That's a genuine, unsolved puzzle.

3. Defensive refusal steering took Qwen from 50% to 100% refusal on borderline prompts — and failed on TinyLlama, which has almost no baseline refusal to amplify.

4. Cross-model transfer of directions FAILS with naive projection (coefficient -0.21). "Learn once, steer everywhere" is not true. Transfer is an open research problem.

Two methodological gotchas worth knowing before you reproduce steering work: instruct models must be evaluated in their native chat mode (base mode annihilates the effect), and a true baseline requires an explicit empty knob set.

Everything is reproducible: the paper, the scripts, and every JSON report are in the repo. I included the negative results on purpose — that's what makes the numbers trustworthy.

<REPO_URL>

#LLM #AI #Interpretability #OpenSource #MachineLearning

---

## Show HN

**Title:** Show HN: IA-Instrument — steer an LLM like an instrument (RepE + CAA, reproducible benchmarks)

**Text:**

I built an open-source library that controls a language model through its internal representations (steering vectors), and then measured it honestly on 5 real models across 4 families.

The library:
- RepE + CAA, from scratch, ~1900 lines, torch + transformers only
- `make_knob("positive", pos, neg)` → `play(prompt, knobs={"positive": 2.0})`
- read() to measure a concept, polyphony (multiple knobs), persistence, CLI
- 44 tests

The experiments (all numbers regenerate from the repo):

1. Steering vs prompting, paired, same scorer: steering more consistent (100% of prompts vs 85-92%) and stronger on an instruct model (0.535 vs 0.409).

2. Cross-model: sentiment control correlates 0.89-0.95 with amplitude on 4/5 families. DeepSeek-R1-Distill (reasoning) resists: 0.35.

3. Transfer: naive cross-family steering transfer FAILS (coef -0.21). Negative result, reported explicitly.

Two findings I think matter most for anyone doing steering research:
- Instruct models must be evaluated in chat mode. Base continuation gives 0.000 effect; chat mode gives +0.548 on the same knob.
- A "no knobs" baseline must be an explicit empty set. In many APIs (including ours), omitting the knobs argument plays all knobs at full amplitude.

This is NOT a new method — it's a clean implementation + an honest empirical study including the failures. Paper, scripts, and all JSON evidence are in the repo.

<REPO_URL>

Feedback very welcome, especially on the DeepSeek-reasoning-resists-steering result and the transfer negative.
