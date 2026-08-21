# Posts — IA-Instrument launch (English)

> Tone: 100% human, zero hype. Every number below is REAL (`experiments/run_all.py`,
> regenerable). We are NOT claiming to "revolutionize AI" — we ship a clean, reproducible
> steering-vector harness + an honest benchmark, including the parts that FAIL.
> ⚠️ Replace `https://github.com/KorroAi/steerio` with the real repo URL after the push.

---

## 🧵 X thread (14 tweets, English)

**1.** I built an open-source instrument that lets you TURN KNOBS inside a language model
instead of writing prompts. The results are... honest. 🧵

**2.** What is steering? A LLM has directions inside its hidden layers that correspond to
concepts (positive, formal, refusal). A steering vector is one of those directions, added
at inference time. Amplitude −2 → +2 = turning a volume knob. Measurable, monotonic.

**3.** The lib (Python + torch, from scratch, ~1900 lines): `make_knob("positive",
pos_examples, neg_examples)` then `play(prompt, knobs={"positive": 2.0})`. RepE (Zou 2023)
+ CAA (Arditi 2024), 44 tests, MIT. I didn't reinvent the method — I made one reproducible.

**4.** Does it actually shift the output? Here's `play("The food at the restaurant was",
knobs={"positive": -1})` → *"so bad that I had to leave. I was so angry"*. And
`knobs={"positive": +1}` → *"delicious, and the service was great. The staff was friendly."*
Same prompt, opposite knob, opposite sentiment.

**5.** It's not one lucky sample. The curve is **monotonic**: amplitude → effect, and the
effect is measurable. That's the part most steering papers skip — they show one example.

**6.** Our clean benchmark: **5 real open models, 4 families** (gpt2, Qwen 0.5B/1.5B,
TinyLlama 1.1B, DeepSeek-R1-Distill 1.5B). Same protocol, external lexical scorers, seed 0.

**7.** Sentiment is **steerable on 4 of 5 families** (correlation 0.89–0.95). gpt2 = 0.95,
Qwen = 0.92 / 0.89, TinyLlama = 0.89. That's a solid, reproducible signal.

**8.** The **cliff** (the honest part): **DeepSeek-R1-Distill only responds at 0.35** —
reasoning models barely move. And on TinyLlama, steerable *tone* but no baseline refusal.

**9.** Steering vs prompting on the instruct model (Qwen 0.5B): steering **wins**
(+0.535 vs +0.409) AND is more **coherent** — it shifted **100% of 16 prompts** (sign test
p=1.00) vs 85–92% for the best prompt. Same job, fewer surprises.

**10.** The big caveat nobody highlights: **direction transfer FAILS.** A naive zero-pad
gpt2 → Qwen transfer gives **−0.21** — it *counter-steers*. So you can't build one knob and
reuse it everywhere; you build per model, per layer, on open-weight models.

**11.** What it's honestly for (and not): it's a **research / teaching harness**, not a
production control panel. You need open weights to inject a vector — so it's local small
models, not GPT/Claude. And raw steering can degrade coherence.

**12.** Who uses it: (a) interpretability researchers who want a reproducible CAA/RepE
baseline + parameter sweeps on small models; (b) anyone running a small local model who
wants a "formality" / "positivity" knob instead of re-writing prompts; (c) teachers making
representations tangible — `read("I love this day.", name="positive")` shows the concept.

**13.** Why open-source it: most steering papers show ONE hand-picked example. This repo
**asks whether steering works**, reports the curve, and publishes where it fails. That's the
contribution — a reproducible protocol + honest negatives, not a miracle.

**14.** `pip install` the repo, load a tiny model, turn a knob, see the shift. Everything in
this thread reproduces with `experiments/run_all.py`. Repo: `https://github.com/KorroAi/steerio`.

---

## 📄 GitHub description (short, English)

> A reproducible steering-vector harness (RepE + CAA) under a "musical instrument" API —
> turn a knob inside a small open model instead of writing a prompt. Ships an honest
> benchmark on 5 open models (4 families): sentiment steerable on 4/5 (corr 0.89–0.95),
> DeepSeek-reasoning resists (0.35), and **cross-model direction transfer fails** (−0.21).
> Not a new method — a reproducible protocol + the negative results most papers skip.

---

## 💬 Hacker News / LinkedIn blurb (English)

> Most "steering vectors" posts show one cherry-picked example. I built an open, reproducible
> harness that actually measures whether steering works on open small models — and it
> reports the failures too. Sentiment steers monotonically on 4/5 families (corr .89–.95);
> DeepSeek-reasoning barely moves (.35); and cross-model direction transfer **doesn't work**
> (−.21, it counter-steers). It's an interpretability tool, not a production trick. Repo:
> `https://github.com/KorroAi/steerio`. Happy to be wrong — run `experiments/run_all.py` and check.
