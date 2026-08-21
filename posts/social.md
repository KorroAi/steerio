# steerio launch posts (English)

> Tone: 100% human, zero hype. Every number below is real (`experiments/run_all.py`,
> regenerable). We do NOT claim to "revolutionize AI." We ship a reproducible steering
> harness plus an honest benchmark, including the parts that fail.
> Repo: https://github.com/KorroAi/steerio

---

## 🧵 X thread (12 tweets, English)

**1.** I built an open source tool that lets you TURN KNOBS inside a language model instead
of writing prompts. The results are honest. 🧵

**2.** What is steering? A LLM has directions inside its hidden layers that map to concepts:
positive, formal, refusal. A steering vector is one of those directions, added at inference
time. Amplitude −2 to +2 turns a volume knob. Measurable, monotonic.

**3.** The lib: `make_knob("positive", pos_examples, neg_examples)` then
`play(prompt, knobs={"positive": 2.0})`. RepE (Zou 2023) + CAA (Arditi 2024), from scratch,
44 tests, MIT. I did not reinvent the method. I made it reproducible.

**4.** Does it shift the output? `play("The food at the restaurant was", knobs={"positive": -1})`
→ "so bad that I had to leave. I was so angry." And `knobs={"positive": +1}` → "delicious, and
the service was great. The staff was friendly." Same prompt, opposite knob, opposite sentiment.

**5.** Not one lucky sample. The curve is monotonic: amplitude to effect, measurable. That is
the part most steering papers skip. They show one example.

**6.** Our benchmark: 5 real open models, 4 families (gpt2, Qwen 0.5B and 1.5B, TinyLlama
1.1B, DeepSeek-R1-Distill 1.5B). Same protocol, external lexical scorers, seed 0.

**7.** Sentiment is steerable on 4 of 5 families. Correlation 0.89 to 0.95. gpt2 0.95,
Qwen 0.92 and 0.89, TinyLlama 0.89. A solid, reproducible signal.

**8.** The cliff. The honest part: DeepSeek-R1-Distill only responds at 0.35. Reasoning models
barely move. And TinyLlama has steerable tone but no baseline refusal.

**9.** Steering versus prompting on the instruct model (Qwen 0.5B): steering wins
(+0.535 vs +0.409) and is more coherent. It shifted 100% of 16 prompts (sign test p=1.00)
versus 85 to 92% for the best prompt. Same job, fewer surprises.

**10.** The big caveat nobody highlights. Direction transfer FAILS. A naive zero pad from
gpt2 to Qwen gives −0.21. It counter steers. So you build one knob per model, per layer, on
open weight models.

**11.** Who uses it. Interpretability researchers who want a reproducible CAA and RepE
baseline. Anyone running a small local model who wants a formality or positivity knob instead
of rewriting prompts. Teachers making representations tangible.

**12.** Why open source it. Most steering papers show ONE hand picked example. This repo asks
whether steering works, reports the curve, and publishes where it fails. That is the
contribution: a reproducible protocol plus honest negatives, not a miracle. Repo:
[https://github.com/KorroAi/steerio](https://github.com/KorroAi/steerio)

---

## 📄 GitHub description

> Turn a knob inside a small open model instead of writing a prompt. A reproducible RepE and
> CAA steering harness with an honest benchmark, including the failures.

---

## 💬 Hacker News / LinkedIn blurb (English)

> Most steering vector posts show one cherry picked example. I built an open, reproducible
> harness that actually measures whether steering works on open small models, and reports the
> failures too. Sentiment steers monotonically on 4 of 5 families (corr 0.89 to 0.95).
> DeepSeek reasoning barely moves (0.35). Cross model direction transfer does not work, it
> counter steers at −0.21. It is an interpretability tool, not a production trick. Repo:
> https://github.com/KorroAi/steerio. Happy to be wrong. Run `experiments/run_all.py` and check.
