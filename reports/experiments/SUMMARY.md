# Résultats d'expériences — l'steerio

> Chiffres **réels** générés par les scripts `experiments/` sur 5 modèles (4 familles),
> protocole commun : CAA couche 60 %, greedy seed 0, scoreurs lexicaux externes,
> mode natif (chat pour instruct), baselines vraies (`knobs={}`).
> Rapports JSON horodatés dans ce dossier.

## Exp 1 — Steering vs prompting (pouvoir de contrôle = |écart max à baseline|)

| Modèle | Dimension | Steering | Prompt |
|---|---|---|---|
| gpt2 | sentiment | +0.271 | +0.302 |
| gpt2 | ton | +0.047 | +0.014 |
| Qwen2.5-0.5B-Instruct | sentiment | +0.535 | +0.409 |
| Qwen2.5-0.5B-Instruct | ton | +0.006 | +0.028 |

## Exp 2 — Corrélation échelle→score

| Modèle | Famille | Sentiment | Ton | Refus |
|---|---|---|---|---|
| gpt2 | GPT-2 | +0.95 | +0.85 | — |
| Qwen2.5-0.5B | Qwen | +0.92 | −0.38 | +0.97 |
| Qwen2.5-1.5B | Qwen | +0.89 | +0.97 | +0.97 |
| TinyLlama-1.1B | Llama | +0.89 | +0.35 | −0.35 |
| DeepSeek-R1-Distill-1.5B | DeepSeek | +0.35 | +0.69 | +0.71 |

## Exp 3 — Transfert de directions

| Design | Delta natif | Delta transféré | Coef |
|---|---|---|---|
| A (hidden identique, Qwen1.5B→DeepSeek) | 0.000 | 0.000 | inconclusif |
| B (zero-pad cross-famille, gpt2→Qwen0.5B) | +0.535 | −0.115 | −0.21 |
