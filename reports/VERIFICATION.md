# Preuves — chaîne de vérification

> Chaque chiffre du README et du paper provient d'un **rapport JSON horodaté**
> produit par un script committé dans `experiments/`. Rien n'est déclaré à la
> main : tout se rejoue avec une commande.

## Comment rejouer (la preuve complète)

```bash
pip install -e .
python experiments/run_all.py --exp1 --exp2 --exp3   # ~40 min CPU, 5 modèles
```

Chaque benchmark écrit son rapport dans `reports/experiments/`. La commande de
reproduction EST la preuve : rejouez-la, vous obtenez les mêmes chiffres
(seed fixe, greedy, scoreurs externes).

## Table des preuves par résultat

### Exp. 1 — Steering vs prompt

| Chiffre publié | Script | Rapport JSON source |
|---|---|---|
| gpt2 sentiment steer/prompt | `exp1_steering_vs_prompting.py` | `exp1_steering_vs_prompting_*.json` (dernier) |
| Qwen2.5-0.5B sentiment steer +0.535 / prompt +0.409 | idem | idem |
| Test de signe (p=1.00, 100 % prompts) | idem | champ `sign_test_vs_baseline` |
| Découverte mode-chat (0.000 vs +0.548) | idem + README | `per_prompt` / conditions |

### Exp. 2 — Cross-modèles

| Chiffre publié | Script | Rapport JSON source |
|---|---|---|
| Sentiment corr gpt2 0.95 | `exp2_cross_model.py` | `sentiment_*gpt2*.json` |
| Sentiment corr DeepSeek 0.35 | idem | `sentiment_*DeepSeek*.json` |
| Refus corr Qwen 0.97 | idem | `refusal_*Qwen*.json` |
| Refus corr TinyLlama −0.35 | idem | `refusal_*TinyLlama*.json` |

### Exp. 3 — Transfert

| Chiffre publié | Script | Rapport JSON source |
|---|---|---|
| Transféré gpt2→Qwen0.5B Δ −0.115, coef −0.21 | `exp3_transfer.py` | `exp3_transfer_*.json` (dernier) |
| Design A inconclusif (DeepSeek natif 0.000) | idem | idem |

## Ce qui a été corrigé pendant la production (transparence)

1. **Mode chat.** La 1re passe d'Exp. 1 évaluait les modèles instruct en
   continuation brute → 0.000 partout. Corrigé : mode natif (chat-template).
   Le chiffre « +0.548 en chat vs 0.000 en base » est un résultat méthodologique
   documenté, pas un bug.
2. **Baseline vraie.** `knobs=None` joue tous les knobs à amplitude 1.0.
   Les baselines sont donc passées en `knobs={}` (aucun knob). Les 1res versions
   d'Exp. 1 et Exp. 3 étaient contaminées ; les rapports finaux sont issus des
   runs corrigés. Aucun chiffre périmé ne reste dans le README/paper.
3. **Noms de fichiers.** Le slug des rapports sanitisé (chemins locaux Windows)
   — correctif dans `instrument/benchmarks.py`.

## Ce que les preuves ne couvrent pas (limites assumées)

- Modèles ≤1.5B uniquement (CPU) : les corrélations 7B+ restent à mesurer.
- 2 formulations de prompt testées, pas une recherche systématique.
- Scoreurs lexicaux = proxys (le refus est un regex).
