# 🎹 l'IA-instrument

> **Une thèse, une implémentation, trois expériences.** Un modèle de langage peut être *joué*
> comme un instrument de musique : on ne le programme pas par le prompt, on **tourne des
> boutons** dans ses représentations internes — des *steering vectors* dont l'effet est
> mesurable, monotone et transférable.

Ce dépôt implémente, de zéro et en stdlib-torch, deux familles de contrôle par
représentations internes — **RepE** (Representation Engineering, Zou et al. 2023) et
**CAA** (Contrastive Activation Addition, Arditi et al. 2024) — sous une API
« instrument de musique », puis les soumet à **trois expériences contrôlées sur 5 modèles
réels (4 familles)** : steering vs prompting, benchmark cross-modèles, transfert de
directions.

> **Honnêteté académique.** Ce n'est PAS une nouvelle méthode : c'est une implémentation
> propre et reproductible de méthodes publiées, + une **évaluation empirique systématique**
> (Exp. 1–3) qui, elle, est notre contribution. Les chiffres du README sont les sorties
> réelles des scripts dans `experiments/` — reproductibles commande pour commande.

---

## 1. Pourquoi

Aujourd'hui, contrôler un LLM = écrire un prompt (une *instruction* dont l'effet est flou,
non mesurable, sensible à la formulation). L'alternative démontrée ici : exposer une
**interface de réglage** sur les représentations internes du modèle.

| | Prompt « sois prudent » | Bouton « prudence » (steering) |
|---|---|---|
| Agit sur | le texte lu par le modèle (il *décide* d'obéir) | la représentation interne (déplacement direct) |
| Effet | flou, non monotone | mesurable, monotone (courbe amplitude → effet) |
| Coût | tokens d'instruction à chaque appel | un vecteur ajouté, quasi nul |
| Inverse | « ne sois pas positif » est faible | amplitude négative = contrôle inversé |
| Portabilité | lié à la formulation | knobs sauvegardés/rechargés ; transfert inter-familles **non démontré** (Exp. 3) |

La thèse n'est pas « le steering remplace le prompt » — c'est **un contrôle orthogonal**,
avec des limites claires (section 6).

## 2. API en 30 secondes

```python
from instrument import Instrument

inst = Instrument("openai-community/gpt2")          # charge un modèle local

# 1. Fabriquer un bouton par contraste (CAA)
inst.make_knob("positive", positive=POS, negative=NEG, layer=7)

# 2. Jouer une note : tourner le bouton, écouter la continuation
inst.play("The book was", max_new_tokens=24, knobs={"positive": 2.0})
#   → "a great book. It's a great book, and I'm very happy with it."

# 3. Polyphonie : plusieurs boutons à la fois
inst.play("Dear team,", max_new_tokens=24, knobs={"positive": 0.5, "formal": 1.0})

# 4. Lire un concept (RepE)
inst.read("I love this wonderful day.", name="positive")
#   → {"cosine": 0.21, "projection": 0.94}
```

CLI : `python -m instrument play|build|benchmark` (voir `--help`).

## 3. Installation

```bash
pip install -e .        # torch, transformers, numpy
python -m pytest tests/ -q   # 44 tests (tiny-random-gpt2, aucun poids réel requis)
```

## 4. Expériences — protocole commun

Les 3 expériences partagent un protocole strict, conçu pour être **honnête et
reproductible** :

- **Directions** : CAA (contraste brut) sur la couche à 60 % de profondeur
  (`layer_ratio=0.6`), amplitude 1 ≈ un contraste complet.
- **Génération** : greedy, seed fixe, `max_new_tokens` court.
- **Mesure** : scoreurs **lexicaux externes** (sentiment, formalité, refus),
  indépendants du modèle — on ne mesure jamais avec les logits du modèle steuré.
- **Mode natif** : les modèles *instruct* sont évalués **en chat-template**
  (leur mode d'usage réel) ; gpt2 en continuation base. Découverte méthodologique :
  évaluer un modèle instruct en continuation brute **annule** l'effet du steering
  (0.000), alors qu'en mode chat le même knob produit +0.548 (voir Exp. 1).

**Modèles** (≤1.5B, 4 familles, chargement local CPU) :

| Modèle | Famille | Couches | Hidden |
|---|---|---|---|
| gpt2 | GPT-2 | 12 | 768 |
| Qwen2.5-0.5B-Instruct | Qwen | 24 | 896 |
| Qwen2.5-1.5B-Instruct | Qwen | 28 | 1536 |
| TinyLlama-1.1B-Chat | Llama | 22 | 2048 |
| DeepSeek-R1-Distill-Qwen-1.5B | DeepSeek | 28 | 1536 |

### Exp. 1 — Steering vs prompting (contrôle apparié)

Même modèle, mêmes prompts d'évaluation, même scoreur. Conditions :
baseline (aucun contrôle) · steer +1/−1 (knob) · prompt +/− (instruction textuelle,
2 formulations). **Pouvoir de contrôle** = |écart max à la baseline| (steer à ±1 ;
prompt = meilleure des 2 formulations) :

| Modèle | Dimension | Steering | Prompt | Meilleur contrôle |
|---|---|---|---|---|
| gpt2 | sentiment | +0.271 | **+0.302** | prompt (marge fine) |
| gpt2 | ton | **+0.047** | +0.014 | steering |
| Qwen2.5-0.5B | sentiment | **+0.535** | +0.409 | **steering** |
| Qwen2.5-0.5B | ton | 0.006 | **+0.028** | prompt (les deux ≈ nuls) |

Résultats qualitatifs (détails par condition dans `reports/experiments/exp1_*.json`) :

1. **Le steering est plus cohérent.** Sur Qwen2.5-0.5B, le knob `+1` déplace **100 %** des
   16 prompts (test de signe p=1.00), contre 85–92 % pour le meilleur prompt. Sur gpt2,
   le steering négatif est lui aussi cohérent (p=0.00 : aucun prompt inversé), alors que
   le prompting positif ne l'est pas pleinement (p=0.67).
2. **Le steering gagne nettement sur le modèle instruct** (+0.535 vs +0.409), à coût
   marginal nul. Sur gpt2 les deux contrôles sont proches. **Le ton** reste peu
   contrôlable par l'une ou l'autre méthode sur ces petits modèles (limite honnête).

### Exp. 2 — Steerabilité cross-modèles

Balayage d'amplitude (5 niveaux) sur les 5 modèles, 3 dimensions
(sentiment / ton / refus-défensif). Métriques : corrélation échelle→score, pente,
baseline. Rapports JSON complets dans `reports/experiments/`.

Corrélation **échelle → score** (Pearson sur 5 amplitudes) ; pente = score / unité d'amplitude. « refus » en mode chat, usage défensif (direction qui augmente le refus).

| Modèle | Famille | Sentiment corr (pente) | Ton corr (pente) | Refus corr (baseline) |
|---|---|---|---|---|
| gpt2 | GPT-2 | **0.95** (+0.126) | 0.85 (+0.061) | — (pas de chat) |
| Qwen2.5-0.5B | Qwen | **0.92** (+0.205) | −0.38 | **0.97** (0.50) |
| Qwen2.5-1.5B | Qwen | **0.89** (+0.120) | **0.97** | **0.97** (0.58) |
| TinyLlama-1.1B-Chat | Llama | **0.89** (+0.057) | 0.35 | −0.35 (0.08) |
| DeepSeek-R1-Distill-1.5B | DeepSeek | 0.35 | 0.69 | 0.71 (0.00) |

Lecture honnête :

- **Sentiment : fort sur 4 familles sur 5** (corr 0.89–0.95). Le **DeepSeek** (modèle
  *reasoning*) répond peu au knob sentiment en mode chat (0.35) — un résultat
  authentique et intéressant.
- **Ton : dépend du modèle** — fort sur Qwen2.5-1.5B (0.97) et gpt2 (0.85), faible sur
  TinyLlama (0.35), **négatif** sur Qwen2.5-0.5B (−0.38 : le knob « formal » va à contre-
  sens de la métrique sur ce petit modèle).
- **Refus (défensif) : fort sur les Qwen** (0.97, montée de 50 % → 100 % de refus sur
  Qwen2.5-0.5B). **TinyLlama-Chat ne refuse pas** (baseline 8 %, corr négative) : le
  knob échoue — le modèle n'a pas de comportement de refus à amplifier.

### Exp. 3 — Transfert de directions

A. **Hidden identique** (même architecture) : knob construit sur Qwen2.5-1.5B →
appliqué tel quel à DeepSeek-R1-Distill-Qwen-1.5B (1536 dims).
B. **Hidden différent** : knob gpt2 (768) → projeté par zero-pad sur Qwen2.5-0.5B (896),
couche proportionnelle. Métrique : **coefficient de transfert** = effet transféré / effet natif.

| Design | Source → Cible | Hidden | Delta natif | Delta transféré | Coef. transfert |
|---|---|---:|---:|---:|---:|
| A — hidden identique | Qwen2.5-1.5B → DeepSeek-R1-Distill-1.5B | 1536→1536 | 0.000 | 0.000 | **inconclusif** |
| B — zero-pad cross-famille | gpt2 → Qwen2.5-0.5B | 768→896 | +0.535 | **−0.115** | **−0.21** |

Résultat honnête et important : **le transfert naïf de directions échoue.**

- **Design B** : le vecteur gpt2, projeté par zero-pad sur Qwen2.5-0.5B, produit un effet
  **négatif** (coefficient −0.21) — il *contre-steure* le sentiment au lieu de le pousser.
  Un steering vector n'est donc **pas portable entre familles** par simple projection.
- **Design A** : inconclusif — DeepSeek est peu steurable en sentiment nativement
  (corr 0.35, Exp. 2), donc il n'existe pas d'effet natif de référence à transférer.
  Transférer vers une cible non-steurable ne peut pas démontrer de transfert.

## 5. Résultats

**Synthèse honnête des trois expériences :**

1. **Le steering est réel, mesurable et souvent supérieur au prompt** — cohérent sur 100 %
   des prompts (Exp. 1), corrélation échelle→effet élevée sur le sentiment pour **4 familles
   sur 5** (Exp. 2), refus défensif puissant sur les Qwen (50 % → 100 %).
2. **Mais ce n'est pas magique ni universel** : le ton est peu contrôlable sur ces petits
   modèles, DeepSeek (reasoning) répond peu au steering sentiment, et **le transfert de
   directions entre familles échoue** par projection naïve (Exp. 3, coef −0.21).
3. **Ce dépôt est une implémentation reproductible + une évaluation empirique honnête** —
   pas une nouvelle méthode. La contribution est le protocole (scoreurs externes, mode
   natif, baselines vraies, rapports JSON horodatés) et les résultats, y compris négatifs.

## 6. Limites honnêtes

1. **Réglage brut** : un steering vector déplace une direction ; il peut dégrader la
   cohérence globale. Ce n'est pas de l'alignement magique.
2. **Modèles petits** : les benchmarks portent sur ≤1.5B sur CPU. Les corrélations sur
   des modèles plus gros (7B+) restent à mesurer.
3. **Adversaire motivé** : une injection de prompt agressive peut surpasser un vecteur.
4. **Il faut fabriquer les boutons** : un prompt s'écrit à l'instant T ; un knob se
   construit par contraste (coût de préparation).
5. **Exp. 1** : le prompting est sensible à la formulation — deux formulations testées,
   pas une recherche systématique d'instructions.
6. **Usage défensif** : le benchmark « refus » ne fournit que la direction qui
   **renforce** le refus. Aucune direction « comply ».

## 7. Reproduction

```bash
python experiments/run_all.py --exp1 --exp2 --exp3   # ~40 min CPU, 5 modèles
```

Chaque expérience écrit des rapports JSON horodatés dans `reports/experiments/`
+ un `SUMMARY.md`. Les poids (`models_cache/`) sont téléchargés en mode copie (pas de
symlink, Windows-friendly) et gitignorés.

## 8. Références

- Zou et al., *Representation Engineering*, arXiv:2310.01405 (ICLR 2024).
- Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction* (CAA), arXiv:2406.11717.
- Turner et al., *Activation Addition*, arXiv:2308.10248.
- Templeton et al. (Anthropic), *Scaling Monosemanticity* (2024).

## Licence

MIT — voir [LICENSE](LICENSE).
