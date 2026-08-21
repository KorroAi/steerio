"""Configuration partagée des expériences de l'steerio.

Méthodologie (protocole commun) :
  - Construction des directions : CAA (contraste brut) sur la couche 60 % de
    la profondeur (layer_ratio = 0.6).
  - Génération : greedy (do_sample=False, seed=0), max_new_tokens court.
  - Mesure : scoreurs lexicaux externes, indépendants du modèle.
  - Modes : sentiment/tone en continuation « base » (même format d'entrée pour
    tous les modèles, y compris instruct) ; refus en chat-template (nécessite
    un modèle instruct).

Les chiffres produits par ces scripts sont RÉELS : chaque rapport JSON est
écrit à partir des générations réellement exécutées sur les modèles.
"""
from __future__ import annotations

import os
from typing import Sequence

# Répertoire local des poids (download en mode copie : pas de symlinks,
# requis sur Windows sans Developer Mode).
MODELS_CACHE: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models_cache")


def resolve_model_path(model_id: str) -> str:
    """Retourne le chemin local du modèle s'il est présent, sinon l'id HF.

    Les modèles déjà dans le cache HF (gpt2, Qwen2.5-0.5B) se chargent par id ;
    les modèles téléchargés dans models_cache/ par chemin local.
    """
    local = os.path.join(os.path.abspath(MODELS_CACHE), model_id.split("/")[-1])
    if os.path.isdir(local):
        return local
    return model_id

# ------------------------------------------------------- modèles (familles)
# 5 modèles <= 1.5B, 4 familles : GPT-2, Qwen, Llama, DeepSeek.
MODELS: list[str] = [
    "openai-community/gpt2",                 # GPT-2  (base, 12 couches, 768)
    "Qwen/Qwen2.5-0.5B-Instruct",            # Qwen   (24 couches, 896)
    "Qwen/Qwen2.5-1.5B-Instruct",            # Qwen   (28 couches, 1536)
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",    # Llama  (22 couches, 2048)
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",  # DeepSeek (28 couches, 1536)
]

FAMILY: dict[str, str] = {
    "openai-community/gpt2": "GPT-2",
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "Llama",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "DeepSeek",
}

# Modèles avec chat-template (peuvent exécuter le benchmark de refus).
INSTRUCT_MODELS: list[str] = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
]

# ------------------------------------------------------- paramètres communs
SCALES: list[float] = [-2.0, -1.0, 0.0, 1.0, 2.0]
LAYER_RATIO: float = 0.6
SEED: int = 0
MAX_NEW_TOKENS: int = 32
DO_SAMPLE: bool = False

REPORT_DIR: str = "reports/experiments"

# ------------------------------------------------------- instructions (Exp 1)
# Deux formulations par polarité, pour documenter la sensibilité du prompting.
POS_INSTRUCTIONS: list[str] = [
    "Continue in a positive tone.",
    "Make your response positive and upbeat.",
]
NEG_INSTRUCTIONS: list[str] = [
    "Continue in a negative tone.",
    "Make your response gloomy and negative.",
]

# ------------------------------------------------------- sous-ensembles prompts
# Limite de prompts pour garder les temps de calcul CPU raisonnables tout en
# conservant une puissance statistique (>= 12 prompts par dimension).
def sent_prompts() -> list[str]:
    from steerio import prompts
    return prompts.SENTIMENT_EVAL_PROMPTS[:16]


def tone_prompts() -> list[str]:
    from steerio import prompts
    return prompts.TONE_EVAL_PROMPTS


def refusal_prompts() -> list[str]:
    from steerio import prompts
    return prompts.REFUSAL_EVAL_PROMPTS[:12]
