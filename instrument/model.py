"""Wrapper modèle : chargement, découverte des couches, extraction d'activations,
génération avec steering, lecture RepE des représentations.

Compatible GPT2 (`transformer.h`), Llama/Qwen (`model.layers`) et la plupart
des causal LMs de transformers 5.x.
"""
from __future__ import annotations

import random
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

_TRANSFORMERS_V5 = int(transformers.__version__.split(".")[0]) >= 5


def set_seed(seed: int) -> None:
    """Fige l'aléatoire (reproductibilité des benchmarks)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def discover_layers(model) -> torch.nn.ModuleList:
    """Localise la ModuleList des blocs transformateur du modèle causal.

    Essaie d'abord les chemins canoniques, puis cherche dans les enfants.
    """
    for path in (
        "transformer.h",
        "model.layers",
        "gpt_neox.layers",
        "decoder.layers",
        "language_model.model.layers",
    ):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, torch.nn.ModuleList) and len(obj) > 0:
                return obj
        except AttributeError:
            continue

    # Fallback : chercher une ModuleList de blocs (pas d'embeddings)
    def looks_like_blocks(ml: torch.nn.ModuleList) -> bool:
        return len(ml) > 0 and "Embedding" not in type(ml[0]).__name__

    for sub in model.children():
        if isinstance(sub, torch.nn.ModuleList) and looks_like_blocks(sub):
            return sub
        for subsub in sub.children():
            if isinstance(subsub, torch.nn.ModuleList) and looks_like_blocks(subsub):
                return subsub
    raise ValueError("Impossible de localiser les couches transformateur du modèle")


class InstrumentModel:
    """Un modèle causal + tokenizer, avec les primitives RepE/CAA."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
        cache_dir: str | None = None,
        trust_remote_code: bool = False,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # génération batch decoder-only : padding à gauche (recommandé transformers)
        self.tokenizer.padding_side = "left"
        # transformers 5 a renommé `torch_dtype` en `dtype` (l'ancien nom est déprécié)
        dtype_kwarg = "dtype" if _TRANSFORMERS_V5 else "torch_dtype"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **{dtype_kwarg: torch_dtype},
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device != "cpu":
            self.model = self.model.to(self.device)
        self.layers = discover_layers(self.model)
        self.n_layers = len(self.layers)
        self.hidden_size = int(self.model.config.hidden_size)
        # vocab_size : getattr avec default est évalué paresseusement (get_output_embeddings
        # peut être None sur certains modèles)
        vocab = getattr(self.model.config, "vocab_size", None)
        if vocab is None:
            emb = self.model.get_output_embeddings()
            vocab = emb.weight.shape[0] if emb is not None else len(self.tokenizer)
        self.vocab_size = int(vocab)

    # ---------------------------------------------------------------- couches
    def pick_default_layer(self, ratio: float = 0.6) -> int:
        """Couche de travail par défaut (60 % de la profondeur)."""
        return max(0, min(self.n_layers - 1, int(self.n_layers * ratio)))

    # ------------------------------------------------------ activations (RepE)
    def _apply_chat(self, texts: list[str]) -> list[str]:
        """Formate les textes en tour utilisateur si le tokenizer a un chat_template."""
        if not getattr(self.tokenizer, "chat_template", None):
            return texts
        return [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": t}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for t in texts
        ]

    @torch.no_grad()
    def extract_activations(
        self,
        texts: str | Sequence[str],
        layer: int | None = None,
        position: str = "last",
        batch_size: int = 8,
        apply_chat: bool = False,
    ) -> torch.Tensor:
        """Activations du flux résiduel à `layer` pour chaque texte.

        position : "last" = dernier token réel (masque attention) ;
                   "mean" = moyenne sur les tokens réels.
        apply_chat : formate les textes en tour utilisateur (modèles instruct).
        Retourne [n_texts, hidden] (ou [hidden] si un seul texte).
        """
        layer = self.pick_default_layer() if layer is None else layer
        if not (0 <= layer < self.n_layers):
            raise ValueError(f"couche {layer} hors bornes [0, {self.n_layers})")
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        if apply_chat:
            texts = self._apply_chat(texts)

        store: dict[str, torch.Tensor] = {}

        def hook_fn(module, args, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            store["h"] = h.detach()

        handle = self.layers[layer].register_forward_hook(hook_fn)
        gathered: list[torch.Tensor] = []
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                self.model(**enc)
                h = store.pop("h")
                mask = enc["attention_mask"]
                if position == "last":
                    idx = mask.sum(dim=1) - 1
                    per = h[torch.arange(h.size(0), device=h.device), idx]
                elif position == "mean":
                    m = mask.unsqueeze(-1)
                    per = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1)
                else:
                    raise ValueError(f"position inconnue : {position!r}")
                gathered.append(per.float().cpu())
        finally:
            handle.remove()
        acts = torch.cat(gathered, dim=0)
        return acts[0] if single else acts

    # ------------------------------------------------------------ génération
    @torch.no_grad()
    def generate(
        self,
        prompts: str | Sequence[str],
        max_new_tokens: int = 32,
        steering: Mapping[int, object] | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float = 1.0,
        seed: int | None = None,
        eos_token_id: int | None = None,
        apply_chat: bool = False,
        **kwargs,
    ) -> str | list[str]:
        """Génère des continuations, éventuellement avec des filtres de steering.

        steering : dict {couche: SteeringFilter} (voir instrument/steering.py).
        apply_chat : formate les prompts en tour utilisateur (modèles instruct).
        Retourne une str si un seul prompt, sinon une liste.
        """
        if seed is not None:
            set_seed(seed)
        single = isinstance(prompts, str)
        texts = [prompts] if single else list(prompts)
        if apply_chat:
            texts = self._apply_chat(texts)
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
            temperature=float(temperature),
            repetition_penalty=float(repetition_penalty),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=(
                eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id
            ),
        )
        if top_p is not None:
            gen_kwargs["top_p"] = float(top_p)
        if top_k is not None:
            gen_kwargs["top_k"] = int(top_k)
        gen_kwargs.update(kwargs)

        handles = []
        if steering:
            from .steering import attach_steering

            handles = attach_steering(self.model, self.layers, steering)
        try:
            out = self.model.generate(**enc, **gen_kwargs)
        finally:
            for handle in handles:
                handle.remove()

        generated = out[:, prompt_len:]
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return decoded[0] if single else decoded

    # ------------------------------------------------- lecture / sondes (RepE)
    @torch.no_grad()
    def forward_logits(self, text: str) -> torch.Tensor:
        """Logits du dernier token de `text` (forme [vocab])."""
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        return self.model(**enc).logits[0, -1].float().cpu()

    @torch.no_grad()
    def read(self, text: str, direction: torch.Tensor, layer: int | None = None) -> dict:
        """Lecture RepE : projection/cosinus de l'activation de `text` sur la direction."""
        h = self.extract_activations(text, layer=layer)
        v = direction.float()
        cos = float(F.cosine_similarity(h, v, dim=-1))
        proj = float(h @ v / (v.norm() + 1e-8))
        return {"cosine": cos, "projection": proj}

    @torch.no_grad()
    def word_logprob(self, text: str, words: Sequence[str]) -> dict[str, float]:
        """Log-probabilité (normalisée) des mots donnés comme token suivant.

        Pour un mot multi-tokens, on retient la probabilité de son premier
        token (approximation documentée) : la valeur est exacte pour les mots
        mono-tokens, indicatrice sinon.
        """
        logits = self.forward_logits(text)
        logp = F.log_softmax(logits, dim=-1)
        result: dict[str, float] = {}
        for word in words:
            ids = self.tokenizer.encode(" " + word, add_special_tokens=False)
            if not ids:
                continue
            result[word] = float(logp[ids[0]].item())
        return result

    def generate_with_probs(
        self,
        text: str,
        words: Sequence[str],
        steering: Mapping[int, object] | None = None,
        layer: int | None = None,
    ) -> dict[str, float]:
        """Log-probabilité de mots sous l'effet d'un steering (mesure directe du contrôle)."""
        if steering:
            handles = []
            from .steering import attach_steering

            handles = attach_steering(self.model, self.layers, steering)
            try:
                return self.word_logprob(text, words)
            finally:
                for handle in handles:
                    handle.remove()
        return self.word_logprob(text, words)
