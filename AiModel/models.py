from __future__ import annotations

import os
import time
from typing import Iterable, List, Optional

import numpy as np
import openai
import torch
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

from config import (
    LOCAL_MODEL_DIR,
    MAX_NEW_TOKENS,
    OPENAI_MODEL,
    USE_LOCAL_MISTRAL,
    USE_OPENAI,
)


def get_optimal_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str | None = None):
        self.device = device or get_optimal_device()
        self.model = SentenceTransformer(model_name, device=self.device)
        sample = self.model.encode(["hi"], convert_to_numpy=True, normalize_embeddings=True)
        self.dim = sample.shape[1]

    def embed_texts(self, texts: List[str], batch_size: int = 64):
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            emb = self.model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
            out.append(emb)
        return np.vstack(out)

    def embed(self, text: str):
        return self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]


class LocalLLM:
    def __init__(self, max_new_tokens: int = MAX_NEW_TOKENS):
        self.max_new_tokens = max_new_tokens
        self.local_path = LOCAL_MODEL_DIR
        self._model: Llama | None = None
        self.repo_id = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
        self.filename = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

    def _ensure_model(self):
        if self._model is not None:
            return

        os.makedirs(self.local_path, exist_ok=True)
        model_path = os.path.join(self.local_path, self.filename)
        if not os.path.exists(model_path):
            hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename,
                local_dir=self.local_path,
                local_dir_use_symlinks=False,
            )

        self._model = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=2,
            n_gpu_layers=0,
            verbose=False,
        )

    def _messages(self, prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

    def generate(self, prompt: str) -> dict:
        self._ensure_model()
        start_time = time.time()
        try:
            output = self._model.create_chat_completion(
                messages=self._messages(prompt),
                max_tokens=self.max_new_tokens,
                temperature=0.7,
                stream=False,
            )
            text = output["choices"][0]["message"]["content"]
            return {
                "text": text.strip(),
                "usage": {"backend": "qwen-1.5b-gguf", "time": round(time.time() - start_time, 2)},
            }
        except Exception as e:
            return {"text": f"Error: {str(e)}", "usage": {"error": str(e)}}

    def generate_stream(self, prompt: str) -> Iterable[str]:
        self._ensure_model()
        for part in self._model.create_chat_completion(
            messages=self._messages(prompt),
            max_tokens=self.max_new_tokens,
            temperature=0.7,
            stream=True,
        ):
            delta = part.get("choices", [{}])[0].get("delta", {})
            token = delta.get("content", "")
            if token:
                yield token


class LLM:
    def __init__(self):
        self.use_openai = USE_OPENAI
        self.local_llm: Optional[LocalLLM] = None
        if self.use_openai:
            openai.api_key = os.environ.get("OPENAI_API_KEY")
        if USE_LOCAL_MISTRAL:
            self.local_llm = LocalLLM(max_new_tokens=MAX_NEW_TOKENS)

    def generate(self, prompt: str, model: Optional[str] = None):
        if self.local_llm:
            try:
                return self.local_llm.generate(prompt)
            except Exception as exc:
                if not self.use_openai:
                    return {"text": f"ERROR: Local LLM generation failed: {exc}", "usage": {"error": str(exc)}}

        if not self.use_openai:
            return {"text": "OPENAI_DISABLED: " + prompt[:2000], "usage": {}}

        model = model or OPENAI_MODEL
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.0,
            )
            txt = resp["choices"][0]["message"]["content"]
            return {"text": txt, "raw": resp}
        except Exception as e:
            return {"text": f"OpenAI Error: {str(e)}", "usage": {}}

    def generate_stream(self, prompt: str, model: Optional[str] = None) -> Iterable[str]:
        if self.local_llm:
            try:
                yield from self.local_llm.generate_stream(prompt)
                return
            except Exception as exc:
                if not self.use_openai:
                    yield f"ERROR: Local LLM generation failed: {exc}"
                    return

        if not self.use_openai:
            yield "OPENAI_DISABLED: " + prompt[:2000]
            return

        response = self.generate(prompt, model=model)
        text = response.get("text", "") if isinstance(response, dict) else str(response)
        for token in text.split(" "):
            if token:
                yield token + " "
