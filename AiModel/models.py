import os
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import openai
import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional
from mlx_lm import load, generate

from config import (
    USE_OPENAI,
    OPENAI_MODEL,
    USE_LOCAL_MISTRAL, # You can keep this var name or rename it in config
    MISTRAL_MODEL_PATH, # Update this in config.py to: "microsoft/Phi-3.5-mini-instruct"
    MISTRAL_MAX_NEW_TOKENS,
)

# --- Helper for Device Detection ---
def get_optimal_device() -> str:
    """
    Auto-detects the best available accelerator.
    Returns: 'mps' for Mac, 'cuda' for Nvidia, 'cpu' otherwise.
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2", device=None):
        self.device = device or get_optimal_device()
        print(f"Loading Embedder on: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Warmup
        sample = self.model.encode(["hi"], convert_to_numpy=True, normalize_embeddings=True)
        self.dim = sample.shape[1]

    def embed_texts(self, texts: List[str], batch_size: int = 64):
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            emb = self.model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
            out.append(emb)
        return np.vstack(out)

    def embed(self, text: str):
        return self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]


class ReRanker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device=None):
        self.device = device or get_optimal_device()
        print(f"Loading ReRanker on: {self.device}")
        self.model = CrossEncoder(model_name, device=self.device)

    def rerank(self, query: str, candidates: List[str], batch_size: int = 32):
        scores = []
        for i in range(0, len(candidates), batch_size):
            pairs = [[query, c] for c in candidates[i:i+batch_size]]
            scores_batch = self.model.predict(pairs)
            scores.extend(scores_batch.tolist())
        return scores


class LocalLLM:
    def __init__(self, model_name: str, max_new_tokens: int = 512):
        self.model_name = model_name # Use "mlx-community/Phi-3.5-mini-instruct-4bit"
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def _ensure_model(self):
        if self._model: return
        print(f"Loading {self.model_name} via MLX...")
        # MLX handles loading seamlessly on Apple Silicon
        self._model, self._tokenizer = load(self.model_name)

    def generate(self, prompt: str) -> dict:
        self._ensure_model()
        
        # MLX includes built-in chat templating
        formatted_prompt = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )

        text = generate(
            self._model, 
            self._tokenizer, 
            prompt=formatted_prompt, 
            max_tokens=self.max_new_tokens,
            verbose=False
        )

        return {"text": text.strip(), "usage": {"source": "mlx"}}

class LLM:
    def __init__(self):
        self.use_openai = USE_OPENAI
        self.local_llm: Optional[LocalLLM] = None

        if USE_OPENAI:
            openai.api_key = os.environ.get("OPENAI_API_KEY")

        if USE_LOCAL_MISTRAL: # You can keep the flag name or change it
            # We use the new generic LocalLLM class
            self.local_llm = LocalLLM(
                model_name=MISTRAL_MODEL_PATH, 
                max_new_tokens=MISTRAL_MAX_NEW_TOKENS,
            )

    def generate(self, prompt: str, model: Optional[str] = None):
        if self.local_llm:
            try:
                return self.local_llm.generate(prompt)
            except Exception as exc:
                error_msg = f"Local LLM generation failed: {exc}"
                print(error_msg)
                
                if not self.use_openai:
                     return {
                        "text": f"ERROR: {error_msg}",
                        "usage": {"error": str(exc)},
                    }

        if not self.use_openai:
            return {
                "text": "OPENAI_DISABLED: " + prompt[:2000],
                "usage": {}
            }
        
        model = model or OPENAI_MODEL
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role":"user","content":prompt}],
                max_tokens=512,
                temperature=0.0
            )
            txt = resp["choices"][0]["message"]["content"]
            return {"text": txt, "raw": resp}
        except Exception as e:
            return {"text": f"OpenAI Error: {str(e)}", "usage": {}}