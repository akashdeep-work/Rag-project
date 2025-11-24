import os
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import openai
from typing import List, Optional

from config import USE_OPENAI, OPENAI_MODEL

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2", device="cpu"):
        self.model = SentenceTransformer(model_name, device=device)
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
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, candidates: List[str], batch_size: int = 32):
        scores = []
        for i in range(0, len(candidates), batch_size):
            pairs = [[query, c] for c in candidates[i:i+batch_size]]
            scores_batch = self.model.predict(pairs)
            scores.extend(scores_batch.tolist())
        return scores


class LLM:
    def __init__(self):
        if USE_OPENAI:
            openai.api_key = os.environ.get("OPENAI_API_KEY")

    def generate(self, prompt: str, model: Optional[str] = None):
        if not USE_OPENAI:
            return {
                "text": "OPENAI_DISABLED: " + prompt[:2000],
                "usage": {}
            }
        model = model or OPENAI_MODEL
        # simple ChatCompletion call; adapt to latest OpenAI client as needed
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            max_tokens=512,
            temperature=0.0
        )
        txt = resp["choices"][0]["message"]["content"]
        return {"text": txt, "raw": resp}
