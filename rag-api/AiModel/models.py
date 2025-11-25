import os
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import openai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional

from config import (
    USE_OPENAI,
    OPENAI_MODEL,
    USE_LOCAL_MISTRAL,
    MISTRAL_MODEL_PATH,
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
        # Optimize: Auto-detect device if not provided
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
        # Optimize: Auto-detect device if not provided
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


class LocalMistralLLM:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 256,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        # Optimize: Default to MPS if available
        self.device = device or get_optimal_device()
        self.max_new_tokens = max_new_tokens
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForCausalLM] = None

    def _ensure_model(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        print(f"Loading Local LLM ({self.model_name}) on {self.device}...")
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Optimize: Load in float16 for Mac/GPU speed & memory efficiency
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16, 
            device_map=self.device,
            low_cpu_mem_usage=True
        )
        # Note: device_map handles .to(device) automatically usually, 
        # but strictly ensuring it fits the pipeline:
        if self.device == "cuda" or self.device == "mps":
             self._model.to(self.device)

    def generate(self, prompt: str) -> dict:
        self._ensure_model()
        assert self._tokenizer is not None and self._model is not None

        # Optimize: Generalized instruct formatting (optional but recommended for Mistral)
        # If your prompt already has [INST] tags, this won't break it, 
        # but usually raw prompts need formatting.
        formatted_prompt = prompt
        if "[INST]" not in prompt:
            formatted_prompt = f"[INST] {prompt} [/INST]"

        inputs = self._tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            truncation=True
        ).to(self.device)

        # Optimize: torch.no_grad is crucial for inference speed
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.3,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated_text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Clean up the prompt from the response
        # (Handling the case where the model echoes the prompt)
        if "[/INST]" in generated_text:
            generated_text = generated_text.split("[/INST]")[-1].strip()
        elif generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        return {
            "text": generated_text,
            "usage": {
                "model": self.model_name,
                "source": "local",
                "device": self.device
            },
        }


class LLM:
    def __init__(self):
        self.use_openai = USE_OPENAI
        self.local_llm: Optional[LocalMistralLLM] = None

        if USE_OPENAI:
            openai.api_key = os.environ.get("OPENAI_API_KEY")

        if USE_LOCAL_MISTRAL:
            self.local_llm = LocalMistralLLM(
                model_name=MISTRAL_MODEL_PATH,
                max_new_tokens=MISTRAL_MAX_NEW_TOKENS,
                # Device is auto-detected inside LocalMistralLLM now
            )

    def generate(self, prompt: str, model: Optional[str] = None):
        if self.local_llm:
            try:
                return self.local_llm.generate(prompt)
            except Exception as exc:
                # fall back to OpenAI or at least return an explicit error message
                error_msg = f"Local Mistral generation failed: {exc}"
                print(error_msg)
                
                if not self.use_openai:
                     return {
                        "text": f"ERROR: {error_msg}",
                        "usage": {"error": str(exc)},
                    }
                # If OpenAI is enabled, we fall through to the logic below

        if not self.use_openai:
            return {
                "text": "OPENAI_DISABLED: " + prompt[:2000],
                "usage": {}
            }
        
        model = model or OPENAI_MODEL
        try:
            # simple ChatCompletion call; adapt to latest OpenAI client as needed
            # Note: Older OpenAI versions use ChatCompletion.create
            # Newer versions (1.0+) use client.chat.completions.create
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