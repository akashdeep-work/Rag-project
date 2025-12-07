import os
import subprocess
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import openai
import torch
from typing import List, Optional
from mlx_lm import load, generate
import platform
import time
import sys

from config import (
    USE_OPENAI,
    OPENAI_MODEL,
    USE_LOCAL_MISTRAL, 
    MISTRAL_MODEL_PATH, 
    MISTRAL_MAX_NEW_TOKENS,
    LOCAL_MODEL_DIR
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
    def __init__(self, model_name: str = MISTRAL_MODEL_PATH, max_new_tokens: int = 512):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.local_path = LOCAL_MODEL_DIR
        self.device_type = self._detect_device()
        
        # Placeholders for the loaded backend
        self._model = None
        self._tokenizer = None
        
        print(f"🖥️  Hardware Detected: {self.device_type.upper()}")

    def _detect_device(self):
        """Detects if we should use Apple MLX or Standard PyTorch."""
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            try:
                import mlx.core
                return "mps" # Apple Silicon
            except ImportError:
                print("⚠️  Apple Silicon detected but 'mlx' not installed.")
                return "cpu"
        
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda" # NVIDIA GPU
            return "cpu" # Standard CPU
        except ImportError:
            return "cpu"

    def _ensure_model(self):
        if self._model is not None:
            return

        if self.device_type == "mps":
            self._load_mlx_backend()
        else:
            self._load_pytorch_backend()

    def _load_mlx_backend(self):
        # 1. Check for local converted model
        if not os.path.exists(f"{self.local_path}/config.json"):
            print(f"⚠️  Local MLX model not found at {self.local_path}")
            print(f"🔨 Quantizing '{self.model_name}' to 4-bit (MLX)...")
            
            command = [
                sys.executable, "-m", "mlx_lm.convert",
                "--hf-path", self.model_name,
                "--mlx-path", self.local_path,
                "-q",          
                "--q-bits", "4"
            ]
            try:
                subprocess.run(command, check=True)
                print("✅ Conversion complete! Saved locally.")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"MLX Quantization failed: {e}")

        # 2. Load
        print(f"🚀 Loading MLX model from: {self.local_path}")
        self._model, self._tokenizer = load(self.local_path)

    def _load_pytorch_backend(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"⚡ Initializing PyTorch backend on {self.device_type.upper()}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.device_type == "cuda":
            print("📦 Using NF4 4-bit Quantization (BitsAndBytes)...")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            print("⏳ Loading model to CPU (this might be slow)...")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True
            )
            print("🔨 Applying Dynamic Int8 Quantization for CPU...")
            self._model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )

    def generate(self, prompt: str) -> dict:
        self._ensure_model()
        start_time = time.time()
        
        # --- PATH A: MLX Generation ---
        if self.device_type == "mps":
            messages = [{"role": "user", "content": prompt}]
            
            # Use tokenizer chat template
            prompt_formatted = self._tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            text = generate(
                self._model, 
                self._tokenizer, 
                prompt=prompt_formatted, 
                max_tokens=self.max_new_tokens, 
                verbose=False
            )
            
            return {
                "text": text.strip(),
                "usage": {"backend": "mlx-4bit", "time": round(time.time() - start_time, 2)}
            }

        # --- PATH B: PyTorch Generation ---
        else:
            import torch
            messages = [{"role": "user", "content": prompt}]
            inputs = self._tokenizer.apply_chat_template(
                messages, 
                return_tensors="pt", 
                add_generation_prompt=True
            ).to(self._model.device)

            with torch.no_grad():
                output_ids = self._model.generate(
                    inputs, 
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7
                )

            input_len = inputs.shape[1]
            text = self._tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
            
            return {
                "text": text.strip(),
                "usage": {"backend": f"pytorch-{self.device_type}", "time": round(time.time() - start_time, 2)}
            }

class LLM:
    def __init__(self):
        self.use_openai = USE_OPENAI
        self.local_llm: Optional[LocalLLM] = None

        if USE_OPENAI:
            openai.api_key = os.environ.get("OPENAI_API_KEY")

        if USE_LOCAL_MISTRAL:
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