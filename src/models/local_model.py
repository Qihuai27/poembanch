"""
Local model implementation using HuggingFace Transformers.
"""
import time
from typing import List, Optional

from .base import BaseModel, ModelConfig, GenerationResult


class LocalModel(BaseModel):
    """Local model using HuggingFace Transformers."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.device = None

    def load(self) -> None:
        """Load the model and tokenizer."""
        if self._loaded:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Determine device
        if self.config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device

        # Determine dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.float16)

        print(f"Loading model from: {self.config.model_path}")
        print(f"Device: {self.device}, Dtype: {self.config.torch_dtype}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            padding_side='left'
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
            self.model = self.model.to(self.device)

        self.model.eval()
        self._loaded = True
        print("Model loaded successfully!")

    def _build_prompt(self, prompt: str) -> str:
        """Build prompt with chat template if available."""
        messages = [{"role": "user", "content": prompt}]

        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        return prompt

    def generate(self, prompt: str) -> GenerationResult:
        """Generate response for a single prompt."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch

        start_time = time.time()

        try:
            # Build prompt
            full_prompt = self._build_prompt(prompt)

            # Tokenize
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            input_length = inputs.input_ids.shape[-1]

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.temperature > 0,
                    temperature=max(self.config.temperature, 0.01),
                    top_p=self.config.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only new tokens
            response = self.tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            ).strip()

            latency = time.time() - start_time
            tokens_used = outputs.shape[-1] - input_length

            return GenerationResult(
                response=response,
                raw_response=outputs,
                latency=latency,
                tokens_used=tokens_used,
                success=True
            )

        except Exception as e:
            return GenerationResult(
                response="",
                latency=time.time() - start_time,
                success=False,
                error_message=str(e)
            )

    def batch_generate(self, prompts: List[str]) -> List[GenerationResult]:
        """Generate responses for multiple prompts."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch

        results = []
        start_time = time.time()

        try:
            # Build prompts
            full_prompts = [self._build_prompt(p) for p in prompts]

            # Tokenize with padding
            inputs = self.tokenizer(
                full_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.temperature > 0,
                    temperature=max(self.config.temperature, 0.01),
                    top_p=self.config.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            total_latency = time.time() - start_time
            per_sample_latency = total_latency / len(prompts)

            # Decode each response
            for i, output in enumerate(outputs):
                input_length = (inputs.attention_mask[i] == 1).sum().item()
                response = self.tokenizer.decode(
                    output[input_length:],
                    skip_special_tokens=True
                ).strip()

                results.append(GenerationResult(
                    response=response,
                    latency=per_sample_latency,
                    tokens_used=len(output) - input_length,
                    success=True
                ))

        except Exception as e:
            # Return error results for all prompts
            for _ in prompts:
                results.append(GenerationResult(
                    response="",
                    latency=0,
                    success=False,
                    error_message=str(e)
                ))

        return results

    def unload(self) -> None:
        """Unload model to free GPU memory."""
        import gc

        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        gc.collect()
        self._loaded = False
        print("Model unloaded.")
