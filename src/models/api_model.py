"""
API model implementations for various providers.
"""
import time
import os
from typing import List, Optional
from abc import abstractmethod

from .base import BaseModel, ModelConfig, GenerationResult


class APIModel(BaseModel):
    """Base class for API-based models."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = None

    def load(self) -> None:
        """Initialize API client."""
        self._loaded = True

    def unload(self) -> None:
        """Cleanup API client."""
        self.client = None
        self._loaded = False

    @abstractmethod
    def _call_api(self, prompt: str) -> GenerationResult:
        """Make API call. Must be implemented by subclasses."""
        pass

    def generate(self, prompt: str) -> GenerationResult:
        """Generate response via API."""
        if not self._loaded:
            self.load()
        return self._call_api(prompt)

    def batch_generate(self, prompts: List[str]) -> List[GenerationResult]:
        """Generate responses for multiple prompts sequentially."""
        return [self.generate(p) for p in prompts]


class OpenAIModel(APIModel):
    """OpenAI API model (also works for compatible APIs)."""

    def load(self) -> None:
        """Initialize OpenAI client."""
        from openai import OpenAI

        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        api_base = self.config.api_base or os.getenv("OPENAI_API_BASE")

        if api_base:
            self.client = OpenAI(api_key=api_key, base_url=api_base)
        else:
            self.client = OpenAI(api_key=api_key)

        self._loaded = True

    def _call_api(self, prompt: str) -> GenerationResult:
        """Call OpenAI-compatible API."""
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )

            content = response.choices[0].message.content or ""
            latency = time.time() - start_time
            tokens_used = response.usage.total_tokens if response.usage else 0

            return GenerationResult(
                response=content.strip(),
                raw_response=response,
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


class AnthropicModel(APIModel):
    """Anthropic Claude API model."""

    def load(self) -> None:
        """Initialize Anthropic client."""
        import anthropic

        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)
        self._loaded = True

    def _call_api(self, prompt: str) -> GenerationResult:
        """Call Anthropic API."""
        start_time = time.time()

        try:
            response = self.client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_new_tokens,
                messages=[{"role": "user", "content": prompt}]
            )

            content = response.content[0].text if response.content else ""
            latency = time.time() - start_time
            tokens_used = response.usage.input_tokens + response.usage.output_tokens

            return GenerationResult(
                response=content.strip(),
                raw_response=response,
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


class ZhipuModel(APIModel):
    """Zhipu AI (GLM) API model."""

    def load(self) -> None:
        """Initialize Zhipu client."""
        from zhipuai import ZhipuAI

        api_key = self.config.api_key or os.getenv("ZHIPUAI_API_KEY")
        self.client = ZhipuAI(api_key=api_key)
        self._loaded = True

    def _call_api(self, prompt: str) -> GenerationResult:
        """Call Zhipu API."""
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )

            content = response.choices[0].message.content or ""
            latency = time.time() - start_time
            tokens_used = response.usage.total_tokens if response.usage else 0

            return GenerationResult(
                response=content.strip(),
                raw_response=response,
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


class QwenAPIModel(APIModel):
    """Alibaba Qwen API model via DashScope."""

    def load(self) -> None:
        """Initialize DashScope."""
        import dashscope

        api_key = self.config.api_key or os.getenv("DASHSCOPE_API_KEY")
        dashscope.api_key = api_key
        self._loaded = True

    def _call_api(self, prompt: str) -> GenerationResult:
        """Call DashScope API."""
        from dashscope import Generation

        start_time = time.time()

        try:
            response = Generation.call(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                result_format='message'
            )

            if response.status_code == 200:
                content = response.output.choices[0].message.content
                tokens_used = response.usage.total_tokens if response.usage else 0

                return GenerationResult(
                    response=content.strip(),
                    raw_response=response,
                    latency=time.time() - start_time,
                    tokens_used=tokens_used,
                    success=True
                )
            else:
                return GenerationResult(
                    response="",
                    latency=time.time() - start_time,
                    success=False,
                    error_message=f"API error: {response.code} - {response.message}"
                )

        except Exception as e:
            return GenerationResult(
                response="",
                latency=time.time() - start_time,
                success=False,
                error_message=str(e)
            )


class DeepSeekModel(APIModel):
    """DeepSeek API model (OpenAI compatible)."""

    def load(self) -> None:
        """Initialize DeepSeek client."""
        from openai import OpenAI

        api_key = self.config.api_key or os.getenv("DEEPSEEK_API_KEY")
        api_base = self.config.api_base or "https://api.deepseek.com"

        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self._loaded = True

    def _call_api(self, prompt: str) -> GenerationResult:
        """Call DeepSeek API."""
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name or "deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )

            content = response.choices[0].message.content or ""
            latency = time.time() - start_time
            tokens_used = response.usage.total_tokens if response.usage else 0

            return GenerationResult(
                response=content.strip(),
                raw_response=response,
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
