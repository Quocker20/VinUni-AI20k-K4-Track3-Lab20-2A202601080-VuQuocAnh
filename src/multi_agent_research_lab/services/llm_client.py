import logging
from dataclasses import dataclass
from typing import Any

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)

# GPT-4o-mini pricing: $0.15 / 1M prompt tokens, $0.60 / 1M completion tokens
COST_PER_INPUT_TOKEN_USD = 0.15 / 1_000_000
COST_PER_OUTPUT_TOKEN_USD = 0.60 / 1_000_000


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client with OpenRouter/OpenAI support."""

    def __init__(self, model: str | None = None) -> None:
        self.settings = get_settings()
        self.model = model or self.settings.effective_model
        self.api_key = self.settings.effective_api_key
        self.base_url = self.settings.effective_base_url
        self.timeout = float(self.settings.timeout_seconds)

        self._client: openai.OpenAI | None = None
        if self.api_key:
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(
            (
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.RateLimitError,
                openai.InternalServerError,
            )
        ),
        reraise=True,
    )
    def _call_api(self, messages: list[dict[str, str]], temperature: float = 0.2) -> Any:
        if not self._client:
            raise openai.OpenAIError("No API key configured.")
        return self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
        )

    def complete(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.2
    ) -> LLMResponse:
        """Return a model completion with token usage and cost estimation.

        If API key is not configured, provides a structured offline fallback.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if not self.api_key or not self._client:
            logger.info("OPENROUTER_API_KEY is not set. Generating offline mock completion.")
            mock_content = (
                "[Offline Mode / API key pending]\n\n"
                f"Summary for: '{user_prompt[:120]}...'\n\n"
                "1. Core Insights: GraphRAG and advanced agentic architectures integrate "
                "structured knowledge graphs with dense vector retrieval to capture "
                "multi-hop relationships across large corpora.\n"
                "2. Trade-offs: Single-agent systems offer lower latency and simpler debugging, "
                "whereas multi-agent architectures enhance context isolation and verification "
                "at the cost of higher token expenditure."
            )
            mock_in = len(system_prompt.split()) + len(user_prompt.split())
            mock_out = len(mock_content.split())
            mock_cost = (mock_in * COST_PER_INPUT_TOKEN_USD) + (
                mock_out * COST_PER_OUTPUT_TOKEN_USD
            )
            return LLMResponse(
                content=mock_content,
                input_tokens=mock_in,
                output_tokens=mock_out,
                cost_usd=mock_cost,
            )

        try:
            response = self._call_api(messages, temperature=temperature)
            content = response.choices[0].message.content or ""
            in_tokens: int | None = (
                int(response.usage.prompt_tokens) if response.usage else None
            )
            out_tokens: int | None = (
                int(response.usage.completion_tokens) if response.usage else None
            )
            cost_val: float | None = None
            if in_tokens is not None and out_tokens is not None:
                cost_val = (in_tokens * COST_PER_INPUT_TOKEN_USD) + (
                    out_tokens * COST_PER_OUTPUT_TOKEN_USD
                )
            return LLMResponse(
                content=content,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                cost_usd=cost_val,
            )
        except Exception as exc:
            logger.warning("LLM API call failed: %s. Falling back to offline response.", exc)
            fallback_content = (
                f"[API Call Failed: {exc}]\n\n"
                f"Fallback summary for: '{user_prompt[:120]}...'\n\n"
                "Analysis shows key trade-offs in agent orchestration, memory management, "
                "and evidence grounding."
            )
            return LLMResponse(
                content=fallback_content,
                input_tokens=len(user_prompt.split()),
                output_tokens=len(fallback_content.split()),
                cost_usd=0.0,
            )
