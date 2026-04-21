"""Thin OpenAI chat wrapper that records every call into the active Trace."""

from __future__ import annotations

import time

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import OPENAI_API_KEY
from .trace import AgentCall, current_trace

# Per-request timeout for OpenAI — guards against hung connections.
# With our retry (3 attempts), a truly dead call still fails after ~3×.
OPENAI_TIMEOUT_S = 60.0

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY missing; check .env")
        _client = OpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT_S)
    return _client


# Only retry on transient errors. AuthenticationError / PermissionDeniedError
# are permanent — retrying just delays the user seeing a useful error.
_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type(_RETRYABLE),
    reraise=True,
)
def _call_openai(model: str, messages: list[dict], temperature: float) -> tuple[str, int, int]:
    resp = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        timeout=OPENAI_TIMEOUT_S,
    )
    content = resp.choices[0].message.content or ""
    usage = resp.usage
    tokens_in = usage.prompt_tokens if usage else 0
    tokens_out = usage.completion_tokens if usage else 0
    return content, tokens_in, tokens_out


def chat(
    *,
    agent: str,
    step: str,
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
) -> str:
    """Run one chat completion and log it into the active Trace (if any)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    t0 = time.perf_counter()
    try:
        content, tokens_in, tokens_out = _call_openai(model, messages, temperature)
    except (AuthenticationError, PermissionDeniedError) as exc:
        raise RuntimeError(
            "OpenAI rejected the API key (AuthenticationError). "
            "Verify OPENAI_API_KEY is set correctly. On HuggingFace Spaces, "
            "set it under Settings → Variables and secrets. "
            f"Upstream detail: {exc}"
        ) from exc
    latency_ms = (time.perf_counter() - t0) * 1000

    trace = current_trace()
    if trace is not None:
        trace.add(
            AgentCall(
                agent=agent,
                step=step,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response=content,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
            )
        )
    return content
