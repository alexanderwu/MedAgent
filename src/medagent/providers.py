"""Provider seam: everything Gemini- or Ollama-specific lives here.

The agent loop in agent.py is provider-agnostic — it holds a transcript of
provider-native messages (`Any`) and drives it through the Provider protocol.
Tool-result payloads keep the `{"result": ...}` / `{"error": ...}` convention;
each provider serializes them its own way. Neither API supplies usable call
ids, so the loop's synthetic ids never reach the provider — only function
names do.
"""

import json
from dataclasses import dataclass
from typing import Any, Protocol, cast

import httpx
import ollama
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from medagent import tools
from medagent.config import LOCAL_MODEL, LOCAL_NUM_CTX, MODEL


@dataclass
class FunctionCall:
    name: str
    args: dict[str, Any]


@dataclass
class StepResult:
    message: Any  # provider-native assistant message (appended to transcript)
    text: str
    function_calls: list[FunctionCall]
    truncated: bool  # hit max-tokens finish reason


class ProviderError(Exception):
    """Provider failure with a user-facing message; the loop maps it to _fail."""


class Provider(Protocol):
    name: str
    model: str

    def generate(
        self, system: str, contents: list[Any], use_tools: bool
    ) -> StepResult: ...

    def user_text(self, text: str) -> Any: ...

    def tool_results(self, results: list[tuple[str, dict[str, Any]]]) -> list[Any]: ...


class GeminiProvider:
    name = "gemini"

    def __init__(self) -> None:
        self.model = MODEL
        self.client = genai.Client()

    def generate(self, system: str, contents: list[Any], use_tools: bool) -> StepResult:
        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            tools=(
                [
                    genai_types.Tool(
                        function_declarations=cast(Any, tools.FUNCTION_DECLARATIONS)
                    )
                ]
                if use_tools
                else None
            ),
            automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )
        try:
            response = self.client.models.generate_content(
                model=self.model, contents=cast(Any, contents), config=config
            )
        except genai_errors.APIError as e:
            raise ProviderError(f"Gemini API error ({e.code}): {e.message}") from e
        except httpx.RequestError as e:
            raise ProviderError(
                "Could not reach the Gemini API — check your network."
            ) from e

        if not response.candidates:
            raise ProviderError("The Gemini API returned an empty response.")
        candidate = response.candidates[0]
        content = candidate.content
        if content is None:
            raise ProviderError("The Gemini API returned an empty response.")

        calls = [
            FunctionCall(call.name or "", dict(call.args or {}))
            for call in (response.function_calls or [])
        ]
        text = "".join(p.text for p in content.parts or [] if p.text)
        return StepResult(
            message=content,
            text=text,
            function_calls=calls,
            truncated=candidate.finish_reason == genai_types.FinishReason.MAX_TOKENS,
        )

    def user_text(self, text: str) -> Any:
        return genai_types.Content(role="user", parts=[genai_types.Part(text=text)])

    def tool_results(self, results: list[tuple[str, dict[str, Any]]]) -> list[Any]:
        # Gemini expects all function responses for a turn in ONE user message.
        parts = [
            genai_types.Part.from_function_response(name=name, response=payload)
            for name, payload in results
        ]
        return [genai_types.Content(role="user", parts=parts)]


# FUNCTION_DECLARATIONS is already JSON-schema-shaped; Ollama just wants the
# OpenAI-style {"type": "function", "function": {...}} wrapper around each.
_OLLAMA_TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": decl} for decl in tools.FUNCTION_DECLARATIONS
]


class OllamaProvider:
    name = "ollama"

    def __init__(self) -> None:
        self.model = LOCAL_MODEL
        self.client = ollama.Client()  # respects OLLAMA_HOST

    def generate(self, system: str, contents: list[Any], use_tools: bool) -> StepResult:
        messages = [{"role": "system", "content": system}, *contents]
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                tools=_OLLAMA_TOOLS if use_tools else None,
                # Ollama's 4096-token default is too small for schema dumps
                # plus CSV results.
                options={"num_ctx": LOCAL_NUM_CTX},
            )
        except ollama.ResponseError as e:
            hint = (
                f" — run `ollama pull {self.model}` first"
                if e.status_code == 404
                else ""
            )
            raise ProviderError(f"Ollama error: {e.error}{hint}") from e
        except (httpx.RequestError, ConnectionError) as e:
            raise ProviderError(
                "Could not reach Ollama — is `ollama serve` running? "
                "(set OLLAMA_HOST if it is not on localhost:11434)"
            ) from e

        msg = response.message
        # gpt-oss thinking output is deliberately dropped: msg.thinking is
        # neither displayed nor fed back into the transcript.
        message: dict[str, Any] = {"role": msg.role, "content": msg.content or ""}
        if msg.tool_calls:
            message["tool_calls"] = [call.model_dump() for call in msg.tool_calls]
        calls = [
            FunctionCall(call.function.name, dict(call.function.arguments or {}))
            for call in (msg.tool_calls or [])
        ]
        return StepResult(
            message=message,
            text=msg.content or "",
            function_calls=calls,
            truncated=response.done_reason == "length",
        )

    def user_text(self, text: str) -> Any:
        return {"role": "user", "content": text}

    def tool_results(self, results: list[tuple[str, dict[str, Any]]]) -> list[Any]:
        return [
            {"role": "tool", "content": json.dumps(payload), "tool_name": name}
            for name, payload in results
        ]


def make_provider(name: str) -> Provider:
    if name == "gemini":
        return GeminiProvider()
    if name == "ollama":
        return OllamaProvider()
    raise ValueError(f"Unknown provider {name!r} (expected 'gemini' or 'ollama').")
