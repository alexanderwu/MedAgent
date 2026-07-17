# Local LLM toggle for MedAgent (Ollama + gpt-oss:20b)

## Context

MedAgent currently talks only to the Gemini API; approved query results (MIMIC-IV rows) leave the machine. The goal is a provider toggle — Gemini (cloud) vs. a local model via Ollama — so analysis can run fully on-machine (PHI stays local). Target model: **gpt-oss:20b** (fits the 5060 Ti 16GB via native MXFP4; strong native tool-calling). Ollama is already installed (the justfile uses it for commit messages).

Constraints (from project memory): keep the hand-rolled, resumable `AgentSession` loop — no LangChain, no SDK-managed tool loop, no refactor to callbacks/generators. The provider becomes a thin seam; the approval state machine stays untouched.

Decisions already made with Alex: sidebar radio + `MEDAGENT_PROVIDER` env default; official `ollama` Python package (not OpenAI-compat endpoint).

## Design: provider seam

New module `src/medagent/providers.py` containing everything provider-specific; `agent.py` becomes provider-agnostic.

```python
@dataclass
class FunctionCall:
    name: str
    args: dict[str, Any]

@dataclass
class StepResult:
    message: Any                      # provider-native assistant message (appended to transcript)
    text: str
    function_calls: list[FunctionCall]
    truncated: bool                   # hit max-tokens finish reason

class ProviderError(Exception): ...   # user-facing message; loop maps it to _fail()

class Provider(Protocol):
    name: str      # "gemini" | "ollama"
    model: str
    def generate(self, system: str, contents: list[Any], use_tools: bool) -> StepResult: ...
    def user_text(self, text: str) -> Any                              # initial user message
    def tool_results(self, results: list[tuple[str, dict]]) -> list[Any]  # (fn_name, payload) in call order → transcript messages
```

- Tool-result payloads keep the existing `{"result": ...}` / `{"error": ...}` convention; each provider serializes them its own way.
- Synthetic call ids (`call_{iter}_{index}`) stay in the loop — neither API supplies usable ids; providers only need the function *name* to build responses (matches the current `_call_names` mechanism).
- `GeminiProvider`: wraps `genai.Client`; `generate` builds the existing `GenerateContentConfig` (system_instruction, tools, AFC disabled); maps `genai_errors.APIError` / `httpx.RequestError` / empty-candidates to `ProviderError`. `tool_results` returns one `Content(role="user", parts=[Part.from_function_response...])` — the current one-message-per-turn Gemini requirement.
- `OllamaProvider`: wraps `ollama.Client()` (respects `OLLAMA_HOST`); transcript entries are Ollama chat dicts. `generate` prepends `{"role": "system", ...}` at call time and passes `tools=` converted from `tools.FUNCTION_DECLARATIONS` (already JSON-schema-shaped → trivial mapping to `{"type": "function", "function": {...}}`), plus `options={"num_ctx": LOCAL_NUM_CTX}` (Ollama's 4096 default is too small for schema dumps + CSV results). `tool_results` returns one `{"role": "tool", "content": json.dumps(payload), "tool_name": name}` message per call. Maps `ollama.ResponseError` / connection errors to `ProviderError` with actionable hints ("is `ollama serve` running?", "run `ollama pull gpt-oss:20b`"). gpt-oss thinking output is ignored (its `thinking` field is not fed back or displayed — future enhancement).
- Factory `make_provider(name: str) -> Provider` reading model names from config.

## File changes

1. **`pyproject.toml`** — add `ollama` dependency (`uv add ollama`).
2. **`src/medagent/config.py`** — add `PROVIDER = os.getenv("MEDAGENT_PROVIDER", "gemini")`, `LOCAL_MODEL = os.getenv("MEDAGENT_LOCAL_MODEL", "gpt-oss:20b")`, `LOCAL_NUM_CTX = int(os.getenv("MEDAGENT_LOCAL_NUM_CTX", "16384"))`. `MODEL` keeps its Gemini meaning.
3. **`src/medagent/providers.py`** (new) — as designed above.
4. **`src/medagent/agent.py`** — `AgentSession.client: genai.Client` → `provider: Provider`; `contents: list[Any]` built via `provider.user_text()`; `_step` calls `provider.generate(...)` inside a single `except ProviderError` → `_fail`; `resolved_results` stores payload dicts instead of `types.Part`; `_flush_tool_results` does `self.contents.extend(provider.tool_results(ordered))`; delete `_function_response`/`_text_of`/`_call_names` name-tracking where it moves into providers; drop `google.*`/`httpx` imports. State machine (`PendingSQL`, `resolve_sql`, statuses) unchanged.
5. **`src/medagent/app.py`** — sidebar radio "Provider" (`Gemini (cloud)` / `Local · gpt-oss:20b`) defaulting from `config.PROVIDER`, mirroring the dataset radio; mid-question provider switch drops the agent (same pattern as the dataset-switch guard at [app.py:84-87](src/medagent/app.py#L84-L87)); `_client()` → `@st.cache_resource def _provider(name)`; model caption shows active provider/model; **approval caption becomes provider-aware** — Gemini keeps the "sends result rows to Google" warning, local says results stay on this machine.
6. **`src/medagent/cli.py`** — `--provider {gemini,ollama}` flag (default `config.PROVIDER`); header line and approval prompt provider-aware.
7. **`.env.example`** — add `MEDAGENT_PROVIDER=gemini` and `MEDAGENT_LOCAL_MODEL=gpt-oss:20b`.
8. **`README.md`** — Quick Start note + short "Local LLM (Ollama)" section: `ollama pull gpt-oss:20b`, the toggle, and the privacy implication (local provider keeps result rows on-machine); update tech-stack table row.

## Verification

1. `just lint` (ruff + mypy) clean.
2. One-time: `ollama pull gpt-oss:20b` (~13 GB download — needs Alex or their approval).
3. CLI smoke tests against the demo DB:
   - `uv run python -m medagent.cli "What are the most common admission types?" --provider ollama` — approve path: schema → SQL → answer.
   - Same question, reject the query — model should adapt, not error.
   - Default (Gemini) run unchanged — regression check on the refactored seam.
4. `just app`: toggle the sidebar radio both ways mid-session; confirm the approval caption changes and an in-flight question is dropped cleanly on switch.

## Out of scope (deliberately)

- Streaming responses, displaying gpt-oss thinking traces, OpenAI-compat multi-runtime support (LM Studio/vLLM), and automated tests (repo has none yet; the justfile test target is still commented out).
