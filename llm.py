"""
LLM integration module for Chat-with-SQL supporting both Ollama (local) and Google AI Studio (Gemini).
Provides zero-SDK, pure-requests client interaction, SQL prompt building, robust SQL extraction,
API key loading/validation, fallback chains, and low-end CPU optimization.
"""

import os
import re
from pathlib import Path
import requests

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5-coder:3b"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemma-3-27b-it",
    "gemini-3.8-flash",
]
GEMINI_FALLBACK_CHAIN = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemma-3-27b-it",
]
DEFAULT_ENV_PATH = Path(__file__).resolve().parent / ".env"

_LAST_USED_MODEL: str | None = None


class LLMError(RuntimeError):
    """Base exception raised for LLM errors."""
    pass


class OllamaError(LLMError):
    """Custom exception raised for Ollama API or generation errors."""
    pass


class GeminiError(LLMError):
    """Custom exception raised for Google AI Studio API or generation errors."""
    pass


def get_last_used_model() -> str | None:
    """Returns the last successfully used LLM model name."""
    return _LAST_USED_MODEL


def _redact_secret(text: str, secret: str | None) -> str:
    """Replaces any occurrence of secret in text with [REDACTED]."""
    if not secret or not isinstance(secret, str) or len(secret.strip()) < 4:
        return text
    cleaned_secret = secret.strip()
    return text.replace(cleaned_secret, "[REDACTED]")


def load_api_key(
    explicit_key: str | None = None, env_path: Path | str | None = None
) -> tuple[str | None, str]:
    """Loads and validates Gemini API key from explicit arg, environment, or .env file.
    Returns (cleaned_key_or_None, source) where source in {"sidebar", "env", ".env", "none"}.
    Ignores empty strings and placeholder values.
    """
    placeholders = {
        "paste_your_google_ai_studio_key_here",
        "your_api_key",
        "your_gemini_api_key",
    }

    def _is_valid(val: str | None) -> bool:
        if not val or not isinstance(val, str):
            return False
        stripped = val.strip()
        if not stripped:
            return False
        if stripped.lower() in placeholders:
            return False
        return True

    # 1. explicit_key
    if explicit_key is not None:
        if _is_valid(explicit_key):
            return explicit_key.strip(), "sidebar"

    # 2. environment variable
    env_val = os.environ.get("GEMINI_API_KEY")
    if _is_valid(env_val):
        return env_val.strip(), "env"

    # 3. .env file
    target_env_path = Path(env_path) if env_path else DEFAULT_ENV_PATH
    if target_env_path.exists() and target_env_path.is_file():
        try:
            content = target_env_path.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "GEMINI_API_KEY" in line:
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        k = parts[0].strip()
                        if k == "GEMINI_API_KEY":
                            v = parts[1].strip()
                            # Strip surrounding quotes if present
                            if (v.startswith('"') and v.endswith('"')) or (
                                v.startswith("'") and v.endswith("'")
                            ):
                                v = v[1:-1].strip()
                            if _is_valid(v):
                                return v, ".env"
        except Exception:
            pass

    return None, "none"


def validate_gemini_api_key(api_key: str, timeout: float = 10.0) -> tuple[bool, str]:
    """Validates Gemini API key by calling GET https://generativelanguage.googleapis.com/v1beta/models
    with header x-goog-api-key. Returns (True, success_msg) or (False, error_msg).
    """
    if not api_key or not api_key.strip():
        return False, "API key cannot be empty."

    cleaned_key = api_key.strip()
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    headers = {"x-goog-api-key": cleaned_key}

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            return True, "API key verified with Google AI Studio."
        else:
            err_text = _redact_secret(response.text, cleaned_key)
            return (
                False,
                f"Google AI Studio returned HTTP {response.status_code}: {err_text}",
            )
    except requests.exceptions.ConnectionError:
        return False, "Connection error validating API key with Google AI Studio."
    except requests.exceptions.Timeout:
        return False, f"Validation request timed out after {timeout}s."
    except Exception as e:
        err_text = _redact_secret(str(e), cleaned_key)
        return False, f"Unexpected error validating API key: {err_text}"


def list_local_models(
    base_url: str = DEFAULT_OLLAMA_URL, timeout: float = 3.0
) -> tuple[list[str], str | None]:
    """Lists local Ollama models by querying /api/tags.
    Returns (sorted_model_names, None) on success.
    On connection error, timeout, or non-200 HTTP error, returns ([], error_message).
    NEVER raises exceptions.
    """
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200:
            return (
                [],
                f"Ollama server returned HTTP status {response.status_code} from {url}.",
            )

        data = response.json()
        models_raw = data.get("models", [])
        model_names = [
            m.get("name") for m in models_raw if isinstance(m, dict) and m.get("name")
        ]
        return (sorted(model_names), None)
    except requests.exceptions.ConnectionError:
        return (
            [],
            f"Could not connect to Ollama at {base_url}. Is Ollama running?",
        )
    except requests.exceptions.Timeout:
        return [], f"Connection to Ollama at {base_url} timed out after {timeout}s."
    except Exception as e:
        return [], f"Unexpected error listing Ollama models: {str(e)}"


def build_sql_prompt(question: str, schema_text: str) -> list[dict[str, str]]:
    """Builds system and user chat messages instructing the model to generate SQLite queries."""
    system_prompt = (
        "You are an expert SQLite query generator.\n"
        "Output ONLY a single valid SQLite SELECT query (or read-only WITH CTE).\n"
        "Do NOT include explanations, markdown formatting, or multiple statements.\n"
        "NEVER generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or PRAGMA.\n"
        "Strictly use ONLY the table names, column names, and categorical values provided in the schema below.\n"
        "Use standard SQLite syntax (LIMIT, not TOP; LIKE, not ILIKE).\n"
        "Multi-table join & aggregation guidance:\n"
        "- Follow the -- JOIN PATH: relationships in the schema strictly; NEVER assume a foreign key column exists on a table unless it is listed in that table's CREATE TABLE definition.\n"
        "- When a question asks for totals, averages, counts, highest, or lowest per department / branch / group, use aggregate functions (SUM, AVG, COUNT) with GROUP BY rather than picking a single row.\n"
        "- When calculating unpaid fees, sum unpaid_fee directly across all matching rows (do NOT filter only payment_status = 'Unpaid', because 'Partial' records also have unpaid balances).\n\n"
        f"Database Schema:\n{schema_text}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def build_schema_prompt(request: str, schema_text: str) -> list[dict[str, str]]:
    """Builds system and user chat messages instructing the model to generate SQLite DDL/Trigger/Insert statements."""
    system_prompt = (
        "You are an expert SQLite database schema and trigger generation assistant.\n"
        "Output ONLY a single valid SQLite statement: CREATE TABLE, CREATE VIEW, CREATE INDEX, CREATE TRIGGER, or INSERT INTO.\n"
        "Do NOT include explanations, markdown formatting, or multiple statements.\n"
        "NEVER generate DROP, DELETE, TRUNCATE, ALTER, ATTACH, or PRAGMA.\n"
        "For triggers, use valid SQLite syntax: CREATE TRIGGER <name> BEFORE|AFTER INSERT|UPDATE ON <table> FOR EACH ROW [WHEN <condition>] BEGIN <statements>; END;\n"
        "Use SELECT RAISE(ABORT, 'message'); for validation errors inside triggers.\n\n"
        f"Existing Database Schema:\n{schema_text}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request},
    ]


def extract_sql(raw_response: str) -> str:
    """Cleans and extracts pure SQL from raw LLM output, handling think blocks,
    markdown code fences, conversational preambles, and trailing explanations.
    """
    if not raw_response:
        return ""

    text = raw_response.strip()

    # 1. Strip <think>...</think> reasoning blocks (case-insensitive, DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    if re.search(r"<think>", text, flags=re.IGNORECASE):
        text = re.sub(r"<think>.*", "", text, flags=re.IGNORECASE | re.DOTALL)

    text = text.strip()

    # 2. Extract markdown code fences if present
    fence_pattern = re.compile(
        r"```(?:sql)?\s*(.*?)\s*```", flags=re.IGNORECASE | re.DOTALL
    )
    match = fence_pattern.search(text)
    if match:
        text = match.group(1).strip()

    # 3. If no fences were present, find first standalone SQL starter keyword
    if not match:
        keyword_match = re.search(
            r"\b(SELECT|WITH|CREATE|INSERT)\b", text, flags=re.IGNORECASE
        )
        if keyword_match:
            text = text[keyword_match.start() :]

    text = text.strip()

    # 4. Trigger / semicolon trailing prose handling
    if re.match(r"^\s*CREATE\s+(?:TEMP(?:ORARY)?\s+)?TRIGGER", text, flags=re.IGNORECASE):
        in_quote = False
        end_match_idx = -1
        i = 0
        while i < len(text):
            ch = text[i]
            if in_quote:
                if ch == "'":
                    if i + 1 < len(text) and text[i + 1] == "'":
                        i += 2
                        continue
                    in_quote = False
            else:
                if ch == "'":
                    in_quote = True
                elif text[i : i + 3].upper() == "END" and (
                    i == 0 or not text[i - 1].isalnum() and text[i - 1] != "_"
                ):
                    j = i + 3
                    while j < len(text) and text[j].isspace():
                        j += 1
                    if j < len(text) and text[j] == ";":
                        j += 1
                    end_match_idx = j
                    break
            i += 1

        if end_match_idx != -1:
            trigger_block = text[:end_match_idx].strip()
            trailing = text[end_match_idx:].strip()
            sql_follow_pattern = re.compile(
                r"^(SELECT|WITH|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|PRAGMA)\b",
                flags=re.IGNORECASE,
            )
            if trailing and sql_follow_pattern.match(trailing):
                text = trigger_block + " " + trailing
            else:
                text = trigger_block
    else:
        in_quote = False
        semi_idx = -1
        i = 0
        while i < len(text):
            ch = text[i]
            if in_quote:
                if ch == "'":
                    if i + 1 < len(text) and text[i + 1] == "'":
                        i += 2
                        continue
                    in_quote = False
            else:
                if ch == "'":
                    in_quote = True
                elif ch == ";":
                    semi_idx = i
                    break
            i += 1

        if semi_idx != -1:
            trailing = text[semi_idx + 1 :].strip()
            sql_follow_pattern = re.compile(
                r"^(SELECT|WITH|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|PRAGMA)\b",
                flags=re.IGNORECASE,
            )
            if not trailing or not sql_follow_pattern.match(trailing):
                text = text[: semi_idx + 1]

    return text.strip()


def _call_ollama(
    messages: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = 60.0,
    low_end_cpu: bool = False,
) -> str:
    """Sends messages to Ollama /api/chat endpoint."""
    global _LAST_USED_MODEL
    url = f"{base_url.rstrip('/')}/api/chat"

    options = {"temperature": 0.0}
    keep_alive = None
    effective_timeout = max(timeout, 120.0) if low_end_cpu else timeout

    if low_end_cpu:
        options = {
            "temperature": 0.0,
            "num_ctx": 2048,
            "num_predict": 256,
            "num_thread": 2,
        }
        keep_alive = "5m"

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive

    try:
        response = requests.post(url, json=payload, timeout=effective_timeout)
    except requests.exceptions.ConnectionError as e:
        raise OllamaError(
            f"Could not connect to Ollama at {base_url}. Please ensure Ollama is running."
        ) from e
    except requests.exceptions.Timeout as e:
        raise OllamaError(
            f"Ollama request timed out after {effective_timeout} seconds. The model might be loading or busy."
        ) from e
    except Exception as e:
        raise OllamaError(
            f"Unexpected error communicating with Ollama: {str(e)}"
        ) from e

    if response.status_code == 404:
        raise OllamaError(
            f"Model '{model}' not found in Ollama (HTTP 404). Please run: ollama pull {model}"
        )
    elif response.status_code != 200:
        raise OllamaError(
            f"Ollama API error (HTTP {response.status_code}): {response.text}"
        )

    try:
        data = response.json()
        raw_content = data["message"]["content"]
    except (KeyError, ValueError) as e:
        raise OllamaError(
            f"Malformed JSON response from Ollama API: {response.text}"
        ) from e

    _LAST_USED_MODEL = model
    return raw_content


def _call_gemini(
    messages: list[dict[str, str]],
    model: str = DEFAULT_GEMINI_MODEL,
    timeout: float = 60.0,
    api_key: str | None = None,
    auto_fallback: bool = True,
) -> str:
    """Calls Google AI Studio Gemini API with fallback chain and Gemma compatibility."""
    global _LAST_USED_MODEL
    resolved_key, _ = load_api_key(api_key)
    if not resolved_key:
        raise GeminiError(
            "Google AI Studio API key not found. Please provide GEMINI_API_KEY in .env or sidebar."
        )

    # Extract system prompt and user prompt from messages
    system_prompt = ""
    user_prompt = ""
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            system_prompt = content
        elif role == "user":
            user_prompt = content

    # Build candidate models
    candidate_models = [model]
    if auto_fallback:
        for m in GEMINI_FALLBACK_CHAIN:
            if m not in candidate_models:
                candidate_models.append(m)

    last_error = None
    for target_model in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": resolved_key}

        # Payload construction (Gemma vs Gemini)
        if target_model.lower().startswith("gemma"):
            combined_text = f"{system_prompt}\n\nUser request:\n{user_prompt}" if system_prompt else user_prompt
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": combined_text}],
                    }
                ],
                "generationConfig": {"temperature": 0.0},
            }
        else:
            payload = {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": user_prompt}],
                    }
                ],
                "generationConfig": {"temperature": 0.0},
            }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            status = response.status_code

            if status == 200:
                data = response.json()
                try:
                    parts = data["candidates"][0]["content"]["parts"]
                    text_content = "".join(p.get("text", "") for p in parts)
                    _LAST_USED_MODEL = target_model
                    return text_content
                except (KeyError, IndexError, TypeError) as parse_err:
                    raise GeminiError(
                        f"Malformed response from Google AI Studio ({target_model}): {response.text}"
                    ) from parse_err

            elif status in (400, 401, 403):
                err_text = _redact_secret(response.text, resolved_key)
                raise GeminiError(
                    f"Google AI Studio authentication/request error (HTTP {status}): {err_text}"
                )

            elif status in (429, 503):
                last_error = GeminiError(
                    f"Google AI Studio model {target_model} rate limited or unavailable (HTTP {status}): {response.text}"
                )
                continue  # Try next fallback model
            else:
                err_text = _redact_secret(response.text, resolved_key)
                last_error = GeminiError(
                    f"Google AI Studio API error (HTTP {status}): {err_text}"
                )
                continue

        except requests.exceptions.RequestException as req_err:
            last_error = GeminiError(
                f"Network error communicating with Google AI Studio ({target_model}): {str(req_err)}"
            )
            continue

    if last_error:
        raise last_error
    raise GeminiError("All Gemini fallback models exhausted without successful response.")


def generate_sql(
    question: str,
    schema_text: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = 60.0,
    provider: str = "ollama",
    api_key: str | None = None,
    auto_fallback: bool = True,
    low_end_cpu: bool = False,
) -> str:
    """Generates SQLite query from question and schema using Ollama or Gemini."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    messages = build_sql_prompt(question, schema_text)

    if provider.lower() == "gemini":
        gemini_model = model if model != DEFAULT_MODEL else DEFAULT_GEMINI_MODEL
        raw_content = _call_gemini(
            messages,
            model=gemini_model,
            timeout=timeout,
            api_key=api_key,
            auto_fallback=auto_fallback,
        )
    else:
        raw_content = _call_ollama(
            messages,
            model=model,
            base_url=base_url,
            timeout=timeout,
            low_end_cpu=low_end_cpu,
        )

    sql = extract_sql(raw_content)
    if not sql:
        raise LLMError(
            f"LLM returned a response, but no valid SQL could be extracted. Raw response:\n{raw_content}"
        )

    return sql


def repair_sql(
    question: str,
    schema_text: str,
    bad_sql: str,
    error_message: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = 60.0,
    provider: str = "ollama",
    api_key: str | None = None,
    auto_fallback: bool = True,
    low_end_cpu: bool = False,
) -> str:
    """Repairs failed SQLite query given error message using Ollama or Gemini."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")
    if not bad_sql or not bad_sql.strip():
        raise ValueError("Bad SQL cannot be empty.")
    if not error_message or not error_message.strip():
        raise ValueError("Error message cannot be empty.")

    system_prompt = (
        "You are an expert SQLite query repair assistant.\n"
        "Your task is to fix a previously generated SQLite query that resulted in a runtime error.\n"
        "Output ONLY a corrected single read-only SQLite SELECT or WITH query.\n"
        "Do NOT include explanations, markdown formatting, or multiple statements.\n"
        "NEVER generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or PRAGMA.\n\n"
        f"Database Schema:\n{schema_text}"
    )

    user_prompt = (
        f"Original Question: {question}\n\n"
        f"Failed SQL Query:\n{bad_sql}\n\n"
        f"SQLite Error Message:\n{error_message}\n\n"
        "Provide the corrected read-only SQLite query:"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if provider.lower() == "gemini":
        gemini_model = model if model != DEFAULT_MODEL else DEFAULT_GEMINI_MODEL
        raw_content = _call_gemini(
            messages,
            model=gemini_model,
            timeout=timeout,
            api_key=api_key,
            auto_fallback=auto_fallback,
        )
    else:
        raw_content = _call_ollama(
            messages,
            model=model,
            base_url=base_url,
            timeout=timeout,
            low_end_cpu=low_end_cpu,
        )

    sql = extract_sql(raw_content)
    if not sql:
        raise LLMError(
            f"LLM returned a response, but no valid SQL could be extracted during repair. Raw response:\n{raw_content}"
        )

    return sql


def generate_schema_sql(
    request: str,
    schema_text: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = 60.0,
    provider: str = "ollama",
    api_key: str | None = None,
    auto_fallback: bool = True,
    low_end_cpu: bool = False,
) -> str:
    """Generates DDL/Trigger statement using Ollama or Gemini."""
    if not request or not request.strip():
        raise ValueError("Request cannot be empty.")

    messages = build_schema_prompt(request, schema_text)

    if provider.lower() == "gemini":
        gemini_model = model if model != DEFAULT_MODEL else DEFAULT_GEMINI_MODEL
        raw_content = _call_gemini(
            messages,
            model=gemini_model,
            timeout=timeout,
            api_key=api_key,
            auto_fallback=auto_fallback,
        )
    else:
        raw_content = _call_ollama(
            messages,
            model=model,
            base_url=base_url,
            timeout=timeout,
            low_end_cpu=low_end_cpu,
        )

    sql = extract_sql(raw_content)
    if not sql:
        raise LLMError(
            f"LLM returned a response, but no valid SQL could be extracted. Raw response:\n{raw_content}"
        )

    return sql
