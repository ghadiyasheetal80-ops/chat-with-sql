"""
Offline unit tests for llm.py using pytest and unittest.mock.
Ensures zero live network calls or live Ollama/Gemini dependencies.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import requests

from llm import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_GEMINI_MODEL,
    OllamaError,
    GeminiError,
    load_api_key,
    validate_gemini_api_key,
    get_last_used_model,
    build_sql_prompt,
    build_schema_prompt,
    extract_sql,
    generate_sql,
    generate_schema_sql,
    list_local_models,
    repair_sql,
)


def test_extract_sql_plain():
    sql = "SELECT * FROM users WHERE age > 21;"
    assert extract_sql(sql) == "SELECT * FROM users WHERE age > 21;"


def test_extract_sql_fences_sql():
    content = "Here is your query:\n```sql\nSELECT name FROM students;\n```\nHope that helps!"
    assert extract_sql(content) == "SELECT name FROM students;"


def test_extract_sql_fences_generic():
    content = "```\nSELECT * FROM products ORDER BY price DESC;\n```"
    assert (
        extract_sql(content) == "SELECT * FROM products ORDER BY price DESC;"
    )


def test_extract_sql_think_block():
    content = "<think>\nThinking about the schema...\n</think>\nSELECT id FROM orders;"
    assert extract_sql(content) == "SELECT id FROM orders;"


def test_extract_sql_conversational_preamble():
    content = "Sure! Here is your query:\nSELECT id, title FROM books;"
    assert extract_sql(content) == "SELECT id, title FROM books;"


def test_extract_sql_trailing_explanation():
    content = "SELECT * FROM employees;\nThis query returns all rows from employees table."
    assert extract_sql(content) == "SELECT * FROM employees;"


def test_extract_sql_semicolon_inside_string_literal():
    content = "SELECT name FROM students WHERE name = 'A; B';"
    assert extract_sql(content) == "SELECT name FROM students WHERE name = 'A; B';"


def test_extract_sql_preserves_stacked_sql_for_safety_guard():
    content = "SELECT * FROM students; DROP TABLE students;"
    assert extract_sql(content) == "SELECT * FROM students; DROP TABLE students;"


def test_build_sql_prompt():
    question = "List all employees"
    schema_text = "CREATE TABLE employees (id INT, name TEXT);"
    messages = build_sql_prompt(question, schema_text)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "SQLite query generator" in messages[0]["content"]
    assert schema_text in messages[0]["content"]
    assert "NEVER generate INSERT" in messages[0]["content"]

    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == question


@patch("llm.requests.get")
def test_list_local_models_success(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "models": [{"name": "qwen2.5-coder:3b"}, {"name": "llama3:8b"}]
    }
    mock_get.return_value = mock_response

    models, err = list_local_models("http://localhost:11434")
    assert err is None
    assert models == ["llama3:8b", "qwen2.5-coder:3b"]
    mock_get.assert_called_once_with(
        "http://localhost:11434/api/tags", timeout=3.0
    )


@patch("llm.requests.get")
def test_list_local_models_connection_error(mock_get):
    mock_get.side_effect = requests.exceptions.ConnectionError()

    models, err = list_local_models("http://localhost:11434")
    assert models == []
    assert err is not None
    assert "Could not connect" in err


@patch("llm.requests.get")
def test_list_local_models_timeout(mock_get):
    mock_get.side_effect = requests.exceptions.Timeout()

    models, err = list_local_models("http://localhost:11434")
    assert models == []
    assert "timed out" in err


@patch("llm.requests.post")
def test_generate_sql_happy_path(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {"content": "```sql\nSELECT * FROM customers;\n```"}
    }
    mock_post.return_value = mock_response

    sql = generate_sql("Show all customers", "CREATE TABLE customers (id INT);")
    assert sql == "SELECT * FROM customers;"

    mock_post.assert_called_once()
    called_args, called_kwargs = mock_post.call_args
    assert called_args[0] == "http://127.0.0.1:11434/api/chat"
    json_payload = called_kwargs["json"]
    assert json_payload["model"] == DEFAULT_MODEL
    assert json_payload["stream"] is False
    assert json_payload["options"]["temperature"] == 0.0


def test_generate_sql_empty_question():
    with pytest.raises(ValueError, match="Question cannot be empty"):
        generate_sql("   ", "schema")


@patch("llm.requests.post")
def test_generate_sql_404_error(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_post.return_value = mock_response

    with pytest.raises(OllamaError, match="not found in Ollama.*ollama pull"):
        generate_sql("question", "schema", model="missing-model")


@patch("llm.requests.post")
def test_generate_sql_connection_error(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError()

    with pytest.raises(OllamaError, match="Could not connect to Ollama"):
        generate_sql("question", "schema")


@patch("llm.requests.post")
def test_generate_sql_timeout(mock_post):
    mock_post.side_effect = requests.exceptions.Timeout()

    with pytest.raises(OllamaError, match="timed out"):
        generate_sql("question", "schema")


def test_extract_sql_trigger_block_preservation():
    trigger_content = (
        "Here is the trigger:\n"
        "```sql\n"
        "CREATE TRIGGER check_age AFTER INSERT ON users "
        "BEGIN "
        "SELECT RAISE(ABORT, 'Age must be >= 18'); "
        "END;\n"
        "```\n"
        "This trigger validates user age upon insertion."
    )
    expected = (
        "CREATE TRIGGER check_age AFTER INSERT ON users "
        "BEGIN "
        "SELECT RAISE(ABORT, 'Age must be >= 18'); "
        "END;"
    )
    assert extract_sql(trigger_content) == expected


def test_extract_sql_trigger_without_code_fence():
    content = (
        "CREATE TRIGGER trg_audit AFTER UPDATE ON accounts "
        "BEGIN "
        "INSERT INTO audit_log(acc_id) VALUES(NEW.id); "
        "END;\n"
        "Note: this logs account updates."
    )
    expected = (
        "CREATE TRIGGER trg_audit AFTER UPDATE ON accounts "
        "BEGIN "
        "INSERT INTO audit_log(acc_id) VALUES(NEW.id); "
        "END;"
    )
    assert extract_sql(content) == expected


def test_extract_sql_preamble_stripping_create_table():
    content = "Sure, here is your table schema:\nCREATE TABLE items (id INT, name TEXT);"
    assert extract_sql(content) == "CREATE TABLE items (id INT, name TEXT);"


@patch("llm.requests.post")
def test_repair_sql_happy_path(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {"content": "SELECT id FROM users WHERE active = 1;"}
    }
    mock_post.return_value = mock_response

    repaired = repair_sql(
        question="Show active users",
        schema_text="CREATE TABLE users (id INT, active INT);",
        bad_sql="SELECT id FROM users WHERE active = 'yes';",
        error_message="datatype mismatch",
    )
    assert repaired == "SELECT id FROM users WHERE active = 1;"
    mock_post.assert_called_once()
    called_json = mock_post.call_args[1]["json"]
    user_msg = called_json["messages"][1]["content"]
    assert "Show active users" in user_msg
    assert "SELECT id FROM users WHERE active = 'yes';" in user_msg
    assert "datatype mismatch" in user_msg


@patch("llm.requests.post")
def test_generate_schema_sql_happy_path(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {
            "content": "CREATE TRIGGER check_val BEFORE INSERT ON data BEGIN SELECT RAISE(ABORT, 'invalid'); END;"
        }
    }
    mock_post.return_value = mock_response

    sql = generate_schema_sql("Create a trigger to validate data insertion", "CREATE TABLE data (val INT);")
    assert "CREATE TRIGGER check_val" in sql
    assert "SELECT RAISE(ABORT, 'invalid');" in sql
    mock_post.assert_called_once()


def test_generate_schema_sql_empty_request():
    with pytest.raises(ValueError, match="Request cannot be empty"):
        generate_schema_sql("   ", "schema")


# --- NEW TESTS FOR API KEY, GEMINI, FALLBACK, REDACTION, LOW-END CPU ---

def test_load_api_key_priority_and_placeholders(tmp_path):
    key, src = load_api_key("paste_your_google_ai_studio_key_here")
    assert key is None
    assert src == "none"

    key, src = load_api_key("AIzaSyTestKey123456789")
    assert key == "AIzaSyTestKey123456789"
    assert src == "sidebar"

    env_file = tmp_path / ".env"
    env_file.write_text('GEMINI_API_KEY="env_file_key_123"', encoding="utf-8")

    with patch.dict("os.environ", {"GEMINI_API_KEY": "os_env_key_999"}):
        key, src = load_api_key(None, env_path=env_file)
        assert key == "os_env_key_999"
        assert src == "env"

    with patch.dict("os.environ", {}, clear=True):
        if "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]
        key, src = load_api_key(None, env_path=env_file)
        assert key == "env_file_key_123"
        assert src == ".env"


@patch("llm.requests.get")
def test_validate_gemini_api_key_headers_and_redaction(mock_get):
    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_get.return_value = mock_resp_200

    valid, msg = validate_gemini_api_key("AIzaSySecretKeyXYZ")
    assert valid is True
    assert "verified" in msg
    mock_get.assert_called_once()
    called_url, called_kwargs = mock_get.call_args
    assert "key=" not in called_url[0]
    assert called_kwargs["headers"]["x-goog-api-key"] == "AIzaSySecretKeyXYZ"

    mock_resp_403 = MagicMock()
    mock_resp_403.status_code = 403
    mock_resp_403.text = "Invalid API key: AIzaSySecretKeyXYZ provided."
    mock_get.return_value = mock_resp_403

    valid, msg = validate_gemini_api_key("AIzaSySecretKeyXYZ")
    assert valid is False
    assert "AIzaSySecretKeyXYZ" not in msg
    assert "[REDACTED]" in msg


@patch("llm.requests.post")
def test_generate_sql_gemini_happy_path(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "SELECT * FROM users WHERE active = 1;"}]
                }
            }
        ]
    }
    mock_post.return_value = mock_resp

    with patch("llm.load_api_key", return_value=("secret-test-key", "sidebar")):
        sql = generate_sql(
            "Show active users",
            "CREATE TABLE users (id INT, active INT);",
            provider="gemini",
            api_key="secret-test-key",
        )
        assert sql == "SELECT * FROM users WHERE active = 1;"

        mock_post.assert_called_once()
        url, kwargs = mock_post.call_args
        assert "key=" not in url[0]
        assert kwargs["headers"]["x-goog-api-key"] == "secret-test-key"
        json_payload = kwargs["json"]
        assert "systemInstruction" in json_payload
        assert json_payload["generationConfig"]["temperature"] == 0.0


@patch("llm.requests.post")
def test_generate_sql_gemma_compatibility(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "SELECT name FROM gemma_users;"}]
                }
            }
        ]
    }
    mock_post.return_value = mock_resp

    with patch("llm.load_api_key", return_value=("secret-test-key", "sidebar")):
        sql = generate_sql(
            "List users",
            "CREATE TABLE gemma_users (name TEXT);",
            provider="gemini",
            model="gemma-3-27b-it",
            api_key="secret-test-key",
        )
        assert sql == "SELECT name FROM gemma_users;"

        called_kwargs = mock_post.call_args[1]
        json_payload = called_kwargs["json"]
        assert "systemInstruction" not in json_payload
        user_content = json_payload["contents"][0]["parts"][0]["text"]
        assert "SQLite query generator" in user_content
        assert "List users" in user_content


@patch("llm.requests.post")
def test_gemini_fallback_chain_and_last_model(mock_post):
    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.text = "Rate limit exceeded"

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "SELECT COUNT(*) FROM orders;"}]
                }
            }
        ]
    }

    mock_post.side_effect = [resp_429, resp_429, resp_200]

    with patch("llm.load_api_key", return_value=("secret-test-key", "sidebar")):
        sql = generate_sql(
            "Count orders",
            "CREATE TABLE orders (id INT);",
            provider="gemini",
            model="gemini-3.5-flash-lite",
            api_key="secret-test-key",
            auto_fallback=True,
        )
        assert sql == "SELECT COUNT(*) FROM orders;"
        assert mock_post.call_count == 3
        assert get_last_used_model() == "gemma-3-27b-it"


@patch("llm.requests.post")
def test_ollama_low_end_cpu_mode(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "message": {"content": "SELECT * FROM low_cpu;"}
    }
    mock_post.return_value = mock_resp

    sql = generate_sql(
        "Select all",
        "CREATE TABLE low_cpu (id INT);",
        provider="ollama",
        low_end_cpu=True,
        timeout=30.0,
    )
    assert sql == "SELECT * FROM low_cpu;"

    mock_post.assert_called_once()
    called_kwargs = mock_post.call_args[1]
    assert called_kwargs["timeout"] >= 120.0
    json_payload = called_kwargs["json"]
    assert json_payload["keep_alive"] == "5m"
    assert json_payload["options"]["num_ctx"] == 2048
    assert json_payload["options"]["num_predict"] == 256
    assert json_payload["options"]["num_thread"] == 2
    assert get_last_used_model() == DEFAULT_MODEL
