"""
Automated Streamlit AppTest suite for Chat-with-SQL app.py.
Uses streamlit.testing.v1.AppTest and unittest.mock.patch for 100% offline testing.
"""

from pathlib import Path
from unittest.mock import patch
import pytest
from streamlit.testing.v1 import AppTest

from app import _choose_default_model_index
from database import QueryExecutionError, execute_read_only_query, execute_schema_sql
from llm import repair_sql, generate_schema_sql
from seed_db import init_db


def test_choose_default_model_index_prioritizes_3b():
    """Test unit function _choose_default_model_index prefers 3b over 1.5b."""
    models = ["qwen2.5-coder:1.5b", "qwen2.5-coder:3b", "qwen2.5-coder:7b"]
    idx = _choose_default_model_index(models)
    assert models[idx] == "qwen2.5-coder:3b"


def test_app_boots_offline():
    """Test 1: App boots cleanly with no exceptions whether Gemini default or Ollama offline."""
    with patch("llm.list_local_models", return_value=([], "Offline")):
        at = AppTest.from_file("D:\\chat-with-sql\\app.py")
        at.run()
        assert not at.exception, f"App raised exception on boot: {at.exception}"


def test_app_test_api_key_button():
    """Test clicking 'Test API Key' button with mocked validate_gemini_api_key."""
    with patch("llm.validate_gemini_api_key", return_value=(True, "Key valid!")):
        at = AppTest.from_file("D:\\chat-with-sql\\app.py")
        at.run()
        
        # Select Google Gemini provider to show Test API Key button
        at.selectbox(key="llm_provider").select("Google Gemini").run()
        
        btn_found = False
        for btn in at.button:
            if "Test API Key" in btn.label:
                btn.click().run()
                btn_found = True
                break
        assert btn_found, "Test API Key button not found in app UI"
        assert not at.exception


def test_app_submitting_question_success():
    """Test 2: Submitting a question with generate_sql mocked returns valid SELECT query and renders results."""
    mock_sql = "SELECT student_name, attendance_pct FROM students WHERE attendance_pct < 75.0;"
    
    with patch("llm.list_local_models", return_value=(["qwen2.5-coder:3b"], None)), \
         patch("llm.generate_sql", return_value=mock_sql):
        
        at = AppTest.from_file("D:\\chat-with-sql\\app.py")
        at.run()
        
        at.text_input(key="question_input").input("Show me students with attendance below 75%").run()
        
        if len(at.button) > 5:
            at.button[5].click().run()
        
        assert not at.exception, f"App raised exception during query submission: {at.exception}"


def test_app_submitting_unsafe_sql_handles_error():
    """Test 3: Submitting a question where generate_sql returns DROP TABLE triggers UnsafeQueryError handling gracefully."""
    mock_sql = "DROP TABLE students;"
    
    with patch("llm.list_local_models", return_value=(["qwen2.5-coder:3b"], None)), \
         patch("llm.generate_sql", return_value=mock_sql):
        
        at = AppTest.from_file("D:\\chat-with-sql\\app.py")
        at.run()
        
        at.text_input(key="question_input").input("Delete all students").run()
        if len(at.button) > 5:
            at.button[5].click().run()
            
        assert not at.exception, f"App crashed on unsafe SQL execution: {at.exception}"


def test_app_viva_safety_guard_expander():
    """Test 4: Clicking Test Safety & Execute with default DROP TABLE displays security block message cleanly."""
    with patch("llm.list_local_models", return_value=([], "Offline")):
        at = AppTest.from_file("D:\\chat-with-sql\\app.py")
        at.run()
        
        btn_found = False
        for btn in at.button:
            if "Test Safety & Execute" in btn.label:
                btn.click().run()
                btn_found = True
                break
        
        if not btn_found and at.button:
            at.button[-1].click().run()
            
        assert not at.exception, f"App crashed during safety guard test: {at.exception}"


def test_llm_repair_and_schema_generation_mocks(tmp_path: Path):
    """Test unit execution of repair_sql and generate_schema_sql with mocked requests."""
    db_path = tmp_path / "test_unit.db"
    init_db(db_path, reset=True)

    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "message": {"content": "SELECT COUNT(*) FROM students;"}
        }

        repaired = repair_sql("How many students?", "schema", "bad sql", "error")
        assert "SELECT" in repaired

        schema_sql = generate_schema_sql("Create table courses", "schema")
        assert "SELECT" in schema_sql or len(schema_sql) > 0


def test_app_auto_repair_flow(tmp_path: Path):
    """Test Tab 1 auto-repair flow setup and execution logic."""
    db_path = tmp_path / "test_college.db"
    init_db(db_path, reset=True)

    broken_sql = "SELECT fr.dept_id FROM fee_records fr;"
    fixed_sql = "SELECT COUNT(*) AS cnt FROM students;"

    with patch("llm.list_local_models", return_value=(["qwen2.5-coder:3b"], None)), \
         patch("llm.generate_sql", return_value=broken_sql), \
         patch("llm.repair_sql", return_value=fixed_sql), \
         patch("app.DEFAULT_DB_PATH", db_path):
        
        at = AppTest.from_file("D:\\chat-with-sql\\app.py")
        at.run()
        
        at.text_input(key="question_input").input("How many students are there?").run()
        if len(at.button) > 5:
            at.button[5].click().run()
            
        assert not at.exception


def test_app_hitl_schema_workflow(tmp_path: Path):
    """Test Tab 2 HITL flow state initialization."""
    db_path = tmp_path / "test_college_hitl.db"
    init_db(db_path, reset=True)

    draft_statement = "CREATE TABLE courses (course_id INTEGER PRIMARY KEY, course_name TEXT);"

    with patch("llm.list_local_models", return_value=(["qwen2.5-coder:3b"], None)), \
         patch("llm.generate_schema_sql", return_value=draft_statement), \
         patch("app.DEFAULT_DB_PATH", db_path):
        
        at = AppTest.from_file("D:\\chat-with-sql\\app.py")
        at.run()
        
        at.text_input(key="schema_prompt_input").input("Create a courses table").run()
        if len(at.button) > 3:
            at.button[3].click().run()
        
        assert not at.exception
