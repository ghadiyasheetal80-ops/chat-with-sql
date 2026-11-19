"""
Streamlit UI for Chat-with-SQL.
Provides a minimalist Linear/Vercel-styled interface for natural language to SQL querying,
schema/trigger/DDL inspection, sample database reset, and automated Human-in-the-Loop (HITL) schema management.
"""

import csv
import io
from pathlib import Path

import streamlit as st

from database import (
    QueryExecutionError,
    UnsafeQueryError,
    execute_read_only_query,
    execute_schema_sql,
    get_full_ddl,
    get_schema_summary,
    get_tables_info,
    get_triggers_info,
)
from llm import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_GEMINI_MODEL,
    GEMINI_MODELS,
    OllamaError,
    GeminiError,
    load_api_key,
    validate_gemini_api_key,
    generate_schema_sql,
    generate_sql,
    list_local_models,
    repair_sql,
)
from safety import validate_schema_sql
from seed_db import DEFAULT_DB_PATH, init_db


def _choose_default_model_index(models: list[str]) -> int:
    """
    Returns the index of the best default model according to priority:
    1. Exact "qwen2.5-coder:3b"
    2. Exact "qwen2.5-coder:7b"
    3. Exact "qwen2.5-coder:14b"
    4. Any coder model whose tag is NOT 1.5b
    5. Any "qwen2.5:7b" or "qwen2.5:3b"
    6. Default to 0
    """
    if not models:
        return 0

    for i, m in enumerate(models):
        if m.lower() == "qwen2.5-coder:3b":
            return i

    for i, m in enumerate(models):
        if m.lower() == "qwen2.5-coder:7b":
            return i

    for i, m in enumerate(models):
        if m.lower() == "qwen2.5-coder:14b":
            return i

    for i, m in enumerate(models):
        m_lower = m.lower()
        if "coder" in m_lower and "1.5b" not in m_lower:
            return i

    for i, m in enumerate(models):
        m_lower = m.lower()
        if m_lower in ("qwen2.5:7b", "qwen2.5:3b"):
            return i

    return 0


def main() -> int:
    # 1. Page Config
    st.set_page_config(page_title="Chat with SQL", layout="centered")

    # 2. Minimalist Linear/Vercel custom CSS
    st.markdown(
        """
        <style>
        div[data-testid="stExpander"] {
            border: 1px solid rgba(128, 128, 128, 0.18) !important;
            border-radius: 6px !important;
            background-color: transparent !important;
        }
        .stButton>button {
            border-radius: 6px !important;
            font-weight: 500 !important;
        }
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 3. Startup Database Guarantee
    try:
        init_db(DEFAULT_DB_PATH, reset=False)
    except Exception as e:
        st.error(f"Failed to initialize SQLite database at {DEFAULT_DB_PATH}: {e}")

    # 4. Session State Initialization
    if st.session_state.pop("clear_pending_schema", False):
        st.session_state["pending_schema_sql"] = ""
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "question_input" not in st.session_state:
        st.session_state["question_input"] = ""
    if "pending_schema_sql" not in st.session_state:
        st.session_state["pending_schema_sql"] = ""
    if "schema_prompt_input" not in st.session_state:
        st.session_state["schema_prompt_input"] = ""
    if "schema_exec_msg" not in st.session_state:
        st.session_state["schema_exec_msg"] = None

    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.subheader("Configuration & Schema")

        # Section 1 — LLM Provider
        st.markdown("### 1. LLM Provider")
        provider_choice = st.selectbox("Provider", ["Ollama", "Google Gemini"], key="llm_provider")
        provider = "gemini" if "gemini" in provider_choice.lower() else "ollama"

        selected_model = DEFAULT_MODEL
        ollama_url = DEFAULT_OLLAMA_URL
        api_key_val = None
        low_end_cpu = False

        if provider == "gemini":
            st.markdown("#### Google Gemini Config")
            api_key_input = st.text_input("Gemini API Key", type="password", placeholder="AIzaSy...", key="gemini_api_key_input")
            key, src = load_api_key(api_key_input)
            api_key_val = key
            if key:
                st.success(f"API key loaded from {src}")
            else:
                st.warning("No valid API key loaded. Enter key or set GEMINI_API_KEY env / .env.")

            if st.button("Test API Key", key="test_api_key_btn"):
                if not key:
                    st.error("Please provide an API key to test.")
                else:
                    valid, msg = validate_gemini_api_key(key)
                    if valid:
                        st.success(msg)
                    else:
                        st.error(msg)

            selected_model = st.selectbox("Gemini Model", GEMINI_MODELS, key="gemini_model_select")
        else:
            st.markdown("#### Local Ollama Config")
            ollama_url = st.text_input("Ollama URL", value=DEFAULT_OLLAMA_URL, key="ollama_url_input")
            low_end_cpu = st.checkbox("Low-End CPU Mode", value=False, help="Optimizes CPU threads and context window for weaker hardware.", key="low_end_cpu_checkbox")

            local_models, ollama_err = list_local_models(ollama_url)
            if not ollama_err and local_models:
                st.success(f"Ollama connected ({len(local_models)} models)")
                default_index = _choose_default_model_index(local_models)
                selected_model = st.selectbox("Model", local_models, index=default_index, key="ollama_model_select")
            else:
                st.warning("Ollama offline or no models found.")
                selected_model = st.text_input("Model", value=DEFAULT_MODEL, key="ollama_model_text")

        st.divider()

        # Section 2 — Database Schema (Tables)
        st.markdown("### 2. Database Schema")
        try:
            tables_info = get_tables_info(DEFAULT_DB_PATH)
            for t in tables_info:
                with st.expander(f"{t['name']} ({t['row_count']} rows)"):
                    for col in t["columns"]:
                        pk_str = " (PK)" if col["pk"] else ""
                        st.markdown(f"- `{col['name']}` (`{col['type']}`){pk_str}")
        except Exception as e:
            st.error(f"Error loading schema: {e}")

        # Section 3 — Triggers
        st.markdown("### 3. Triggers")
        try:
            triggers_info = get_triggers_info(DEFAULT_DB_PATH)
            if not triggers_info:
                st.caption("No custom triggers defined.")
            else:
                for tr in triggers_info:
                    with st.expander(f"{tr['name']} (on {tr['table']})"):
                        st.code(tr["sql"], language="sql")
        except Exception as e:
            st.error(f"Error loading triggers: {e}")

        # Section 4 — Backend SQL (DDL)
        st.markdown("### 4. Backend SQL (DDL)")
        with st.expander("View Full Backend SQL (DDL)"):
            try:
                full_ddl = get_full_ddl(DEFAULT_DB_PATH)
                st.code(full_ddl, language="sql")
            except Exception as e:
                st.error(f"Error loading full DDL: {e}")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Reset Sample Database", key="reset_db_btn"):
            try:
                init_db(DEFAULT_DB_PATH, reset=True)
                st.session_state["pending_schema_sql"] = ""
                st.session_state["schema_exec_msg"] = "Sample database reset successfully!"
                st.success("Sample database reset successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to reset database: {e}")

    # ==================== MAIN WORKSPACE ====================
    st.title("Chat with SQL")
    st.markdown(
        "<p style='color: #888888; margin-top: -10px; margin-bottom: 20px;'>"
        "Ask questions in plain English or draft schema changes securely. Powered by local Ollama & SQLite."
        "</p>",
        unsafe_allow_html=True,
    )

    if st.session_state.get("schema_exec_msg"):
        st.success(st.session_state["schema_exec_msg"])
        st.session_state["schema_exec_msg"] = None

    tab_chat, tab_schema = st.tabs(["Ask Database (Read-Only)", "Schema & Triggers (HITL)"])

    # ==================== TAB 1: ASK DATABASE (READ-ONLY) ====================
    with tab_chat:
        examples = [
            "Show me the top 3 students with attendance below 75%",
            "Which department has the highest unpaid fees?",
            "List the top 5 CSE students by marks",
            "Show average marks and attendance for each branch",
        ]

        cols = st.columns(2)
        for idx, ex in enumerate(examples):
            col_idx = idx % 2
            with cols[col_idx]:
                if st.button(ex, key=f"ex_btn_{idx}"):
                    st.session_state["question_input"] = ex
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        with st.form(key="query_form"):
            st.text_input(
                "Ask a question about the college database",
                key="question_input",
                placeholder="e.g., Show me the top 3 students with attendance below 75%",
            )
            submitted = st.form_submit_button("Ask & Run Query", type="primary")

        if submitted:
            question = st.session_state.get("question_input", "").strip()
            if not question:
                st.warning("Please enter a question.")
            else:
                sql = ""
                initial_sql = ""
                repaired = False
                result_obj = None
                error_msg = None

                with st.spinner("Generating and executing SQL..."):
                    try:
                        schema_text = get_schema_summary(DEFAULT_DB_PATH)
                        sql = generate_sql(
                            question=question,
                            schema_text=schema_text,
                            model=selected_model,
                            base_url=ollama_url,
                            provider=provider,
                            api_key=api_key_val,
                            low_end_cpu=low_end_cpu,
                        )
                        try:
                            result_obj = execute_read_only_query(DEFAULT_DB_PATH, sql)
                        except QueryExecutionError as qee:
                            initial_sql = sql
                            repaired_sql = repair_sql(
                                question=question,
                                schema_text=schema_text,
                                bad_sql=sql,
                                error_message=str(qee),
                                model=selected_model,
                                base_url=ollama_url,
                                provider=provider,
                                api_key=api_key_val,
                                low_end_cpu=low_end_cpu,
                            )
                            sql = repaired_sql
                            repaired = True
                            result_obj = execute_read_only_query(DEFAULT_DB_PATH, sql)
                    except (OllamaError, GeminiError) as llm_err:
                        error_msg = f"LLM Error: {llm_err}"
                    except UnsafeQueryError as uqe:
                        error_msg = f"Blocked by SQL Safety Guard: {uqe}"
                    except QueryExecutionError as qee:
                        error_msg = f"Query Execution Error: {qee}"
                    except Exception as e:
                        error_msg = f"Unexpected Error: {e}"

                history_item = {
                    "question": question,
                    "sql": sql,
                    "initial_sql": initial_sql if repaired else None,
                    "repaired": repaired,
                    "result": result_obj,
                    "error": error_msg,
                }
                st.session_state["history"].insert(0, history_item)
                st.session_state["history"] = st.session_state["history"][:10]

        if st.session_state["history"]:
            latest = st.session_state["history"][0]
            st.markdown("### Latest Query Result")
            st.markdown(f"**Question:** `{latest['question']}`")

            if latest.get("repaired"):
                st.caption("Auto-repaired SQL after initial schema/syntax error.")
                if latest.get("initial_sql"):
                    with st.expander("View Initial Broken SQL"):
                        st.code(latest["initial_sql"], language="sql")

            if latest["sql"]:
                st.markdown("**Generated SQL:**")
                st.code(latest["sql"], language="sql")

            if latest["error"]:
                st.error(latest["error"])
            elif latest["result"]:
                res = latest["result"]
                trunc_notice = " · (Results truncated)" if res.truncated else ""
                st.markdown(
                    f"<p style='font-size: 0.85rem; color: #888888;'>"
                    f"Read-only verified · {res.row_count} row(s) · {res.elapsed_ms} ms{trunc_notice}"
                    f"</p>",
                    unsafe_allow_html=True,
                )

                if res.row_count == 0:
                    st.info("Query returned 0 rows.")
                else:
                    data_dicts = [
                        dict(zip(res.columns, row)) for row in res.rows
                    ]
                    st.dataframe(data_dicts, use_container_width=True, hide_index=True)

                    csv_buffer = io.StringIO()
                    writer = csv.writer(csv_buffer)
                    writer.writerow(res.columns)
                    writer.writerows(res.rows)
                    csv_data = csv_buffer.getvalue()

                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name="query_result.csv",
                        mime="text/csv",
                    )

        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("Test SQL Safety Guard (Direct Read-Only Check)"):
            st.markdown(
                "Test the strict read-only safety validator directly without calling Ollama."
            )
            test_sql = st.text_area(
                "Raw SQL Input", value="DROP TABLE students;", key="test_safety_sql"
            )
            if st.button("Test Safety & Execute", key="test_safety_exec_btn"):
                try:
                    test_res = execute_read_only_query(DEFAULT_DB_PATH, test_sql)
                    st.success("Query passed safety check and executed successfully!")
                    st.markdown(
                        f"<p style='font-size: 0.85rem; color: #888888;'>Read-only verified · {test_res.row_count} row(s) · {test_res.elapsed_ms} ms</p>",
                        unsafe_allow_html=True,
                    )
                    if test_res.row_count == 0:
                        st.info("Query returned 0 rows.")
                    else:
                        test_dicts = [
                            dict(zip(test_res.columns, row)) for row in test_res.rows
                        ]
                        st.dataframe(test_dicts, use_container_width=True, hide_index=True)
                except UnsafeQueryError as uqe:
                    st.error(f"Blocked by SQL Safety Guard: {uqe}")
                except Exception as e:
                    st.error(f"Execution Error: {e}")

        if len(st.session_state["history"]) > 0:
            with st.expander("Recent Queries"):
                for idx, hist in enumerate(st.session_state["history"]):
                    st.markdown(f"**{idx+1}. {hist['question']}**")
                    if hist.get("repaired"):
                        st.caption("Auto-repaired SQL query.")
                    if hist["sql"]:
                        st.code(hist["sql"], language="sql")
                    if hist["error"]:
                        st.error(hist["error"])
                    st.divider()

    # ==================== TAB 2: SCHEMA & TRIGGERS (HITL) ====================
    with tab_schema:
        st.markdown(
            "Draft new tables, triggers, or inserts in plain English. Nothing executes until you inspect and approve the SQL."
        )

        hitl_starters = [
            "Create a courses table linked to departments",
            "Create a trigger blocking paid_fee > total_fee",
            "Create a fee_audit_log table for fee changes",
        ]

        hitl_cols = st.columns(3)
        for idx, starter in enumerate(hitl_starters):
            with hitl_cols[idx]:
                if st.button(starter, key=f"hitl_btn_{idx}"):
                    st.session_state["schema_prompt_input"] = starter
                    with st.spinner("Drafting schema SQL..."):
                        try:
                            schema_text = get_schema_summary(DEFAULT_DB_PATH)
                            draft_sql = generate_schema_sql(
                                request=starter,
                                schema_text=schema_text,
                                model=selected_model,
                                base_url=ollama_url,
                                provider=provider,
                                api_key=api_key_val,
                                low_end_cpu=low_end_cpu,
                            )
                            st.session_state["pending_schema_sql"] = draft_sql
                        except Exception as e:
                            st.error(f"Failed to generate schema SQL: {e}")
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        with st.form(key="schema_draft_form"):
            st.text_input(
                "Describe the table, trigger, or insert to add",
                key="schema_prompt_input",
                placeholder="e.g., Create a scholarship table with student_id and amount",
            )
            draft_submitted = st.form_submit_button("Draft SQL (Preview Only)", type="primary")

        if draft_submitted:
            prompt_req = st.session_state.get("schema_prompt_input", "").strip()
            if not prompt_req:
                st.warning("Please enter a description for the schema change.")
            else:
                with st.spinner("Drafting SQL preview..."):
                    try:
                        schema_text = get_schema_summary(DEFAULT_DB_PATH)
                        draft_sql = generate_schema_sql(
                            request=prompt_req,
                            schema_text=schema_text,
                            model=selected_model,
                            base_url=ollama_url,
                            provider=provider,
                            api_key=api_key_val,
                            low_end_cpu=low_end_cpu,
                        )
                        st.session_state["pending_schema_sql"] = draft_sql
                    except (OllamaError, GeminiError) as llm_err:
                        st.error(f"LLM Error: {llm_err}")
                    except Exception as e:
                        st.error(f"Error drafting SQL: {e}")

        if st.session_state.get("pending_schema_sql"):
            st.markdown("### SQL Preview (Review Before Executing)")
            edited_draft = st.text_area(
                "Edit SQL Preview",
                key="pending_schema_sql",
                height=150,
            )

            is_safe_schema, schema_reason = validate_schema_sql(edited_draft)
            if is_safe_schema:
                st.success(f"Safety Check Passed: {schema_reason}")
            else:
                st.error(f"Safety Check Blocked: {schema_reason}")

            col_exec, col_discard = st.columns(2)
            with col_exec:
                approve_clicked = st.button(
                    "Approve & Execute SQL",
                    type="primary",
                    disabled=not is_safe_schema or not edited_draft.strip(),
                    key="approve_schema_btn",
                )
            with col_discard:
                discard_clicked = st.button("Discard Draft", key="discard_schema_btn")

            if approve_clicked:
                with st.spinner("Executing approved schema statement..."):
                    try:
                        exec_res_msg = execute_schema_sql(DEFAULT_DB_PATH, edited_draft)
                        st.session_state["schema_exec_msg"] = exec_res_msg
                        st.session_state["clear_pending_schema"] = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Execution Error: {e}")

            if discard_clicked:
                st.session_state["clear_pending_schema"] = True
                st.session_state["schema_exec_msg"] = "Schema draft discarded."
                st.rerun()

    return 0


if __name__ == "__main__":
    main()
