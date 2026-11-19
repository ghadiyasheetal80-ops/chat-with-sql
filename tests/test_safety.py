"""
Comprehensive pytest unit tests for safety.py SQL guardrails.
"""

import pytest
from safety import (
    validate_read_only_sql,
    validate_schema_sql,
    strip_sql_comments,
    _mask_string_literals,
    FORBIDDEN_KEYWORDS,
    SCHEMA_FORBIDDEN_KEYWORDS,
)


def test_valid_simple_select():
    valid_sql = "SELECT id, name FROM users"
    ok, reason = validate_read_only_sql(valid_sql)
    assert ok is True
    assert reason == "OK"


def test_valid_complex_select():
    valid_sql = """
        SELECT department, COUNT(*), AVG(salary) 
        FROM employees 
        WHERE status = 'active' 
        GROUP BY department 
        HAVING COUNT(*) > 5 
        ORDER BY AVG(salary) DESC 
        LIMIT 10;
    """
    ok, reason = validate_read_only_sql(valid_sql)
    assert ok is True
    assert reason == "OK"


def test_valid_cte_with():
    valid_sql = """
        WITH regional_sales AS (
            SELECT region, SUM(amount) AS total_sales
            FROM orders
            GROUP BY region
        )
        SELECT region, total_sales 
        FROM regional_sales 
        WHERE total_sales > 10000
    """
    ok, reason = validate_read_only_sql(valid_sql)
    assert ok is True
    assert reason == "OK"


def test_forbidden_words_inside_string_literals():
    # Forbidden words ('Update', 'DROP') inside strings should be allowed
    sql = "SELECT name FROM students WHERE name = 'Update' OR branch = 'DROP';"
    ok, reason = validate_read_only_sql(sql)
    assert ok is True
    assert reason == "OK"


def test_column_names_with_forbidden_substrings():
    # Column names containing forbidden words or substrings (e.g. created_at, updated_at)
    sql = "SELECT created_at, updated_at FROM logs WHERE message = 'system update';"
    ok, reason = validate_read_only_sql(sql)
    assert ok is True
    assert reason == "OK"


@pytest.mark.parametrize(
    "unsafe_sql",
    [
        "DROP TABLE students;",
        "DELETE FROM students WHERE id = 1;",
        "UPDATE students SET marks = 100;",
        "INSERT INTO students (name) VALUES ('Alice');",
        "ALTER TABLE students ADD COLUMN age INT;",
        "CREATE TABLE temp (id INT);",
        "PRAGMA table_info(students);",
        "ATTACH DATABASE 'secret.db' AS secret;",
        "VACUUM;",
        "TRUNCATE TABLE logs;",
    ],
)
def test_blocked_destructive_statements(unsafe_sql):
    ok, reason = validate_read_only_sql(unsafe_sql)
    assert ok is False
    assert "Blocked unsafe SQL keyword" in reason or "Only SELECT" in reason


def test_blocked_stacked_multistatements():
    sql1 = "SELECT * FROM students; DROP TABLE students;"
    ok1, reason1 = validate_read_only_sql(sql1)
    assert ok1 is False
    assert "Multi-statement" in reason1

    sql2 = "SELECT 1; SELECT 2;"
    ok2, reason2 = validate_read_only_sql(sql2)
    assert ok2 is False
    assert "Multi-statement" in reason2


def test_blocked_comment_obfuscation():
    # Block comment hiding DROP
    sql1 = "/* comment */ DROP TABLE students"
    ok1, reason1 = validate_read_only_sql(sql1)
    assert ok1 is False

    # Line comment hiding DELETE
    sql2 = "-- intro\nDELETE FROM students"
    ok2, reason2 = validate_read_only_sql(sql2)
    assert ok2 is False

    # Trailing comment injection
    sql3 = "SELECT 1; --\nDROP TABLE students"
    ok3, reason3 = validate_read_only_sql(sql3)
    assert ok3 is False


def test_blocked_write_cte():
    # CTE followed by a delete statement
    sql = "WITH cte AS (SELECT 1) DELETE FROM students"
    ok, reason = validate_read_only_sql(sql)
    assert ok is False
    assert "Blocked unsafe SQL keyword" in reason or "DELETE" in reason


@pytest.mark.parametrize(
    "invalid_input",
    [
        "",
        "   ",
        None,
        12345,
        "/* unclosed comment",
        "SELECT * FROM students WHERE name = 'unclosed quote",
    ],
)
def test_blocked_invalid_inputs(invalid_input):
    ok, reason = validate_read_only_sql(invalid_input)
    assert ok is False


def test_strip_sql_comments_edge_cases():
    assert strip_sql_comments("SELECT 1 -- line comment") == "SELECT 1  "
    assert strip_sql_comments("SELECT /* block */ 1") == "SELECT   1"
    with pytest.raises(ValueError):
        strip_sql_comments("SELECT /* unclosed")
    with pytest.raises(ValueError):
        strip_sql_comments("SELECT 'unclosed")


def test_mask_string_literals():
    masked = _mask_string_literals("SELECT * FROM t WHERE col = 'hello''world; DROP TABLE t;'")
    assert "DROP" not in masked
    assert "SELECT" in masked


def test_schema_valid_create_and_insert():
    queries = [
        "CREATE TABLE courses (id INT PRIMARY KEY, title TEXT);",
        "CREATE VIEW v_courses AS SELECT * FROM courses;",
        "CREATE INDEX idx_courses_title ON courses(title);",
        "INSERT INTO courses (id, title) VALUES (1, 'Python');",
        "CREATE TEMP TABLE temp_t (a INT);",
        "CREATE UNIQUE INDEX idx_unique ON courses(id);",
    ]
    for q in queries:
        ok, reason = validate_schema_sql(q)
        assert ok is True, f"Expected success for {q}, got reason: {reason}"


def test_schema_valid_triggers():
    trig1 = "CREATE TRIGGER t_audit BEFORE INSERT ON students FOR EACH ROW WHEN NEW.marks > 100 BEGIN SELECT RAISE(ABORT, 'bad'); END;"
    ok, reason = validate_schema_sql(trig1)
    assert ok is True, reason

    trig2 = """
        CREATE TRIGGER t_log AFTER INSERT ON students
        BEGIN
            INSERT INTO audit_log (action) VALUES ('inserted');
            UPDATE stats SET count = count + 1;
        END;
    """
    ok, reason = validate_schema_sql(trig2)
    assert ok is True, reason


def test_schema_blocked_destructive_admin():
    blocked = [
        "DROP TABLE students;",
        "DELETE FROM students;",
        "ALTER TABLE students ADD COLUMN gpa REAL;",
        "PRAGMA table_info(students);",
        "ATTACH DATABASE 'secret.db' AS sec;",
        "TRUNCATE TABLE logs;",
        "VACUUM;",
        "REINDEX;",
        "COMMIT;",
        "ROLLBACK;",
    ]
    for q in blocked:
        ok, reason = validate_schema_sql(q)
        assert ok is False, f"Expected failure for destructive query: {q}"
        assert "Blocked destructive/admin SQL keyword" in reason


def test_schema_blocked_trigger_sneaking_destructive():
    trig_drop = "CREATE TRIGGER t1 AFTER INSERT ON students BEGIN DROP TABLE students; END;"
    ok, reason = validate_schema_sql(trig_drop)
    assert ok is False
    assert "Blocked destructive/admin SQL keyword" in reason

    trig_delete = "CREATE TRIGGER t2 BEFORE DELETE ON students BEGIN DELETE FROM logs; END;"
    ok, reason = validate_schema_sql(trig_delete)
    assert ok is False
    assert "Blocked destructive/admin SQL keyword" in reason


def test_schema_blocked_stacked_statements():
    stacked1 = "CREATE TABLE a (id INT); CREATE TABLE b (id INT);"
    ok, reason = validate_schema_sql(stacked1)
    assert ok is False
    assert "Multi-statement" in reason or "allowed" in reason

    stacked2 = "CREATE TRIGGER t1 AFTER INSERT ON students BEGIN SELECT 1; END; DROP TABLE students;"
    ok, reason = validate_schema_sql(stacked2)
    assert ok is False

    stacked3 = "CREATE TABLE x (id INT); DROP TABLE x;"
    ok, reason = validate_schema_sql(stacked3)
    assert ok is False


def test_schema_blocked_standalone_dml_and_select():
    unsupported = [
        "SELECT * FROM students;",
        "UPDATE students SET marks = 0;",
        "REPLACE INTO students (id) VALUES (1);",
        "CREATE DATABASE test;",
    ]
    for q in unsupported:
        ok, reason = validate_schema_sql(q)
        assert ok is False
        assert "Schema Mode only allows" in reason


def test_schema_invalid_inputs():
    for invalid in [None, "", "   ", 12345, "/* unclosed", "SELECT 'unclosed"]:
        ok, reason = validate_schema_sql(invalid)
        assert ok is False
