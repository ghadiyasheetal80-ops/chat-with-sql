"""
Database Execution and Schema Introspection Layer for Chat-with-SQL.
Provides safe read-only SQLite connections, schema inspection, and execution
guarded by validate_read_only_sql with timeout and row-count truncation.
"""

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import time

from safety import validate_read_only_sql, validate_schema_sql
from seed_db import DEFAULT_DB_PATH


class UnsafeQueryError(ValueError):
    """Raised when a query fails read-only safety validation."""
    pass


class QueryExecutionError(RuntimeError):
    """Raised when query execution encounters database or operational errors."""
    pass


@dataclass
class QueryResult:
    """Represents the result of executing a read-only SQL query."""
    columns: list[str]
    rows: list[tuple]
    row_count: int
    truncated: bool
    elapsed_ms: float


def _connect_read_only(db_path: Path | str) -> sqlite3.Connection:
    """
    Opens a SQLite connection in strict read-only URI mode and enforces query_only = ON.
    Raises FileNotFoundError if the database file does not exist.
    """
    path = Path(db_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Database file not found at: {path}")

    uri = f"{path.as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.execute("PRAGMA query_only = ON;")
    return conn


def get_tables_info(db_path: Path | str = DEFAULT_DB_PATH) -> list[dict]:
    """
    Returns a list of dicts for all user tables (excluding sqlite_%), ordered by table name:
    [
        {
            "name": table_name,
            "row_count": count,
            "columns": [{"name": col, "type": col_type, "pk": bool(pk)}]
        },
        ...
    ]
    """
    conn = _connect_read_only(db_path)
    try:
        cursor = conn.cursor()
        # Fetch all user tables ordered by name
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
        )
        tables = [row[0] for row in cursor.fetchall()]

        tables_info = []
        for table in tables:
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM \"{table}\";")
            row_count = cursor.fetchone()[0]

            # Get column info using PRAGMA table_info
            cursor.execute(f"PRAGMA table_info(\"{table}\");")
            # PRAGMA table_info columns: cid, name, type, notnull, dflt_value, pk
            cols_info = []
            for col_row in cursor.fetchall():
                col_name = col_row[1]
                col_type = col_row[2] or "TEXT"
                pk_flag = bool(col_row[5])
                cols_info.append({
                    "name": col_name,
                    "type": col_type,
                    "pk": pk_flag
                })

            tables_info.append({
                "name": table,
                "row_count": row_count,
                "columns": cols_info
            })

        return tables_info
    finally:
        conn.close()


def get_triggers_info(db_path: Path | str = DEFAULT_DB_PATH) -> list[dict]:
    """
    Queries sqlite_master for triggers ordered by name:
    [
        {"name": row[0], "table": row[1], "sql": row[2]},
        ...
    ]
    """
    conn = _connect_read_only(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name, tbl_name, sql FROM sqlite_master WHERE type='trigger' ORDER BY name;"
        )
        triggers = []
        for row in cursor.fetchall():
            triggers.append({
                "name": row[0],
                "table": row[1],
                "sql": row[2]
            })
        return triggers
    finally:
        conn.close()


def get_full_ddl(db_path: Path | str = DEFAULT_DB_PATH) -> str:
    """
    Returns clean formatted backend SQL containing all CREATE TABLE, CREATE VIEW,
    CREATE INDEX, and CREATE TRIGGER statements from sqlite_master
    (sql IS NOT NULL AND name NOT LIKE 'sqlite_%'), grouped clearly with
    -- Tables, -- Views / Indexes, -- Triggers comments and trailing semicolons.
    """
    conn = _connect_read_only(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY type, name;"
        )
        rows = cursor.fetchall()

        tables = []
        views_indexes = []
        triggers = []

        for obj_type, name, sql in rows:
            # Ensure sql statement ends with a semicolon
            stmt = sql.strip()
            if not stmt.endswith(";"):
                stmt += ";"

            if obj_type == "table":
                tables.append(stmt)
            elif obj_type in ("view", "index"):
                views_indexes.append(stmt)
            elif obj_type == "trigger":
                triggers.append(stmt)

        sections = []
        if tables:
            sections.append("-- Tables\n" + "\n\n".join(tables))
        if views_indexes:
            sections.append("-- Views / Indexes\n" + "\n\n".join(views_indexes))
        if triggers:
            sections.append("-- Triggers\n" + "\n\n".join(triggers))

        return "\n\n".join(sections)
    finally:
        conn.close()


def get_schema_summary(db_path: Path | str = DEFAULT_DB_PATH) -> str:
    """
    Queries sqlite_master for CREATE TABLE DDL statements of user tables,
    appends explicit -- JOIN PATH: hints by inspecting PRAGMA foreign_key_list,
    includes multi-hop join path hints, distinct-value hints, and triggers.
    """
    conn = _connect_read_only(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
        )
        table_rows = cursor.fetchall()
        ddls = [row[1] for row in table_rows if row[1]]
        table_names = [row[0] for row in table_rows]

        summary_parts = ["-- DATABASE SCHEMA & TABLES --"]
        summary_parts.extend(ddls)

        # Inspect foreign keys for each table to build JOIN PATH hints
        fk_map = {}  # table -> list of (from_col, ref_table, ref_col)
        for tname in table_names:
            cursor.execute(f"PRAGMA foreign_key_list(\"{tname}\");")
            # pragma fk_list columns: id, seq, table, from, to, on_update, on_delete, match
            fks = []
            for fk in cursor.fetchall():
                ref_table = fk[2]
                from_col = fk[3]
                to_col = fk[4]
                fks.append((from_col, ref_table, to_col))
                summary_parts.append(f"-- JOIN PATH: {tname}.{from_col} = {ref_table}.{to_col}")
            if fks:
                fk_map[tname] = fks

        # Check for multi-hop join paths (e.g. fee_records -> students -> departments)
        # Specifically, if fee_records links to students, and students links to departments
        if "fee_records" in table_names and "students" in table_names and "departments" in table_names:
            summary_parts.append(
                "-- MULTI-HOP JOIN: fee_records links to departments THROUGH students (departments -> students -> fee_records); fee_records has NO dept_id column."
            )

        # Append compact distinct-value hints for low-cardinality text columns if tables exist
        if "departments" in table_names:
            cursor.execute("SELECT DISTINCT dept_code FROM departments ORDER BY dept_code;")
            codes = [r[0] for r in cursor.fetchall()]
            summary_parts.append(f"-- Hint: departments.dept_code values: {codes}")

        if "students" in table_names:
            cursor.execute("SELECT DISTINCT branch FROM students ORDER BY branch;")
            branches = [r[0] for r in cursor.fetchall()]
            summary_parts.append(f"-- Hint: students.branch values: {branches}")

        if "fee_records" in table_names:
            cursor.execute("SELECT DISTINCT payment_status FROM fee_records ORDER BY payment_status;")
            statuses = [r[0] for r in cursor.fetchall()]
            summary_parts.append(f"-- Hint: fee_records.payment_status values: {statuses}")

        # Also list existing trigger names/DDLs if any exist
        cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='trigger' ORDER BY name;")
        triggers = cursor.fetchall()
        if triggers:
            summary_parts.append("-- TRIGGERS --")
            for t_name, t_sql in triggers:
                if t_sql:
                    summary_parts.append(t_sql)

        return "\n\n".join(summary_parts)
    finally:
        conn.close()


def execute_schema_sql(db_path: Path | str, sql: str) -> str:
    """
    Executes a schema modification SQL statement (CREATE TABLE, CREATE VIEW,
    CREATE INDEX, CREATE TRIGGER, or INSERT INTO) after validating with validate_schema_sql.
    """
    is_safe, reason = validate_schema_sql(sql)
    if not is_safe:
        raise UnsafeQueryError(reason)

    path = Path(db_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Database file not found at: {path}")

    conn = sqlite3.connect(path, timeout=5.0)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()

        # Determine rowcount or success message
        rowcount = cursor.rowcount
        if rowcount > 0 and sql.strip().upper().startswith("INSERT"):
            return f"Statement executed successfully ({reason}), affected rows: {rowcount}."
        else:
            return f"Statement executed successfully ({reason})."
    except (sqlite3.IntegrityError, sqlite3.OperationalError, sqlite3.DatabaseError) as e:
        conn.rollback()
        raise QueryExecutionError(f"SQLite error executing schema statement: {e}")
    finally:
        conn.close()


def execute_read_only_query(
    db_path: Path | str,
    sql: str,
    max_rows: int = 500,
    timeout_sec: float = 5.0
) -> QueryResult:
    """
    Executes a read-only SQL query with safety validation, timeout watchdog, and row truncation.
    """
    # 1. Validate SQL safety
    is_safe, reason = validate_read_only_sql(sql)
    if not is_safe:
        raise UnsafeQueryError(reason)

    start = time.perf_counter()
    deadline = start + timeout_sec

    def check_timeout() -> int:
        if time.perf_counter() > deadline:
            return 1
        return 0

    conn = _connect_read_only(db_path)
    try:
        # Install progress handler watchdog (invoked roughly every 1000 VM instructions)
        conn.set_progress_handler(check_timeout, 1000)

        cursor = conn.cursor()
        try:
            cursor.execute(sql)
        except sqlite3.OperationalError as e:
            err_msg = str(e).lower()
            if "interrupted" in err_msg or "timeout" in err_msg or time.perf_counter() > deadline:
                raise QueryExecutionError(f"Query execution timed out or was interrupted: {e}")
            else:
                raise QueryExecutionError(f"SQLite operational error during execution: {e}")
        except sqlite3.DatabaseError as e:
            raise QueryExecutionError(f"Database error during query execution: {e}")

        # Fetch max_rows + 1 to check truncation
        try:
            fetched = cursor.fetchmany(max_rows + 1)
        except sqlite3.OperationalError as e:
            if time.perf_counter() > deadline:
                raise QueryExecutionError(f"Query execution timed out during fetch: {e}")
            raise QueryExecutionError(f"SQLite operational error during fetch: {e}")
        except Exception as e:
            raise QueryExecutionError(f"Error fetching query results: {e}")

        truncated = len(fetched) > max_rows
        rows = fetched[:max_rows]
        row_count = len(rows)

        columns = [desc[0] for desc in (cursor.description or [])]
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)

        return QueryResult(
            columns=columns,
            rows=rows,
            row_count=row_count,
            truncated=truncated,
            elapsed_ms=elapsed_ms
        )
    finally:
        conn.close()
