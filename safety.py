"""
SQL Read-Only Security Guardrail for Chat-with-SQL.

Provides strict, deterministic regex and scanner-based SQL validation to ensure
only read-only queries (SELECT and read-only WITH CTEs) are executed.
"""

import re

# Comprehensive set of forbidden SQL keywords representing data modification,
# schema alteration, transaction control, administrative tasks, and execution.
FORBIDDEN_KEYWORDS: frozenset[str] = frozenset({
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "REPLACE",
    "TRUNCATE",
    "ATTACH",
    "DETACH",
    "PRAGMA",
    "VACUUM",
    "REINDEX",
    "BEGIN",
    "COMMIT",
    "ROLLBACK",
    "SAVEPOINT",
    "RELEASE",
    "GRANT",
    "REVOKE",
    "EXEC",
    "EXECUTE",
    "MERGE",
})


def strip_sql_comments(sql: str) -> str:
    """
    Character-by-character scanner that strips SQL line comments (-- ...)
    and block comments (/* ... */) outside string literals, replacing them
    with a single space. Preserves single-quoted ('...' with '' escapes) and
    double-quoted ("..." with "" escapes) string literals intact.
    
    Raises ValueError if there is an unclosed block comment or unclosed quote.
    """
    if not isinstance(sql, str):
        raise TypeError("SQL input must be a string.")

    result: list[str] = []
    i = 0
    n = len(sql)
    
    in_single_quote = False
    in_double_quote = False
    in_line_comment = False
    in_block_comment = False

    while i < n:
        char = sql[i]
        
        # Handle line comments (-- ...)
        if in_line_comment:
            if char == '\n':
                in_line_comment = False
                result.append('\n')
            i += 1
            continue

        # Handle block comments (/* ... */)
        if in_block_comment:
            if char == '*' and i + 1 < n and sql[i + 1] == '/':
                in_block_comment = False
                i += 2
                result.append(' ')  # Replace block comment with space
            else:
                i += 1
            continue

        # Handle single-quoted string literals ('...' with escaped '')
        if in_single_quote:
            result.append(char)
            if char == "'":
                # Check for escaped single quote ('')
                if i + 1 < n and sql[i + 1] == "'":
                    result.append("'")
                    i += 2
                else:
                    in_single_quote = False
                    i += 1
            else:
                i += 1
            continue

        # Handle double-quoted identifiers/literals ("..." with escaped "")
        if in_double_quote:
            result.append(char)
            if char == '"':
                # Check for escaped double quote ("")
                if i + 1 < n and sql[i + 1] == '"':
                    result.append('"')
                    i += 2
                else:
                    in_double_quote = False
                    i += 1
            else:
                i += 1
            continue

        # Not inside any quote or comment - check for start of comments or quotes
        # Check line comment: --
        if char == '-' and i + 1 < n and sql[i + 1] == '-':
            in_line_comment = True
            i += 2
            result.append(' ')
            continue

        # Check block comment: /*
        if char == '/' and i + 1 < n and sql[i + 1] == '*':
            in_block_comment = True
            i += 2
            continue

        # Check single quote start
        if char == "'":
            in_single_quote = True
            result.append(char)
            i += 1
            continue

        # Check double quote start
        if char == '"':
            in_double_quote = True
            result.append(char)
            i += 1
            continue

        # Normal character outside quotes and comments
        result.append(char)
        i += 1

    # Check for unclosed structures
    if in_block_comment:
        raise ValueError("Unclosed block comment /* ...")
    if in_single_quote:
        raise ValueError("Unclosed single-quoted string literal.")
    if in_double_quote:
        raise ValueError("Unclosed double-quoted string literal.")

    return "".join(result)


def _mask_string_literals(sql: str) -> str:
    """
    Replaces contents inside single-quoted string literals with ''
    so keyword checks and semicolon checks ignore text inside values.
    Example: "WHERE name = 'Update; Drop'" -> "WHERE name = ''"
    """
    result: list[str] = []
    i = 0
    n = len(sql)
    in_quote = False

    while i < n:
        char = sql[i]
        if in_quote:
            if char == "'":
                if i + 1 < n and sql[i + 1] == "'":
                    i += 2  # skip escaped quote
                else:
                    in_quote = False
                    result.append("''")
                    i += 1
            else:
                i += 1
            continue
        
        if char == "'":
            in_quote = True
            i += 1
        else:
            result.append(char)
            i += 1

    if in_quote:
        raise ValueError("Unclosed single-quoted string literal.")

    return "".join(result)


def validate_read_only_sql(sql: str) -> tuple[bool, str]:
    """
    Validates that a given SQL query string is strictly read-only.
    
    Returns:
        (True, "OK") if valid and safe.
        (False, reason_string) if unsafe or invalid.
    """
    # 1. Reject None, non-string, or empty/whitespace-only queries
    if not isinstance(sql, str) or not sql.strip():
        return False, "SQL query cannot be empty or non-string."

    # 2 & 3. Strip comments and check for unclosed quotes/block comments
    try:
        cleaned_comments = strip_sql_comments(sql)
    except ValueError as e:
        return False, f"Invalid SQL syntax: {e}"

    cleaned = cleaned_comments.strip()
    if not cleaned:
        return False, "SQL query cannot be empty after stripping comments."

    # 4. Mask string literals
    try:
        masked = _mask_string_literals(cleaned)
    except ValueError as e:
        return False, f"Invalid SQL syntax: {e}"

    # 5. Handle optional trailing semicolon
    # Remove at most one trailing semicolon after stripping whitespace
    query_body = masked.strip()
    if query_body.endswith(";"):
        query_body = query_body[:-1].strip()

    # If any other semicolon remains, it's a multi-statement query
    if ";" in query_body:
        return False, "Multi-statement SQL queries are not allowed."

    # 6. Extract whole-word tokens
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query_body)
    if not tokens:
        return False, "No valid SQL tokens found in query."

    # 7. Require query to start with SELECT or WITH
    first_token_upper = tokens[0].upper()
    if first_token_upper not in {"SELECT", "WITH"}:
        return False, "Only SELECT (or read-only WITH CTE) queries are permitted."

    # 8. Check every token against forbidden keywords
    for token in tokens:
        token_upper = token.upper()
        if token_upper in FORBIDDEN_KEYWORDS:
            return False, f"Blocked unsafe SQL keyword: {token_upper}"

    return True, "OK"


# Destructive or administrative commands that are ALWAYS forbidden even in Schema Mode.
SCHEMA_FORBIDDEN_KEYWORDS: frozenset[str] = frozenset({
    "DROP",
    "DELETE",
    "TRUNCATE",
    "ATTACH",
    "DETACH",
    "PRAGMA",
    "VACUUM",
    "REINDEX",
    "ALTER",
    "GRANT",
    "REVOKE",
    "EXEC",
    "EXECUTE",
    "MERGE",
    "COMMIT",
    "ROLLBACK",
    "SAVEPOINT",
    "RELEASE",
})


def validate_schema_sql(sql: str) -> tuple[bool, str]:
    """
    Validates that a given SQL query string is safe for HITL Schema Mode
    (allowing single CREATE TABLE, CREATE VIEW, CREATE INDEX, CREATE TRIGGER,
    or INSERT INTO statements, while blocking destructive/admin commands and
    stacked statements).

    Returns:
        (True, reason_string) if valid and safe.
        (False, reason_string) if unsafe or invalid.
    """
    # 1. Reject None, non-string, or empty/whitespace-only queries
    if not isinstance(sql, str) or not sql.strip():
        return False, "SQL query cannot be empty or non-string."

    # 2. Strip comments via strip_sql_comments(sql)
    try:
        cleaned_comments = strip_sql_comments(sql)
    except ValueError as e:
        return False, f"Invalid SQL syntax: {e}"

    cleaned = cleaned_comments.strip()
    if not cleaned:
        return False, "SQL query cannot be empty after stripping comments."

    # Mask string literals via _mask_string_literals(cleaned)
    try:
        masked = _mask_string_literals(cleaned)
    except ValueError as e:
        return False, f"Invalid SQL syntax: {e}"

    # 3. Strip surrounding whitespace and at most one trailing `;`.
    query_body = masked.strip()
    if query_body.endswith(";"):
        query_body = query_body[:-1].strip()

    # 4. Extract whole-word tokens and uppercase them
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query_body)
    if not tokens:
        return False, "No valid SQL tokens found in query."
    tokens_upper = [t.upper() for t in tokens]

    # 5. Check EVERY token against SCHEMA_FORBIDDEN_KEYWORDS
    for token_upper in tokens_upper:
        if token_upper in SCHEMA_FORBIDDEN_KEYWORDS:
            return False, f"Blocked destructive/admin SQL keyword in Schema Mode: {token_upper}"

    # 6. Inspect allowed statement shapes
    first_token = tokens_upper[0]

    if first_token == "CREATE":
        if len(tokens_upper) < 2:
            return False, "Incomplete CREATE statement."

        # Resolve the object type immediately after CREATE [TEMP|TEMPORARY|UNIQUE]
        obj_idx = 1
        if tokens_upper[obj_idx] in {"TEMP", "TEMPORARY", "UNIQUE"}:
            obj_idx += 1
        if obj_idx >= len(tokens_upper):
            return False, "Incomplete CREATE statement."

        obj_keyword = tokens_upper[obj_idx]

        # Check if it is a TRIGGER (Case A)
        if obj_keyword == "TRIGGER":
            # Must contain BEGIN and end with END (tokens_upper[-1] == "END")
            if "BEGIN" not in tokens_upper or tokens_upper[-1] != "END":
                return False, "Schema Mode CREATE TRIGGER must contain BEGIN and end with END."

            begin_indices = [i for i, t in enumerate(tokens_upper) if t == "BEGIN"]
            end_indices = [i for i, t in enumerate(tokens_upper) if t == "END"]
            if not begin_indices or not end_indices or begin_indices[0] >= end_indices[-1]:
                return False, "Schema Mode CREATE TRIGGER has invalid BEGIN/END structure."

            # Ensure there is NO semicolon BEFORE BEGIN
            begin_match = re.search(r"\bBEGIN\b", query_body, re.IGNORECASE)
            if begin_match:
                begin_pos = begin_match.start()
                if ";" in query_body[:begin_pos]:
                    return False, "Multi-statement queries (semicolon before BEGIN) are not allowed in Schema Mode."

            # Ensure NO tokens after END
            end_matches = list(re.finditer(r"\bEND\b", query_body, re.IGNORECASE))
            if end_matches:
                last_end = end_matches[-1]
                after_end = query_body[last_end.end():].strip()
                if after_end:
                    return False, "Trailing statements or tokens after END are not allowed in Schema Mode."

            return True, "OK (CREATE TRIGGER)"

        elif obj_keyword in {"TABLE", "VIEW", "INDEX"}:
            # Case B — CREATE TABLE, CREATE VIEW, CREATE INDEX
            if ";" in query_body:
                return False, "Multi-statement SQL queries are not allowed in Schema Mode."
            if "BEGIN" in tokens_upper:
                return False, "BEGIN is not allowed in this statement type in Schema Mode."

            return True, f"OK (CREATE {obj_keyword})"

        else:
            return False, "Schema Mode only allows single CREATE TABLE, CREATE VIEW, CREATE INDEX, CREATE TRIGGER, or INSERT INTO statements."

    elif first_token == "INSERT":
        # Case C — INSERT INTO ...
        if len(tokens_upper) < 2 or tokens_upper[1] != "INTO":
            return False, "Schema Mode only allows INSERT INTO statements."
        if ";" in query_body:
            return False, "Multi-statement SQL queries are not allowed in Schema Mode."
        if "BEGIN" in tokens_upper:
            return False, "BEGIN is not allowed in INSERT statements in Schema Mode."
        return True, "OK (INSERT)"

    else:
        return False, "Schema Mode only allows single CREATE TABLE, CREATE VIEW, CREATE INDEX, CREATE TRIGGER, or INSERT INTO statements."
