"""
Unit Tests for Database Execution and Schema Introspection Layer.
Uses pytest / unittest with temporary database fixtures.
"""

import unittest
from pathlib import Path
import sqlite3

from seed_db import init_db
from database import (
    UnsafeQueryError,
    QueryExecutionError,
    QueryResult,
    get_tables_info,
    get_schema_summary,
    get_triggers_info,
    get_full_ddl,
    execute_schema_sql,
    execute_read_only_query,
)


class TestDatabaseLayer(unittest.TestCase):
    def setUp(self):
        # Create an isolated temporary database for each test
        self.tmp_dir = Path("D:/chat-with-sql/tests/temp_db")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.tmp_dir / "test_college.db"
        init_db(self.db_path, reset=True)

    def tearDown(self):
        # Clean up temporary database file
        import gc
        gc.collect()
        if self.db_path.exists():
            for _ in range(3):
                try:
                    self.db_path.unlink()
                    break
                except PermissionError:
                    import time
                    time.sleep(0.05)
        if self.tmp_dir.exists():
            try:
                self.tmp_dir.rmdir()
            except OSError:
                pass

    def test_get_tables_info(self):
        tables_info = get_tables_info(self.db_path)
        # Expect exactly 3 tables ordered by name: departments, fee_records, students
        table_names = [t["name"] for t in tables_info]
        self.assertEqual(table_names, ["departments", "fee_records", "students"])

        # Check exact row counts
        counts_map = {t["name"]: t["row_count"] for t in tables_info}
        self.assertEqual(counts_map["departments"], 5)
        self.assertEqual(counts_map["fee_records"], 30)
        self.assertEqual(counts_map["students"], 30)

        # Verify columns exist for students
        students_table = next(t for t in tables_info if t["name"] == "students")
        col_names = [c["name"] for c in students_table["columns"]]
        self.assertIn("roll_no", col_names)
        self.assertIn("branch", col_names)
        self.assertIn("marks", col_names)
        self.assertIn("attendance_pct", col_names)

    def test_get_schema_summary(self):
        summary = get_schema_summary(self.db_path)
        self.assertIn("CREATE TABLE departments", summary)
        self.assertIn("CREATE TABLE students", summary)
        self.assertIn("CREATE TABLE fee_records", summary)
        # Check categorical hints (Paid or Partial or Unpaid)
        self.assertIn("Paid", summary)

    def test_whatsapp_pitch_query_1(self):
        # Query 1: Top 5 CSE students by marks descending
        sql = "SELECT name, marks FROM students WHERE branch = 'CSE' ORDER BY marks DESC LIMIT 5"
        result = execute_read_only_query(self.db_path, sql)
        self.assertEqual(result.row_count, 5)
        self.assertEqual(result.columns, ["name", "marks"])
        self.assertFalse(result.truncated)
        # Verify sorting descending
        marks = [row[1] for row in result.rows]
        self.assertEqual(marks, sorted(marks, reverse=True))

    def test_whatsapp_pitch_query_2(self):
        # Query 2: Students with attendance < 75 sorted by marks DESC limit 3
        sql = "SELECT name, branch, attendance_pct, marks FROM students WHERE attendance_pct < 75 ORDER BY marks DESC LIMIT 3"
        result = execute_read_only_query(self.db_path, sql)
        self.assertEqual(result.row_count, 3)
        self.assertEqual(result.columns, ["name", "branch", "attendance_pct", "marks"])
        for row in result.rows:
            self.assertLess(row[2], 75.0)

    def test_whatsapp_pitch_query_3(self):
        # Query 3: Department with highest unpaid fees JOIN query
        sql = """
            SELECT d.dept_name, SUM(f.unpaid_fee) as total_unpaid
            FROM departments d
            JOIN students s ON d.dept_id = s.dept_id
            JOIN fee_records f ON s.student_id = f.student_id
            GROUP BY d.dept_id
            ORDER BY total_unpaid DESC
            LIMIT 1
        """
        result = execute_read_only_query(self.db_path, sql)
        self.assertEqual(result.row_count, 1)
        self.assertEqual(result.columns, ["dept_name", "total_unpaid"])
        # Mechanical (ME) should be top unpaid
        self.assertEqual(result.rows[0][0], "Mechanical Engineering")
        self.assertGreater(result.rows[0][1], 0.0)

    def test_unsafe_query_rejection(self):
        sql = "DROP TABLE students;"
        with self.assertRaises(UnsafeQueryError):
            execute_read_only_query(self.db_path, sql)

        # Verify table still exists and has 30 rows
        tables_info = get_tables_info(self.db_path)
        students_table = next(t for t in tables_info if t["name"] == "students")
        self.assertEqual(students_table["row_count"], 30)

    def test_invalid_sql_syntax_raises_error(self):
        sql = "SELECT * FROM non_existent_table_xyz;"
        with self.assertRaises(QueryExecutionError):
            execute_read_only_query(self.db_path, sql)

    def test_max_rows_truncation(self):
        # Fetch with max_rows = 5 when table has 30 rows
        sql = "SELECT * FROM students;"
        result = execute_read_only_query(self.db_path, sql, max_rows=5)
        self.assertTrue(result.truncated)
        self.assertEqual(result.row_count, 5)
        self.assertEqual(len(result.rows), 5)

    def test_fee_statuses_and_unpaid_bugfix(self):
        # Verify all three statuses ('Paid', 'Partial', 'Unpaid') exist in fee_records
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT payment_status FROM fee_records;")
        statuses = {row[0] for row in cursor.fetchall()}
        self.assertEqual(statuses, {"Paid", "Partial", "Unpaid"})

        # Verify at least 4-5 students have paid_fee == 0.0 (Unpaid)
        cursor.execute("SELECT COUNT(*) FROM fee_records WHERE paid_fee = 0.0;")
        zero_paid_count = cursor.fetchone()[0]
        self.assertGreaterEqual(zero_paid_count, 4)

        # Verify ME has the highest total unpaid fee
        cursor.execute("""
            SELECT d.dept_code, SUM(f.unpaid_fee) as total_unpaid
            FROM departments d
            JOIN students s ON d.dept_id = s.dept_id
            JOIN fee_records f ON s.student_id = f.student_id
            GROUP BY d.dept_code
            ORDER BY total_unpaid DESC
            LIMIT 1;
        """)
        top_dept = cursor.fetchone()
        self.assertEqual(top_dept[0], "ME")
        conn.close()

    def test_new_database_upgrades_and_functions(self):
        # 1. seed_db creates 'Paid', 'Partial', and 'Unpaid' fee statuses, and ME has highest total unpaid_fee
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT payment_status FROM fee_records;")
        statuses = {row[0] for row in cursor.fetchall()}
        self.assertEqual(statuses, {"Paid", "Partial", "Unpaid"})
        conn.close()

        # 2. Built-in trg_validate_student_scores trigger is returned by get_triggers_info
        triggers = get_triggers_info(self.db_path)
        trigger_names = [t["name"] for t in triggers]
        self.assertIn("trg_validate_student_scores", trigger_names)

        # And aborts invalid INSERT (marks = 150) via execute_schema_sql raising QueryExecutionError
        invalid_insert_sql = "INSERT INTO students (roll_no, name, branch, dept_id, semester, marks, attendance_pct) VALUES ('CSE888', 'Test Bad', 'CSE', 1, 4, 150.0, 90.0);"
        with self.assertRaises(QueryExecutionError):
            execute_schema_sql(self.db_path, invalid_insert_sql)

        # 3. get_schema_summary includes -- JOIN PATH: lines
        schema_summary = get_schema_summary(self.db_path)
        self.assertIn("-- JOIN PATH:", schema_summary)

        # 4. get_full_ddl returns both CREATE TABLE and CREATE TRIGGER statements
        full_ddl = get_full_ddl(self.db_path)
        self.assertIn("CREATE TABLE", full_ddl)
        self.assertIn("CREATE TRIGGER", full_ddl)

        # 5. execute_schema_sql successfully creates a new courses table and custom trigger on db_path,
        # then get_tables_info and get_triggers_info immediately reflect them, while DROP TABLE students raises UnsafeQueryError.
        create_course_sql = "CREATE TABLE courses (course_id INTEGER PRIMARY KEY AUTOINCREMENT, course_name TEXT NOT NULL);"
        res_create = execute_schema_sql(self.db_path, create_course_sql)
        self.assertIn("CREATE TABLE", res_create)

        create_trigger_sql = """
        CREATE TRIGGER trg_validate_course_name
        BEFORE INSERT ON courses
        FOR EACH ROW
        WHEN NEW.course_name = ''
        BEGIN
            SELECT RAISE(ABORT, 'Course name cannot be empty');
        END;
        """
        res_trg = execute_schema_sql(self.db_path, create_trigger_sql)
        self.assertIn("CREATE TRIGGER", res_trg)

        # Verify tables reflect courses
        tables = get_tables_info(self.db_path)
        self.assertIn("courses", [t["name"] for t in tables])

        # Verify triggers reflect trg_validate_course_name
        trgs = get_triggers_info(self.db_path)
        self.assertIn("trg_validate_course_name", [t["name"] for t in trgs])

        # Verify DROP TABLE students raises UnsafeQueryError
        with self.assertRaises(UnsafeQueryError):
            execute_schema_sql(self.db_path, "DROP TABLE students;")

    def test_student_score_validation_trigger(self):
        # Verify trigger trg_validate_student_scores prevents invalid marks or attendance
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Test invalid marks > 100
        with self.assertRaises(sqlite3.IntegrityError):
            cursor.execute(
                "INSERT INTO students (roll_no, name, branch, dept_id, semester, marks, attendance_pct) VALUES (?, ?, ?, ?, ?, ?, ?);",
                ("CSE999", "Bad Marks", "CSE", 1, 4, 105.0, 80.0)
            )
            conn.commit()

        # Test invalid attendance < 0
        with self.assertRaises(sqlite3.IntegrityError):
            cursor.execute(
                "INSERT INTO students (roll_no, name, branch, dept_id, semester, marks, attendance_pct) VALUES (?, ?, ?, ?, ?, ?, ?);",
                ("CSE998", "Bad Attendance", "CSE", 1, 4, 85.0, -5.0)
            )
            conn.commit()

        conn.close()

    def test_new_database_upgrades_and_functions(self):
        # 1. seed_db creates 'Paid', 'Partial', and 'Unpaid' fee statuses, and ME has highest total unpaid_fee
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT payment_status FROM fee_records;")
        statuses = {row[0] for row in cursor.fetchall()}
        self.assertEqual(statuses, {"Paid", "Partial", "Unpaid"})
        conn.close()

        # 2. Built-in trg_validate_student_scores trigger is returned by get_triggers_info
        triggers = get_triggers_info(self.db_path)
        trigger_names = [t["name"] for t in triggers]
        self.assertIn("trg_validate_student_scores", trigger_names)

        # And aborts invalid INSERT (marks = 150) via execute_schema_sql raising QueryExecutionError
        invalid_insert_sql = "INSERT INTO students (roll_no, name, branch, dept_id, semester, marks, attendance_pct) VALUES ('CSE888', 'Test Bad', 'CSE', 1, 4, 150.0, 90.0);"
        with self.assertRaises(QueryExecutionError):
            execute_schema_sql(self.db_path, invalid_insert_sql)

        # 3. get_schema_summary includes -- JOIN PATH: lines
        schema_summary = get_schema_summary(self.db_path)
        self.assertIn("-- JOIN PATH:", schema_summary)

        # 4. get_full_ddl returns both CREATE TABLE and CREATE TRIGGER statements
        full_ddl = get_full_ddl(self.db_path)
        self.assertIn("CREATE TABLE", full_ddl)
        self.assertIn("CREATE TRIGGER", full_ddl)

        # 5. execute_schema_sql successfully creates a new courses table and custom trigger on db_path,
        # then get_tables_info and get_triggers_info immediately reflect them, while DROP TABLE students raises UnsafeQueryError.
        create_course_sql = "CREATE TABLE courses (course_id INTEGER PRIMARY KEY AUTOINCREMENT, course_name TEXT NOT NULL);"
        res_create = execute_schema_sql(self.db_path, create_course_sql)
        self.assertIn("CREATE TABLE", res_create)

        create_trigger_sql = """
        CREATE TRIGGER trg_validate_course_name
        BEFORE INSERT ON courses
        FOR EACH ROW
        WHEN NEW.course_name = ''
        BEGIN
            SELECT RAISE(ABORT, 'Course name cannot be empty');
        END;
        """
        res_trg = execute_schema_sql(self.db_path, create_trigger_sql)
        self.assertIn("CREATE TRIGGER", res_trg)

        # Verify tables reflect courses
        tables = get_tables_info(self.db_path)
        self.assertIn("courses", [t["name"] for t in tables])

        # Verify triggers reflect trg_validate_course_name
        trgs = get_triggers_info(self.db_path)
        self.assertIn("trg_validate_course_name", [t["name"] for t in trgs])

        # Verify DROP TABLE students raises UnsafeQueryError
        with self.assertRaises(UnsafeQueryError):
            execute_schema_sql(self.db_path, "DROP TABLE students;")


if __name__ == "__main__":
    unittest.main()
