"""
Database Seeder for the Chat-with-SQL College Application.
Creates and populates a normalized SQLite database (college.db) with departments,
students, and fee records using Python standard library only.
"""

from pathlib import Path
import sqlite3
import argparse

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "college.db"


def init_db(db_path: Path | str = DEFAULT_DB_PATH, reset: bool = False) -> Path:
    """
    Initializes and seeds the SQLite college database.
    
    Parameters:
    - db_path: Path or string pointing to the SQLite database file.
    - reset: If True, drops existing tables and re-seeds the database from scratch.
    
    Returns:
    - Path object pointing to the initialized database file.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    db_exists = path.exists()
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    # Check if tables already exist and have data
    tables_exist = False
    if db_exists and not reset:
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('departments', 'students', 'fee_records');"
        )
        existing_tables = cursor.fetchall()
        if len(existing_tables) == 3:
            cursor.execute("SELECT COUNT(*) FROM students;")
            count = cursor.fetchone()[0]
            if count > 0:
                tables_exist = True

    if tables_exist and not reset:
        conn.close()
        return path

    # If resetting or building for the first time, drop and recreate schema & triggers
    cursor.executescript("""
        DROP TRIGGER IF EXISTS trg_validate_student_scores;
        DROP TABLE IF EXISTS fee_records;
        DROP TABLE IF EXISTS students;
        DROP TABLE IF EXISTS departments;

        CREATE TABLE departments (
            dept_id INTEGER PRIMARY KEY AUTOINCREMENT,
            dept_code TEXT UNIQUE NOT NULL,
            dept_name TEXT NOT NULL,
            hod_name TEXT NOT NULL
        );

        CREATE TABLE students (
            student_id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_no TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            branch TEXT NOT NULL,
            dept_id INTEGER NOT NULL,
            semester INTEGER NOT NULL,
            marks REAL NOT NULL,
            attendance_pct REAL NOT NULL,
            FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
        );

        CREATE TABLE fee_records (
            fee_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            semester INTEGER NOT NULL,
            total_fee REAL NOT NULL,
            paid_fee REAL NOT NULL,
            unpaid_fee REAL NOT NULL,
            payment_status TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students(student_id)
        );

        CREATE TRIGGER trg_validate_student_scores
        BEFORE INSERT ON students
        For EACH ROW
        WHEN NEW.marks < 0 OR NEW.marks > 100 OR NEW.attendance_pct < 0 OR NEW.attendance_pct > 100
        BEGIN
            SELECT RAISE(ABORT, 'marks and attendance_pct must be between 0 and 100');
        END;
    """)

    # 1. Seed Departments
    departments_data = [
        ("CSE", "Computer Science and Engineering", "Dr. Rajesh Sharma"),
        ("IT", "Information Technology", "Dr. Sunita Verma"),
        ("ECE", "Electronics and Communication Engineering", "Dr. Anil Kumar Gupta"),
        ("ME", "Mechanical Engineering", "Dr. Rameshwar Prasad"),
        ("CE", "Civil Engineering", "Dr. Priya Deshmukh"),
    ]
    cursor.executemany(
        "INSERT INTO departments (dept_code, dept_name, hod_name) VALUES (?, ?, ?);",
        departments_data
    )

    # Fetch dept_id mapping
    cursor.execute("SELECT dept_code, dept_id FROM departments;")
    dept_map = {code: dept_id for code, dept_id in cursor.fetchall()}

    # 2. Seed Students (30 students, 6 per branch, deterministic marks & attendance)
    # At least 8 students with attendance < 75.0 across branches.
    students_data = [
        # CSE (dept_id = dept_map['CSE'])
        ("CSE001", "Aarav Patel", "CSE", dept_map["CSE"], 4, 92.5, 88.0),
        ("CSE002", "Diya Sharma", "CSE", dept_map["CSE"], 4, 95.0, 91.5),
        ("CSE003", "Kabir Mehta", "CSE", dept_map["CSE"], 4, 68.5, 71.0),  # < 75
        ("CSE004", "Ananya Rao", "CSE", dept_map["CSE"], 4, 89.0, 85.0),
        ("CSE005", "Rohan Verma", "CSE", dept_map["CSE"], 4, 62.0, 69.5),  # < 75
        ("CSE006", "Sneha Iyer", "CSE", dept_map["CSE"], 4, 94.5, 93.0),
        
        # IT (dept_id = dept_map['IT'])
        ("IT001", "Arjun Nair", "IT", dept_map["IT"], 4, 88.0, 84.0),
        ("IT002", "Pooja Reddy", "IT", dept_map["IT"], 4, 91.0, 90.0),
        ("IT003", "Karan Joshi", "IT", dept_map["IT"], 4, 59.5, 68.0),  # < 75
        ("IT004", "Neha Kulkarni", "IT", dept_map["IT"], 4, 86.5, 82.5),
        ("IT005", "Aditya Sen", "IT", dept_map["IT"], 4, 74.0, 73.5),  # < 75
        ("IT006", "Tanvi deshmukh", "IT", dept_map["IT"], 4, 90.0, 89.0),

        # ECE (dept_id = dept_map['ECE'])
        ("ECE001", "Devansh Mishra", "ECE", dept_map["ECE"], 4, 82.0, 80.0),
        ("ECE002", "Ishita Roy", "ECE", dept_map["ECE"], 4, 85.5, 86.0),
        ("ECE003", "Yashwant Singh", "ECE", dept_map["ECE"], 4, 64.0, 72.0),  # < 75
        ("ECE004", "Meera Pillai", "ECE", dept_map["ECE"], 4, 79.0, 78.5),
        ("ECE005", "Varun Das", "ECE", dept_map["ECE"], 4, 55.0, 65.0),  # < 75
        ("ECE006", "Ritu Nambiar", "ECE", dept_map["ECE"], 4, 88.5, 87.5),

        # ME (dept_id = dept_map['ME']) - High unpaid fees department
        ("ME001", "Manish Tiwari", "ME", dept_map["ME"], 4, 75.0, 79.0),
        ("ME002", "Kritika Menon", "ME", dept_map["ME"], 4, 81.0, 83.0),
        ("ME003", "Nikhil Kumar", "ME", dept_map["ME"], 4, 54.0, 66.0),  # < 75
        ("ME004", "Swati Pillai", "ME", dept_map["ME"], 4, 72.5, 76.0),
        ("ME005", "Abhishek Dubey", "ME", dept_map["ME"], 4, 61.5, 70.0),  # < 75
        ("ME006", "Shruti Khandelwal", "ME", dept_map["ME"], 4, 83.0, 85.0),

        # CE (dept_id = dept_map['CE'])
        ("CE001", "Rahul Saxena", "CE", dept_map["CE"], 4, 78.0, 81.0),
        ("CE002", "Preeti Chawla", "CE", dept_map["CE"], 4, 84.0, 86.5),
        ("CE003", "Siddharth Jain", "CE", dept_map["CE"], 4, 67.0, 74.0),  # < 75
        ("CE004", "Divya Pillai", "CE", dept_map["CE"], 4, 80.5, 82.0),
        ("CE005", "Amritanshu Ghosh", "CE", dept_map["CE"], 4, 58.0, 67.5),  # < 75
        ("CE006", "Nisha Agarwal", "CE", dept_map["CE"], 4, 89.5, 90.5),
    ]

    cursor.executemany(
        """
        INSERT INTO students (roll_no, name, branch, dept_id, semester, marks, attendance_pct)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        students_data
    )

    # Fetch student_id mapping
    cursor.execute("SELECT student_id, roll_no, branch FROM students;")
    students_rows = cursor.fetchall()

    # 3. Seed Fee Records (30 fee records, 1 per student)
    # Ensuring ME department has the highest total unpaid fee with no tie.
    # Give at least 4-5 students paid_fee = 0.0 (Unpaid), plus others Partial and Paid, covering all three statuses ('Paid', 'Partial', 'Unpaid').
    fee_records_data = []
    for s_id, roll_no, branch in students_rows:
        total_fee = 75000.0
        if branch == "ME":
            # Mechanical students have heavy unpaid fees: ME001, ME003, ME005 get paid_fee = 0.0 (Unpaid)
            if s_id in (19, 21, 23):
                paid_fee = 0.0
                status = "Unpaid"
            elif s_id == 20:
                paid_fee = 25000.0
                status = "Partial"
            elif s_id == 22:
                paid_fee = 30000.0
                status = "Partial"
            else:
                paid_fee = 40000.0
                status = "Partial"
        elif branch == "CSE":
            if s_id == 1:
                paid_fee = 0.0
                status = "Unpaid"
            elif s_id % 3 == 0:
                paid_fee = 50000.0
                status = "Partial"
            else:
                paid_fee = 75000.0
                status = "Paid"
        elif branch == "IT":
            if s_id % 2 == 0:
                paid_fee = 75000.0
                status = "Paid"
            else:
                paid_fee = 37500.0
                status = "Partial"
        elif branch == "ECE":
            if s_id == 13:  # ECE student with paid_fee = 0.0 (Unpaid)
                paid_fee = 0.0
                status = "Unpaid"
            elif s_id % 3 == 0:
                paid_fee = 75000.0
                status = "Paid"
            else:
                paid_fee = 45000.0
                status = "Partial"
        else:  # CE
            if s_id == 25:  # CE student with paid_fee = 0.0 (Unpaid)
                paid_fee = 0.0
                status = "Unpaid"
            elif s_id % 4 != 0:
                paid_fee = 75000.0
                status = "Paid"
            else:
                paid_fee = 20000.0
                status = "Partial"

        unpaid_fee = total_fee - paid_fee
        if unpaid_fee == 0.0:
            status = "Paid"
        elif paid_fee == 0.0:
            status = "Unpaid"
        else:
            status = "Partial"

        fee_records_data.append((s_id, 4, total_fee, paid_fee, unpaid_fee, status))

    cursor.executemany(
        """
        INSERT INTO fee_records (student_id, semester, total_fee, paid_fee, unpaid_fee, payment_status)
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        fee_records_data
    )

    conn.commit()
    conn.close()
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize and seed the Chat-with-SQL college database.")
    parser.add_argument("--reset", action="store_true", help="Reset and re-seed the database from scratch.")
    args = parser.parse_args()

    db_path = init_db(reset=args.reset)
    
    # Print summary row counts
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM departments;")
    dept_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM students;")
    student_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM fee_records;")
    fee_count = cursor.fetchone()[0]
    conn.close()

    print("=" * 60)
    print(f"Database successfully initialized at: {db_path}")
    print(f" - Departments: {dept_count}")
    print(f" - Students:    {student_count}")
    print(f" - Fee Records: {fee_count}")
    print("=" * 60)

