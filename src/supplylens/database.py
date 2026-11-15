import sqlite3
from datetime import datetime, timezone
from .config import DB_PATH


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS suppliers ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "name TEXT NOT NULL, country TEXT NOT NULL, currency TEXT NOT NULL, "
            "created_at TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS purchase_exposures ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "supplier_name TEXT NOT NULL, currency TEXT NOT NULL, "
            "invoice_amount REAL NOT NULL, due_date TEXT, category TEXT, "
            "created_at TEXT NOT NULL)"
        )
        conn.commit()


def save_supplier(name, country, currency):
    init_db()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO suppliers(name,country,currency,created_at) VALUES(?,?,?,?)",
            (name, country, currency.upper(), datetime.now(timezone.utc).isoformat())
        )
        conn.commit()


def list_suppliers():
    init_db()
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM suppliers ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def save_exposure(supplier_name, currency, invoice_amount, due_date="", category=""):
    init_db()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO purchase_exposures "
            "(supplier_name,currency,invoice_amount,due_date,category,created_at) "
            "VALUES(?,?,?,?,?,?)",
            (supplier_name, currency.upper(), float(invoice_amount), str(due_date),
             category, datetime.now(timezone.utc).isoformat())
        )
        conn.commit()


def list_exposures():
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM purchase_exposures ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_supplier(supplier_id):
    init_db()
    with get_connection() as conn:
        conn.execute("DELETE FROM suppliers WHERE id = ?", (int(supplier_id),))
        conn.commit()


def delete_exposure(exposure_id):
    init_db()
    with get_connection() as conn:
        conn.execute("DELETE FROM purchase_exposures WHERE id = ?", (int(exposure_id),))
        conn.commit()


def dashboard_counts():
    init_db()
    with get_connection() as conn:
        suppliers = conn.execute("SELECT COUNT(*) FROM suppliers").fetchone()[0]
        exposures = conn.execute("SELECT COUNT(*) FROM purchase_exposures").fetchone()[0]
    return {"suppliers": suppliers, "exposures": exposures}
