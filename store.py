"""Persistent state for agent actions, in SQLite, so a restart loses nothing.

Every status change is one conditional UPDATE ("only if the status is still X").
That makes approval idempotent: two clicks on "approve" cannot run an action twice.

Statuses:
  pending  -> waiting for a human (high risk)
  rejected -> refused by validation or by a human
  running  -> claimed for execution
  done     -> executed successfully
  failed   -> execution raised an error
An action stuck in "running" after a crash is never retried automatically,
because sending an email twice is worse than not sending it.
"""

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

from config import AGENT_DB

SCHEMA = """
CREATE TABLE IF NOT EXISTS actions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    tool       TEXT NOT NULL,
    args       TEXT NOT NULL,   -- JSON as proposed by the model
    risk       TEXT,            -- NULL when the tool does not exist
    status     TEXT NOT NULL CHECK (status IN ('pending', 'rejected', 'running', 'done', 'failed')),
    detail     TEXT,            -- rejection reason, result or error
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    at        TEXT NOT NULL,
    action_id INTEGER,
    event     TEXT NOT NULL,
    detail    TEXT
);
CREATE TABLE IF NOT EXISTS notes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT NOT NULL,
    text       TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outbox (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    recipient  TEXT NOT NULL,
    subject    TEXT NOT NULL,
    body       TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect():
    db = sqlite3.connect(AGENT_DB, timeout=10)
    db.row_factory = sqlite3.Row
    return db


def init():
    with closing(_connect()) as db:
        db.executescript(SCHEMA)


def _log(db, event, action_id=None, detail=None):
    db.execute(
        "INSERT INTO audit_log (at, action_id, event, detail) VALUES (?, ?, ?, ?)",
        (_now(), action_id, event, detail),
    )


def create_action(tool, args, risk, status, detail=None):
    """Record a proposal and return its id."""
    now = _now()
    with closing(_connect()) as db, db:
        cursor = db.execute(
            "INSERT INTO actions (tool, args, risk, status, detail, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (tool, json.dumps(args, ensure_ascii=False), risk, status, detail, now, now),
        )
        _log(db, f"proposed -> {status}", cursor.lastrowid, detail)
        return cursor.lastrowid


def transition(action_id, from_status, to_status, detail=None):
    """Change the status only if it is still from_status.

    Returns True if this call made the change, False if it was already changed.
    """
    with closing(_connect()) as db, db:
        cursor = db.execute(
            "UPDATE actions SET status = ?, detail = COALESCE(?, detail), updated_at = ?"
            " WHERE id = ? AND status = ?",
            (to_status, detail, _now(), action_id, from_status),
        )
        if cursor.rowcount != 1:
            return False
        _log(db, f"{from_status} -> {to_status}", action_id, detail)
        return True


def get_action(action_id):
    with closing(_connect()) as db:
        row = db.execute("SELECT * FROM actions WHERE id = ?", (action_id,)).fetchone()
    return dict(row) if row else None


def list_actions(status=None):
    query, params = "SELECT * FROM actions", ()
    if status:
        query, params = query + " WHERE status = ?", (status,)
    with closing(_connect()) as db:
        return [dict(row) for row in db.execute(query + " ORDER BY id DESC", params)]


def audit_log(action_id=None):
    query, params = "SELECT * FROM audit_log", ()
    if action_id is not None:
        query, params = query + " WHERE action_id = ?", (action_id,)
    with closing(_connect()) as db:
        return [dict(row) for row in db.execute(query + " ORDER BY id", params)]


def add_note(title, text):
    with closing(_connect()) as db, db:
        return db.execute(
            "INSERT INTO notes (title, text, created_at) VALUES (?, ?, ?)", (title, text, _now())
        ).lastrowid


def add_to_outbox(recipient, subject, body):
    """Demo mode: an approved email is stored here instead of being sent."""
    with closing(_connect()) as db, db:
        return db.execute(
            "INSERT INTO outbox (recipient, subject, body, created_at) VALUES (?, ?, ?, ?)",
            (recipient, subject, body, _now()),
        ).lastrowid