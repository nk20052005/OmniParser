"""
Context Memory Store using SQLite.

Stores conversation history, user preferences, action history, and
frequently used resources to enable context-aware conversations.
"""

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MemoryStore:
    """
    Persistent memory for the conversation engine.

    Stores:
    - Conversation history per user/channel
    - User preferences and frequently used resources
    - Action history for learning patterns
    - Shift summaries
    """

    def __init__(self, db_path: str = "omnitool_memory.db"):
        self._db_path = db_path
        self._local = threading.local()
        self._init_db()

    @property
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        conn = sqlite3.connect(self._db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                channel_id TEXT DEFAULT 'default',
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                intent TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS action_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                intent TEXT NOT NULL,
                tool TEXT NOT NULL,
                parameters TEXT NOT NULL,
                result_success INTEGER NOT NULL,
                result_data TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, key)
            );

            CREATE TABLE IF NOT EXISTS resource_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                resource_name TEXT NOT NULL,
                usage_count INTEGER DEFAULT 1,
                last_used DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS shift_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                shift_start DATETIME NOT NULL,
                shift_end DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, channel_id);
            CREATE INDEX IF NOT EXISTS idx_actions_user ON action_history(user_id);
            CREATE INDEX IF NOT EXISTS idx_resource_user ON resource_usage(user_id, resource_type);
        """)
        conn.commit()
        conn.close()

    def save_conversation_turn(
        self,
        conversation_id: str,
        user_id: str,
        channel_id: str,
        role: str,
        content: str,
        intent: Optional[str] = None,
    ):
        """Save a single conversation turn."""
        self._conn.execute(
            """INSERT INTO conversations (conversation_id, user_id, channel_id, role, content, intent)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (conversation_id, user_id, channel_id, role, content, intent),
        )
        self._conn.commit()

    def get_conversation_history(
        self,
        user_id: str,
        channel_id: str = "default",
        limit: int = 20,
    ) -> list[dict[str, str]]:
        """Get recent conversation history for a user/channel."""
        rows = self._conn.execute(
            """SELECT role, content FROM conversations
               WHERE user_id = ? AND channel_id = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (user_id, channel_id, limit),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def record_action(
        self,
        user_id: str,
        intent: str,
        tool: str,
        parameters: dict[str, Any],
        result: dict[str, Any],
    ):
        """Record a tool execution for pattern learning."""
        self._conn.execute(
            """INSERT INTO action_history (user_id, intent, tool, parameters, result_success, result_data)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                intent,
                tool,
                json.dumps(parameters),
                1 if result.get("success") else 0,
                json.dumps(result.get("data", {})),
            ),
        )
        self._conn.commit()

        # Track resource usage for frequently used resources
        self._update_resource_usage(user_id, intent, parameters)

    def _update_resource_usage(self, user_id: str, intent: str, params: dict):
        """Track which resources a user interacts with most."""
        resource_map = {
            "vm_": ("vm", "vm_name"),
            "incident_": ("incident", "incident_id"),
        }
        for prefix, (res_type, param_key) in resource_map.items():
            if intent.startswith(prefix) and param_key in params:
                res_name = params[param_key]
                existing = self._conn.execute(
                    """SELECT id, usage_count FROM resource_usage
                       WHERE user_id = ? AND resource_type = ? AND resource_name = ?""",
                    (user_id, res_type, res_name),
                ).fetchone()

                if existing:
                    self._conn.execute(
                        """UPDATE resource_usage SET usage_count = usage_count + 1,
                           last_used = CURRENT_TIMESTAMP WHERE id = ?""",
                        (existing["id"],),
                    )
                else:
                    self._conn.execute(
                        """INSERT INTO resource_usage (user_id, resource_type, resource_name)
                           VALUES (?, ?, ?)""",
                        (user_id, res_type, res_name),
                    )
                self._conn.commit()

    def get_frequent_resources(
        self,
        user_id: str,
        resource_type: str,
        limit: int = 5,
    ) -> list[str]:
        """Get the most frequently used resources of a type for a user."""
        rows = self._conn.execute(
            """SELECT resource_name FROM resource_usage
               WHERE user_id = ? AND resource_type = ?
               ORDER BY usage_count DESC LIMIT ?""",
            (user_id, resource_type, limit),
        ).fetchall()
        return [r["resource_name"] for r in rows]

    def set_user_preference(self, user_id: str, key: str, value: str):
        """Set a user preference."""
        self._conn.execute(
            """INSERT INTO user_preferences (user_id, key, value, updated_at)
               VALUES (?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(user_id, key) DO UPDATE SET value = ?, updated_at = CURRENT_TIMESTAMP""",
            (user_id, key, value, value),
        )
        self._conn.commit()

    def get_user_preferences(self, user_id: str) -> dict[str, str]:
        """Get all preferences for a user."""
        rows = self._conn.execute(
            """SELECT key, value FROM user_preferences WHERE user_id = ?""",
            (user_id,),
        ).fetchall()
        prefs = {r["key"]: r["value"] for r in rows}

        # Add frequently used resources
        freq_vms = self.get_frequent_resources(user_id, "vm", limit=3)
        if freq_vms:
            prefs["frequent_vm"] = freq_vms[0]
            prefs["frequent_vms"] = json.dumps(freq_vms)

        return prefs

    def get_recent_actions(
        self,
        user_id: str,
        hours: int = 8,
        limit: int = 50,
    ) -> list[dict]:
        """Get recent actions for shift summary generation."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        rows = self._conn.execute(
            """SELECT intent, tool, parameters, result_success, timestamp
               FROM action_history
               WHERE user_id = ? AND timestamp >= ?
               ORDER BY timestamp DESC LIMIT ?""",
            (user_id, cutoff.isoformat(), limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def save_shift_summary(
        self,
        user_id: str,
        summary: str,
        shift_start: datetime,
        shift_end: datetime,
    ):
        """Save a shift summary."""
        self._conn.execute(
            """INSERT INTO shift_summaries (user_id, summary, shift_start, shift_end)
               VALUES (?, ?, ?, ?)""",
            (user_id, summary, shift_start.isoformat(), shift_end.isoformat()),
        )
        self._conn.commit()
