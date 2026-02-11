"""SQLite store for pre-extracted structured meeting data."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

import structlog

from src.models.schemas import (
    ActionItem,
    Decision,
    MeetingExtractions,
    MeetingInfo,
)

logger = structlog.get_logger()


class StructuredStore:
    """Manages SQLite storage for action items, decisions, topics, and summaries.

    Provides pre-extracted structured data that doesn't require vector search,
    enabling fast retrieval for queries like 'list action items' or 'what decisions were made'.
    """

    def __init__(self, db_path: str = "data/sqlite/meetings.db") -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def init_db(self) -> None:
        """Create tables if they don't exist."""
        cursor = self._conn.cursor()
        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS meetings (
                meeting_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                summary TEXT DEFAULT '',
                speakers TEXT DEFAULT '[]',
                topics TEXT DEFAULT '[]',
                num_chunks INTEGER DEFAULT 0,
                ingested_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS action_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id TEXT NOT NULL,
                assignee TEXT NOT NULL,
                task TEXT NOT NULL,
                deadline TEXT,
                FOREIGN KEY (meeting_id) REFERENCES meetings(meeting_id)
            );

            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id TEXT NOT NULL,
                decision TEXT NOT NULL,
                context TEXT DEFAULT '',
                decided_by TEXT DEFAULT '[]',
                FOREIGN KEY (meeting_id) REFERENCES meetings(meeting_id)
            );
            """
        )
        self._conn.commit()
        logger.info("structured_store_initialized", db_path=self.db_path)

    def store_extractions(
        self, extractions: MeetingExtractions, filename: str = "", num_chunks: int = 0
    ) -> None:
        """Save structured extractions for a meeting.

        Replaces existing data for the meeting if re-ingested.

        Args:
            extractions: Extracted structured data from the meeting.
            filename: Original transcript filename.
            num_chunks: Number of vector chunks created.
        """
        cursor = self._conn.cursor()

        # Delete existing data for this meeting (idempotent re-ingestion)
        cursor.execute(
            "DELETE FROM action_items WHERE meeting_id = ?",
            (extractions.meeting_id,),
        )
        cursor.execute(
            "DELETE FROM decisions WHERE meeting_id = ?",
            (extractions.meeting_id,),
        )
        cursor.execute(
            "DELETE FROM meetings WHERE meeting_id = ?",
            (extractions.meeting_id,),
        )

        # Insert meeting record
        cursor.execute(
            """INSERT INTO meetings (meeting_id, filename, summary, speakers, topics, num_chunks, ingested_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                extractions.meeting_id,
                filename,
                extractions.summary,
                json.dumps(extractions.speakers),
                json.dumps(extractions.topics),
                num_chunks,
                datetime.now(timezone.utc).isoformat(),
            ),
        )

        # Insert action items
        for item in extractions.action_items:
            cursor.execute(
                "INSERT INTO action_items (meeting_id, assignee, task, deadline) VALUES (?, ?, ?, ?)",
                (extractions.meeting_id, item.assignee, item.task, item.deadline),
            )

        # Insert decisions
        for dec in extractions.decisions:
            cursor.execute(
                "INSERT INTO decisions (meeting_id, decision, context, decided_by) VALUES (?, ?, ?, ?)",
                (
                    extractions.meeting_id,
                    dec.decision,
                    dec.context,
                    json.dumps(dec.decided_by),
                ),
            )

        self._conn.commit()
        logger.info(
            "extractions_stored",
            meeting_id=extractions.meeting_id,
            action_items=len(extractions.action_items),
            decisions=len(extractions.decisions),
        )

    def get_action_items(self, meeting_id: str | None = None) -> list[ActionItem]:
        """Retrieve action items, optionally filtered by meeting.

        Args:
            meeting_id: If provided, filter to this meeting only.

        Returns:
            List of ActionItem objects.
        """
        cursor = self._conn.cursor()
        if meeting_id:
            cursor.execute(
                "SELECT assignee, task, deadline FROM action_items WHERE meeting_id = ?",
                (meeting_id,),
            )
        else:
            cursor.execute("SELECT assignee, task, deadline FROM action_items")

        return [
            ActionItem(assignee=row["assignee"], task=row["task"], deadline=row["deadline"])
            for row in cursor.fetchall()
        ]

    def get_decisions(self, meeting_id: str | None = None) -> list[Decision]:
        """Retrieve decisions, optionally filtered by meeting.

        Args:
            meeting_id: If provided, filter to this meeting only.

        Returns:
            List of Decision objects.
        """
        cursor = self._conn.cursor()
        if meeting_id:
            cursor.execute(
                "SELECT decision, context, decided_by FROM decisions WHERE meeting_id = ?",
                (meeting_id,),
            )
        else:
            cursor.execute("SELECT decision, context, decided_by FROM decisions")

        return [
            Decision(
                decision=row["decision"],
                context=row["context"],
                decided_by=json.loads(row["decided_by"]),
            )
            for row in cursor.fetchall()
        ]

    def get_speakers(self, meeting_id: str | None = None) -> list[str]:
        """Retrieve speaker names, optionally filtered by meeting.

        Args:
            meeting_id: If provided, speakers from this meeting only.

        Returns:
            List of speaker names.
        """
        cursor = self._conn.cursor()
        if meeting_id:
            cursor.execute(
                "SELECT speakers FROM meetings WHERE meeting_id = ?",
                (meeting_id,),
            )
        else:
            cursor.execute("SELECT speakers FROM meetings")

        speakers: list[str] = []
        for row in cursor.fetchall():
            speakers.extend(json.loads(row["speakers"]))
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for s in speakers:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return unique

    def get_meeting_summary(self, meeting_id: str) -> str:
        """Get the stored summary for a meeting.

        Args:
            meeting_id: The meeting to get the summary for.

        Returns:
            Summary text, or empty string if not found.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT summary FROM meetings WHERE meeting_id = ?", (meeting_id,)
        )
        row = cursor.fetchone()
        return row["summary"] if row else ""

    def list_meetings(self) -> list[MeetingInfo]:
        """List all ingested meetings.

        Returns:
            List of MeetingInfo objects.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT meeting_id, filename, speakers, topics, num_chunks, ingested_at FROM meetings ORDER BY ingested_at DESC"
        )
        return [
            MeetingInfo(
                meeting_id=row["meeting_id"],
                filename=row["filename"],
                speakers=json.loads(row["speakers"]),
                topics=json.loads(row["topics"]),
                num_chunks=row["num_chunks"],
                ingested_at=datetime.fromisoformat(row["ingested_at"]) if row["ingested_at"] else None,
            )
            for row in cursor.fetchall()
        ]

    def get_meeting_details(self, meeting_id: str) -> MeetingInfo | None:
        """Get details for a single meeting.

        Args:
            meeting_id: The meeting to look up.

        Returns:
            MeetingInfo or None if not found.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT meeting_id, filename, speakers, topics, num_chunks, ingested_at FROM meetings WHERE meeting_id = ?",
            (meeting_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return MeetingInfo(
            meeting_id=row["meeting_id"],
            filename=row["filename"],
            speakers=json.loads(row["speakers"]),
            topics=json.loads(row["topics"]),
            num_chunks=row["num_chunks"],
            ingested_at=datetime.fromisoformat(row["ingested_at"]) if row["ingested_at"] else None,
        )
