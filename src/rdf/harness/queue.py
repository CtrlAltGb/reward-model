"""Abstract WorkQueue with Local (sqlite) and SQS implementations.

Backend selected by RDF_QUEUE env var: local | sqs.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class QueueMessage:
    def __init__(self, receipt: str, body: dict[str, Any]):
        self.receipt = receipt
        self.body = body


class WorkQueue(ABC):
    @abstractmethod
    def enqueue(self, body: dict[str, Any], dedup_id: str | None = None) -> str: ...

    @abstractmethod
    def receive(self, max_messages: int = 1, wait_seconds: int = 0) -> list[QueueMessage]: ...

    @abstractmethod
    def delete(self, receipt: str) -> None: ...

    @abstractmethod
    def send_to_dlq(self, body: dict[str, Any]) -> None: ...

    @abstractmethod
    def depth(self) -> int: ...


class LocalQueue(WorkQueue):
    """SQLite-backed queue for local/test use."""

    def __init__(self, name: str, root: str | None = None):
        if root is None:
            try:
                from rdf.harness.config import get_paths_config
                root = get_paths_config().local_queue_dir
            except Exception:
                root = os.environ.get("RDF_LOCAL_QUEUE_PATH", "/tmp/rdf/queues")
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        self.db_path = root_path / f"{name}.db"
        self.dlq_db_path = root_path / f"{name}-dlq.db"
        self._init_db(self.db_path)
        self._init_db(self.dlq_db_path)

    def _init_db(self, path: Path) -> None:
        with sqlite3.connect(str(path)) as con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    body TEXT NOT NULL,
                    dedup_id TEXT,
                    visible_at REAL NOT NULL,
                    created_at REAL NOT NULL
                )"""
            )

    def _connect(self, dlq: bool = False) -> sqlite3.Connection:
        path = self.dlq_db_path if dlq else self.db_path
        return sqlite3.connect(str(path), timeout=10)

    def enqueue(self, body: dict[str, Any], dedup_id: str | None = None) -> str:
        msg_id = str(uuid.uuid4())
        now = time.time()
        with self._connect() as con:
            if dedup_id:
                row = con.execute(
                    "SELECT id FROM messages WHERE dedup_id=?", (dedup_id,)
                ).fetchone()
                if row:
                    return row[0]
            con.execute(
                "INSERT INTO messages (id, body, dedup_id, visible_at, created_at) VALUES (?,?,?,?,?)",
                (msg_id, json.dumps(body), dedup_id, now, now),
            )
        return msg_id

    def receive(self, max_messages: int = 1, wait_seconds: int = 0) -> list[QueueMessage]:
        deadline = time.time() + wait_seconds
        while True:
            now = time.time()
            with self._connect() as con:
                rows = con.execute(
                    "SELECT id, body FROM messages WHERE visible_at <= ? ORDER BY created_at LIMIT ?",
                    (now, max_messages),
                ).fetchall()
                if rows:
                    msgs = []
                    for row in rows:
                        # extend visibility by 30s
                        con.execute(
                            "UPDATE messages SET visible_at=? WHERE id=?",
                            (now + 30, row[0]),
                        )
                        msgs.append(QueueMessage(receipt=row[0], body=json.loads(row[1])))
                    return msgs
            if time.time() >= deadline:
                return []
            time.sleep(0.1)

    def delete(self, receipt: str) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM messages WHERE id=?", (receipt,))

    def send_to_dlq(self, body: dict[str, Any]) -> None:
        now = time.time()
        with self._connect(dlq=True) as con:
            con.execute(
                "INSERT INTO messages (id, body, dedup_id, visible_at, created_at) VALUES (?,?,?,?,?)",
                (str(uuid.uuid4()), json.dumps(body), None, now, now),
            )

    def depth(self) -> int:
        with self._connect() as con:
            return con.execute(
                "SELECT COUNT(*) FROM messages WHERE visible_at <= ?", (time.time(),)
            ).fetchone()[0]


class SqsQueue(WorkQueue):
    def __init__(self, queue_url: str | None = None, dlq_url: str | None = None):
        import boto3
        self.queue_url = queue_url or os.environ["RDF_SQS_EPISODE_QUEUE_URL"]
        self.dlq_url = dlq_url
        self.client = boto3.client("sqs")

    def enqueue(self, body: dict[str, Any], dedup_id: str | None = None) -> str:
        kw: dict[str, Any] = {
            "QueueUrl": self.queue_url,
            "MessageBody": json.dumps(body),
        }
        if dedup_id:
            kw["MessageDeduplicationId"] = dedup_id
            kw["MessageGroupId"] = "default"
        resp = self.client.send_message(**kw)
        return resp["MessageId"]

    def receive(self, max_messages: int = 1, wait_seconds: int = 20) -> list[QueueMessage]:
        resp = self.client.receive_message(
            QueueUrl=self.queue_url,
            MaxNumberOfMessages=min(max_messages, 10),
            WaitTimeSeconds=wait_seconds,
        )
        return [
            QueueMessage(receipt=m["ReceiptHandle"], body=json.loads(m["Body"]))
            for m in resp.get("Messages", [])
        ]

    def delete(self, receipt: str) -> None:
        self.client.delete_message(QueueUrl=self.queue_url, ReceiptHandle=receipt)

    def send_to_dlq(self, body: dict[str, Any]) -> None:
        if self.dlq_url:
            self.client.send_message(QueueUrl=self.dlq_url, MessageBody=json.dumps(body))

    def depth(self) -> int:
        resp = self.client.get_queue_attributes(
            QueueUrl=self.queue_url,
            AttributeNames=["ApproximateNumberOfMessages"],
        )
        return int(resp["Attributes"].get("ApproximateNumberOfMessages", 0))


def get_queue(name: str, backend: str | None = None, **kwargs) -> WorkQueue:
    backend = backend or os.environ.get("RDF_QUEUE", "local")
    if backend == "local":
        return LocalQueue(name=name, **kwargs)
    if backend == "sqs":
        url = os.environ.get(f"RDF_SQS_{name.upper().replace('-','_')}_QUEUE_URL")
        return SqsQueue(queue_url=url, **kwargs)
    raise ValueError(f"Unknown RDF_QUEUE backend: {backend!r}")
