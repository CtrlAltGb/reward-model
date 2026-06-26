"""Abstract Catalog with Local (parquet + sqlite) and AWS implementations.

Backend selected by RDF_CATALOG env var: local | aws.
"""

from __future__ import annotations

import json
import os
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdf.schemas.models import CatalogRow, DeminfResult, RobometerResult


class Catalog(ABC):
    @abstractmethod
    def upsert_row(self, row: CatalogRow) -> None: ...

    @abstractmethod
    def get_row(self, episode_id: str) -> CatalogRow | None: ...

    @abstractmethod
    def update_robometer(self, result: RobometerResult, pass_: bool, threshold: float) -> None: ...

    @abstractmethod
    def update_deminf(self, result: DeminfResult, pass_: bool, threshold: float) -> None: ...

    @abstractmethod
    def finalize(self, episode_id: str, decision: str, reasons: list[str]) -> None: ...

    @abstractmethod
    def rows_for_task(self, task: str) -> list[CatalogRow]: ...


class LocalCatalog(Catalog):
    """SQLite-backed catalog for local/test use."""

    def __init__(self, root: str | None = None):
        root_path = Path(root or os.environ.get("RDF_LOCAL_CATALOG_PATH", "/tmp/rdf/catalog"))
        root_path.mkdir(parents=True, exist_ok=True)
        self.db_path = root_path / "catalog.db"
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self.db_path)) as con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS catalog (
                    episode_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), timeout=10)

    def upsert_row(self, row: CatalogRow) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO catalog (episode_id, data, updated_at) VALUES (?,?,?)",
                (row.episode_id, row.model_dump_json(), row.updated_at.timestamp()),
            )

    def get_row(self, episode_id: str) -> CatalogRow | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT data FROM catalog WHERE episode_id=?", (episode_id,)
            ).fetchone()
        if row:
            return CatalogRow.model_validate_json(row[0])
        return None

    def _update_row(self, episode_id: str, updates: dict[str, Any]) -> None:
        row = self.get_row(episode_id)
        if row is None:
            raise KeyError(f"Episode {episode_id} not in catalog")
        data = row.model_dump()
        data.update(updates)
        data["updated_at"] = datetime.now(timezone.utc)
        new_row = CatalogRow.model_validate(data)
        self.upsert_row(new_row)

    def update_robometer(self, result: RobometerResult, pass_: bool, threshold: float) -> None:
        self._update_row(
            result.episode_id,
            {
                "robometer_reward": result.robometer_reward,
                "robometer_success_pred": result.robometer_success_pred,
                "robometer_pass": pass_,
                "robometer_model_version": result.model_version,
            },
        )

    def update_deminf(self, result: DeminfResult, pass_: bool, threshold: float) -> None:
        self._update_row(
            result.episode_id,
            {
                "deminf_score": result.deminf_score,
                "deminf_pass": pass_,
                "vae_version": result.vae_version,
            },
        )

    def finalize(self, episode_id: str, decision: str, reasons: list[str]) -> None:
        self._update_row(episode_id, {"final_decision": decision, "reasons": reasons})

    def rows_for_task(self, task: str) -> list[CatalogRow]:
        with self._connect() as con:
            rows = con.execute("SELECT data FROM catalog").fetchall()
        result = []
        for (data,) in rows:
            row = CatalogRow.model_validate_json(data)
            if row.task == task:
                result.append(row)
        return result


class AwsCatalog(Catalog):
    """DynamoDB-backed catalog for AWS deployments."""

    def __init__(self, table_name: str | None = None):
        import boto3
        self.table_name = table_name or os.environ.get("RDF_DYNAMO_TABLE", "rdf-catalog")
        self.table = boto3.resource("dynamodb").Table(self.table_name)

    def upsert_row(self, row: CatalogRow) -> None:
        self.table.put_item(Item=json.loads(row.model_dump_json()))

    def get_row(self, episode_id: str) -> CatalogRow | None:
        resp = self.table.get_item(Key={"episode_id": episode_id})
        item = resp.get("Item")
        return CatalogRow.model_validate(item) if item else None

    def update_robometer(self, result: RobometerResult, pass_: bool, threshold: float) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.table.update_item(
            Key={"episode_id": result.episode_id},
            UpdateExpression=(
                "SET robometer_reward=:rr, robometer_success_pred=:rs, "
                "robometer_pass=:rp, robometer_model_version=:mv, updated_at=:ua"
            ),
            ExpressionAttributeValues={
                ":rr": str(result.robometer_reward),
                ":rs": str(result.robometer_success_pred),
                ":rp": pass_,
                ":mv": result.model_version,
                ":ua": now,
            },
        )

    def update_deminf(self, result: DeminfResult, pass_: bool, threshold: float) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.table.update_item(
            Key={"episode_id": result.episode_id},
            UpdateExpression="SET deminf_score=:ds, deminf_pass=:dp, vae_version=:vv, updated_at=:ua",
            ExpressionAttributeValues={
                ":ds": str(result.deminf_score),
                ":dp": pass_,
                ":vv": result.vae_version,
                ":ua": now,
            },
        )

    def finalize(self, episode_id: str, decision: str, reasons: list[str]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.table.update_item(
            Key={"episode_id": episode_id},
            UpdateExpression="SET final_decision=:fd, reasons=:r, updated_at=:ua",
            ExpressionAttributeValues={":fd": decision, ":r": reasons, ":ua": now},
        )

    def rows_for_task(self, task: str) -> list[CatalogRow]:
        resp = self.table.scan(
            FilterExpression="task = :t",
            ExpressionAttributeValues={":t": task},
        )
        return [CatalogRow.model_validate(item) for item in resp.get("Items", [])]


def get_catalog(backend: str | None = None, **kwargs) -> Catalog:
    backend = backend or os.environ.get("RDF_CATALOG", "local")
    if backend == "local":
        return LocalCatalog(**kwargs)
    if backend == "aws":
        return AwsCatalog(**kwargs)
    raise ValueError(f"Unknown RDF_CATALOG backend: {backend!r}")
