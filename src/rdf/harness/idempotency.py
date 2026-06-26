"""Idempotency helpers — skip if already scored at this model_version."""

from __future__ import annotations

from rdf.harness.catalog import Catalog


def already_scored_robometer(catalog: Catalog, episode_id: str, model_version: str) -> bool:
    row = catalog.get_row(episode_id)
    if row is None:
        return False
    return (
        row.robometer_model_version == model_version
        and row.robometer_reward is not None
    )


def already_scored_deminf(catalog: Catalog, episode_id: str, vae_version: str) -> bool:
    row = catalog.get_row(episode_id)
    if row is None:
        return False
    return row.vae_version == vae_version and row.deminf_score is not None
