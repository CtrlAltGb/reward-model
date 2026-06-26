"""VAE Registry — publish and load DemInf VAE checkpoints + reference latents.

Layout (local or S3):
  registry/task=<task>/vae_version=<v>/
      obs_vae/                    # orbax checkpoint dir
      action_vae/                 # orbax checkpoint dir (optional)
      reference_latents.npz       # {obs_latents, action_latents}
      meta.json                   # {task, vae_version, obs_ckpt, ...}
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rdf.harness.storage import ObjectStore, get_object_store


@dataclass
class VaeArtifact:
    task: str
    vae_version: str
    obs_ckpt: str
    action_ckpt: str | None
    reference_latents_obs: np.ndarray
    reference_latents_action: np.ndarray


class VaeRegistry:
    def __init__(self, store: ObjectStore | None = None, prefix: str = "registry"):
        self.store = store or get_object_store()
        self.prefix = prefix

    def _meta_key(self, task: str, version: str) -> str:
        return f"{self.prefix}/task={task}/vae_version={version}/meta.json"

    def _ref_key(self, task: str, version: str) -> str:
        return f"{self.prefix}/task={task}/vae_version={version}/reference_latents.npz"

    def _ckpt_prefix(self, task: str, version: str, name: str) -> str:
        return f"{self.prefix}/task={task}/vae_version={version}/{name}/"

    def publish(self, artifact: VaeArtifact) -> None:
        """Save reference latents and metadata to the registry."""
        import io

        buf = io.BytesIO()
        np.savez(
            buf,
            obs_latents=artifact.reference_latents_obs,
            action_latents=artifact.reference_latents_action,
        )
        self.store.put_bytes(self._ref_key(artifact.task, artifact.vae_version), buf.getvalue())

        meta = {
            "task": artifact.task,
            "vae_version": artifact.vae_version,
            "obs_ckpt": artifact.obs_ckpt,
            "action_ckpt": artifact.action_ckpt,
        }
        self.store.put_bytes(
            self._meta_key(artifact.task, artifact.vae_version),
            json.dumps(meta).encode(),
        )

    def load(self, task: str, version: str = "current") -> VaeArtifact:
        if version == "current":
            version = self.current_version(task)

        meta_bytes = self.store.get_bytes(self._meta_key(task, version))
        meta = json.loads(meta_bytes)

        ref_bytes = self.store.get_bytes(self._ref_key(task, version))
        import io
        data = np.load(io.BytesIO(ref_bytes))

        return VaeArtifact(
            task=task,
            vae_version=version,
            obs_ckpt=meta["obs_ckpt"],
            action_ckpt=meta.get("action_ckpt"),
            reference_latents_obs=data["obs_latents"],
            reference_latents_action=data["action_latents"],
        )

    def current_version(self, task: str) -> str:
        prefix = f"{self.prefix}/task={task}/"
        keys = self.store.list_keys(prefix)
        versions = []
        for k in keys:
            parts = k.replace(prefix, "").split("/")
            if parts and parts[0].startswith("vae_version="):
                versions.append(parts[0].replace("vae_version=", ""))
        if not versions:
            raise KeyError(f"No VAE version found for task={task!r}")

        def _version_key(v: str):
            # Natural sort: split into (prefix, numeric_suffix) so v10 > v2 > v1
            import re
            m = re.match(r"^(.*?)(\d+)$", v)
            return (m.group(1), int(m.group(2))) if m else (v, 0)

        return sorted(versions, key=_version_key)[-1]
