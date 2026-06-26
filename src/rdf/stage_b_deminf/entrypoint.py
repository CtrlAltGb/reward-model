"""Entrypoint for Stage B DemInf inference worker."""

from __future__ import annotations

import os

from rdf.stage_b_deminf.infer_worker import run_infer_worker


def main():
    run_infer_worker(
        deminf_threshold=float(os.environ.get("RDF_DEMINF_THRESHOLD", "0.0")),
        poll_wait=int(os.environ.get("RDF_POLL_WAIT_SECONDS", "5")),
        embodiment=os.environ.get("RDF_EMBODIMENT", "franka"),
    )


if __name__ == "__main__":
    main()
