"""Entrypoint for Stage A robometer worker (used by Docker CMD)."""

from __future__ import annotations

import os

from rdf.stage_a_robometer.worker import run_worker


def main():
    run_worker(
        robometer_threshold=float(os.environ.get("RDF_ROBOMETER_THRESHOLD", "0.5")),
        poll_wait=int(os.environ.get("RDF_POLL_WAIT_SECONDS", "5")),
    )


if __name__ == "__main__":
    main()
