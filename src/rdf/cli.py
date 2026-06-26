"""CLI entry points."""

from __future__ import annotations

import click


@click.group()
def cli():
    """Robot Data-Filtering Pipeline CLI."""


@cli.group()
def robometer_worker():
    """Stage A Robometer worker commands."""


@robometer_worker.command("run")
@click.option("--threshold", default=0.5, type=float, help="Robometer success threshold")
@click.option("--poll-wait", default=5, type=int, help="SQS long-poll wait seconds")
@click.option("--max-episodes", default=None, type=int, help="Stop after N episodes (testing)")
def robometer_run(threshold, poll_wait, max_episodes):
    """Run the Stage A Robometer worker loop."""
    from rdf.stage_a_robometer.worker import run_worker
    run_worker(
        robometer_threshold=threshold,
        poll_wait=poll_wait,
        max_episodes=max_episodes,
    )


@cli.group()
def deminf_worker():
    """Stage B DemInf worker commands."""


@deminf_worker.command("run")
@click.option("--threshold", default=0.0, type=float, help="DemInf score threshold")
@click.option("--poll-wait", default=5, type=int, help="SQS long-poll wait seconds")
@click.option("--max-cohorts", default=None, type=int, help="Stop after N cohorts (testing)")
@click.option("--embodiment", default="franka", help="Embodiment config name")
def deminf_run(threshold, poll_wait, max_cohorts, embodiment):
    """Run the Stage B DemInf inference worker loop."""
    from rdf.stage_b_deminf.infer_worker import run_infer_worker
    run_infer_worker(
        deminf_threshold=threshold,
        poll_wait=poll_wait,
        max_cohorts=max_cohorts,
        embodiment=embodiment,
    )


@cli.command()
@click.argument("task")
@click.option("--robometer-threshold", default=0.5, type=float)
@click.option("--deminf-threshold", default=0.0, type=float)
def decide(task, robometer_threshold, deminf_threshold):
    """Run decision step for all episodes of a task."""
    from rdf.decision.decide import decide_all
    from rdf.harness.catalog import get_catalog
    catalog = get_catalog()
    counts = decide_all(task, catalog, robometer_threshold, deminf_threshold)
    click.echo(f"Decision counts for {task}: {counts}")


@cli.command()
@click.argument("task")
def materialize(task):
    """Copy kept episodes to clean bucket."""
    from rdf.decision.materialize import materialize_task
    from rdf.harness.catalog import get_catalog
    from rdf.harness.storage import get_object_store
    catalog = get_catalog()
    raw_store = get_object_store()
    clean_store = get_object_store()
    counts = materialize_task(task, catalog, raw_store, clean_store)
    click.echo(f"Materialization counts: {counts}")
