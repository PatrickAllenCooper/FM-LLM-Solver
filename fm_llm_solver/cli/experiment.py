"""
Experiment CLI commands for FM-LLM Solver.
"""

import click
from fm_llm_solver.core.logging import get_logger


@click.group()
def experiment():
    """Experiment management commands."""
    pass


@experiment.command()
@click.option('--config-file', type=click.Path(exists=True), help='Experiment configuration file')
@click.pass_context  
def run(ctx, config_file):
    """Run experiments."""
    logger = get_logger('experiment.run')
    click.echo("🧪 Running experiments...")
    # TODO: Implement experiment runner


@experiment.command()
@click.pass_context
def list_experiments(ctx):
    """List available experiments."""
    click.echo("📋 Available experiments:")
    # TODO: Implement experiment listing 