"""Promote models between stages in MLflow registry."""

import click

from src.models import ModelRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


@click.command()
@click.option(
    "--version",
    required=True,
    type=str,
    help="Model version to promote",
)
@click.option(
    "--stage",
    required=True,
    type=click.Choice(["Staging", "Production", "Archived"]),
    help="Target stage",
)
def promote(version: str, stage: str):
    """Promote model to a new stage.

    Args:
        version: Model version number.
        stage: Target stage (Staging, Production, Archived).
    """
    try:
        registry = ModelRegistry()
        registry.transition_stage(version, stage)
        click.echo(f"✓ Model v{version} promoted to {stage}")
        logger.info(f"Model v{version} promoted to {stage}")
    except Exception as e:
        click.echo(f"✗ Promotion failed: {e}", err=True)
        logger.error(f"Promotion failed: {e}")
        raise


@click.group()
def cli():
    """Model promotion CLI."""
    pass


@cli.command()
@click.option(
    "--version",
    required=True,
    type=str,
    help="Model version to promote",
)
@click.option(
    "--stage",
    required=True,
    type=click.Choice(["Staging", "Production", "Archived"]),
    help="Target stage",
)
def transition(version: str, stage: str):
    """Transition model to a new stage."""
    promote.callback(version, stage)


@cli.command()
@click.option(
    "--stage",
    default="Production",
    type=click.Choice(["Staging", "Production", "Archived"]),
    help="Stage to get latest version",
)
def latest(stage: str):
    """Get latest model version in a stage."""
    try:
        registry = ModelRegistry()
        version = registry.get_latest_version(stage)
        if version:
            click.echo(f"Latest {stage} model version: {version}")
        else:
            click.echo(f"No models found in {stage} stage")
    except Exception as e:
        click.echo(f"✗ Query failed: {e}", err=True)
        raise


if __name__ == "__main__":
    cli()
