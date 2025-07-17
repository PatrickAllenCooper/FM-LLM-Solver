"""
Web interface CLI commands for FM-LLM Solver.
"""

import click
from fm_llm_solver.core.logging import get_logger


@click.group()
def web():
    """Web interface management commands."""
    pass


@web.command()
@click.option("--host", default="localhost", help="Host to bind to")
@click.option("--port", default=5000, help="Port to bind to")
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.pass_context
def run(ctx, host: str, port: int, debug: bool):
    """Run the web interface."""
    config = ctx.obj["config"]
    logger = get_logger("web.run")

    click.echo(f"🌐 Starting web interface on {host}:{port}")

    try:
        from fm_llm_solver.web.app import create_app

        app = create_app()
        app.run(host=host, port=port, debug=debug)
    except Exception as e:
        logger.error(f"Failed to start web interface: {e}")
        click.echo(f"❌ Error: {e}")


@web.command()
@click.pass_context
def init_db(ctx):
    """Initialize the web interface database."""
    click.echo("🗄️  Initializing database...")

    try:
        from fm_llm_solver.web.app import create_app

        app = create_app()
        with app.app_context():
            from fm_llm_solver.web.models import db

            db.create_all()
        click.echo("✅ Database initialized")
    except Exception as e:
        click.echo(f"❌ Database initialization failed: {e}")
