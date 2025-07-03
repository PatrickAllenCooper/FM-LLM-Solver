"""
Configuration CLI commands for FM-LLM Solver.
"""

import click
import json
from pathlib import Path
from typing import Dict, Any

from fm_llm_solver.core.logging import get_logger
from fm_llm_solver.core.config_manager import (
    ConfigurationManager, 
    Environment, 
    SecretProvider,
    ConfigurationTemplate
)


@click.group()
def config():
    """Configuration management commands."""
    pass


@config.command()
@click.option('--environment', type=click.Choice(['development', 'testing', 'staging', 'production']),
              default='development', help='Target environment')
@click.option('--output-dir', type=click.Path(), help='Output directory for config')
@click.pass_context
def init(ctx, environment: str, output_dir: str):
    """Initialize configuration for an environment."""
    logger = get_logger('config.init')
    
    if not output_dir:
        output_dir = Path("config")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    click.echo(f"🔧 Initializing configuration for {environment}")
    
    try:
        env = Environment(environment)
        config_manager = ConfigurationManager(config_dir=output_dir, environment=env)
        
        # Create template based on environment
        if env == Environment.DEVELOPMENT:
            template = ConfigurationTemplate(
                environment=env,
                template_vars={
                    'model.device': 'cuda',
                    'model.quantization': '4bit',
                    'deployment.mode': 'local',
                    'rag.enabled': True,
                    'debug': True
                },
                required_secrets=[],
                optional_secrets=['MATHPIX_APP_ID', 'MATHPIX_APP_KEY']
            )
        elif env == Environment.PRODUCTION:
            template = ConfigurationTemplate(
                environment=env,
                template_vars={
                    'model.device': 'cuda',
                    'model.quantization': '8bit',
                    'deployment.mode': 'cloud',
                    'rag.enabled': True,
                    'debug': False,
                    'security.rate_limit.requests_per_day': 500
                },
                required_secrets=[
                    'SECRET_KEY',
                    'DATABASE_URL',
                    'REDIS_URL'
                ],
                optional_secrets=[
                    'MATHPIX_APP_ID',
                    'MATHPIX_APP_KEY',
                    'AWS_ACCESS_KEY_ID',
                    'AWS_SECRET_ACCESS_KEY'
                ]
            )
        else:
            template = ConfigurationTemplate(
                environment=env,
                template_vars={
                    'model.device': 'cuda',
                    'deployment.mode': 'cloud',
                    'rag.enabled': True
                },
                required_secrets=['SECRET_KEY'],
                optional_secrets=['DATABASE_URL', 'REDIS_URL']
            )
        
        # Save template
        output_path = config_manager.save_template(template)
        
        click.echo(f"✅ Configuration template created: {output_path}")
        
        # Show next steps
        click.echo(f"\n📋 Next steps:")
        click.echo(f"1. Edit {output_path} to customize settings")
        
        if template.required_secrets:
            click.echo(f"2. Set required secrets:")
            for secret in template.required_secrets:
                click.echo(f"   export {secret}=your_value")
        
        if template.optional_secrets:
            click.echo(f"3. Set optional secrets (if needed):")
            for secret in template.optional_secrets:
                click.echo(f"   export {secret}=your_value")
        
        click.echo(f"4. Test configuration: fm-llm config validate")
        
    except Exception as e:
        logger.error(f"Failed to initialize configuration: {e}")
        click.echo(f"❌ Error: {e}")


@config.command()
@click.option('--config-file', type=click.Path(exists=True), help='Configuration file to validate')
@click.option('--environment', type=click.Choice(['development', 'testing', 'staging', 'production']),
              help='Environment to validate for')
@click.pass_context
def validate(ctx, config_file: str, environment: str):
    """Validate configuration file."""
    logger = get_logger('config.validate')
    
    click.echo("🔍 Validating configuration...")
    
    try:
        # Determine config details
        if config_file:
            config_dir = Path(config_file).parent
            config_name = Path(config_file).stem
        else:
            config_dir = "config"
            config_name = "config"
        
        env = Environment(environment) if environment else None
        config_manager = ConfigurationManager(config_dir=config_dir, environment=env)
        
        # Load and validate
        config = config_manager.load_config(config_name, validate=True)
        
        click.echo("✅ Configuration is valid!")
        
        # Show summary
        click.echo(f"\n📊 Configuration Summary:")
        click.echo(f"  • Environment: {config_manager.environment.value}")
        click.echo(f"  • Model: {config.model.name}")
        click.echo(f"  • Provider: {config.model.provider}")
        click.echo(f"  • Deployment: {config.deployment.mode}")
        click.echo(f"  • RAG enabled: {config.rag.enabled}")
        click.echo(f"  • Debug mode: {config.debug}")
        
        # Show warnings if any
        from fm_llm_solver.core.config import validate_config
        warnings = validate_config(config)
        if warnings:
            click.echo(f"\n⚠️  Warnings:")
            for warning in warnings:
                click.echo(f"  • {warning}")
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        click.echo(f"❌ Validation failed: {e}")


@config.command()
@click.pass_context
def show(ctx):
    """Show current configuration."""
    try:
        config = ctx.obj['config']
        
        # Convert to dict for display
        from omegaconf import OmegaConf
        config_dict = OmegaConf.to_container(config, resolve=True)
        
        click.echo("📋 Current Configuration:")
        click.echo(json.dumps(config_dict, indent=2))
        
    except Exception as e:
        click.echo(f"❌ Error showing config: {e}")


@config.command()
@click.option('--key', required=True, help='Configuration key (e.g., model.name)')
@click.option('--value', required=True, help='New value')
@click.option('--config-file', type=click.Path(), help='Configuration file to update')
@click.pass_context
def set(ctx, key: str, value: str, config_file: str):
    """Set a configuration value."""
    logger = get_logger('config.set')
    
    if not config_file:
        config_file = "config/config.yaml"
    
    click.echo(f"🔧 Setting {key} = {value}")
    
    try:
        from omegaconf import OmegaConf
        
        # Load existing config
        if Path(config_file).exists():
            config = OmegaConf.load(config_file)
        else:
            config = OmegaConf.create({})
        
        # Convert value to appropriate type
        from fm_llm_solver.core.config import convert_env_value
        typed_value = convert_env_value(value)
        
        # Set the value
        OmegaConf.set(config, key, typed_value)
        
        # Save back to file
        Path(config_file).parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, config_file)
        
        click.echo(f"✅ Configuration updated: {config_file}")
        
    except Exception as e:
        logger.error(f"Failed to set configuration: {e}")
        click.echo(f"❌ Error: {e}")


@config.command()
@click.option('--environment', help='Environment to show info for')
@click.pass_context
def info(ctx, environment: str):
    """Show configuration environment information."""
    try:
        env = Environment(environment) if environment else None
        config_manager = ConfigurationManager(environment=env)
        
        info = config_manager.get_environment_info()
        
        click.echo("🔍 Configuration Environment Info:")
        click.echo(f"  • Environment: {info['environment']}")
        click.echo(f"  • Config directory: {info['config_dir']}")
        click.echo(f"  • Secret provider: {info['secret_provider']}")
        click.echo(f"  • Has cached config: {info['has_cached_config']}")
        
        if info['available_configs']:
            click.echo(f"  • Available configs: {', '.join(info['available_configs'])}")
        else:
            click.echo("  • No configuration files found")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")


@config.command()
@click.option('--source-env', type=click.Choice(['development', 'testing', 'staging', 'production']),
              required=True, help='Source environment')
@click.option('--target-env', type=click.Choice(['development', 'testing', 'staging', 'production']),
              required=True, help='Target environment')
@click.option('--config-dir', type=click.Path(), default='config', help='Configuration directory')
@click.pass_context
def migrate(ctx, source_env: str, target_env: str, config_dir: str):
    """Migrate configuration from one environment to another."""
    logger = get_logger('config.migrate')
    
    click.echo(f"🔄 Migrating configuration from {source_env} to {target_env}")
    
    try:
        config_dir = Path(config_dir)
        
        # Load source config
        source_path = config_dir / f"config.{source_env}.yaml"
        if not source_path.exists():
            click.echo(f"❌ Source configuration not found: {source_path}")
            return
        
        from omegaconf import OmegaConf
        source_config = OmegaConf.load(source_path)
        
        # Create target config with environment-specific modifications
        target_config = source_config.copy()
        
        # Apply environment-specific changes
        if target_env == 'production':
            OmegaConf.set(target_config, 'debug', False)
            OmegaConf.set(target_config, 'deployment.mode', 'cloud')
            OmegaConf.set(target_config, 'security.rate_limit.requests_per_day', 500)
        elif target_env == 'development':
            OmegaConf.set(target_config, 'debug', True)
            OmegaConf.set(target_config, 'deployment.mode', 'local')
        
        # Save target config
        target_path = config_dir / f"config.{target_env}.yaml"
        OmegaConf.save(target_config, target_path)
        
        click.echo(f"✅ Configuration migrated to: {target_path}")
        click.echo(f"📝 Review and adjust settings as needed for {target_env}")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        click.echo(f"❌ Error: {e}") 