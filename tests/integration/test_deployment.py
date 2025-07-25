#!/usr/bin/env python3
"""
Test script for deployment configuration verification.
Tests basic functionality without requiring actual cloud deployments.
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
import subprocess

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_configuration():
    """Test configuration loading and validation."""
    logger.info("Testing configuration loading...")
    
    try:
        # Test main config
        config_path = PROJECT_ROOT / "config" / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check deployment section
        assert 'deployment' in config, "Deployment section missing from config"
        assert config['deployment']['mode'] in ['local', 'hybrid', 'cloud'], "Invalid deployment mode"
        
        logger.info("✓ Configuration loaded successfully")
        logger.info(f"  - Deployment mode: {config['deployment']['mode']}")
        logger.info(f"  - Web interface port: {config['deployment']['local']['port']}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False

def test_docker_setup():
    """Test Docker configuration."""
    logger.info("\nTesting Docker setup...")
    
    try:
        # Check if Docker is installed
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✓ Docker installed: {result.stdout.strip()}")
        else:
            logger.error("✗ Docker not installed or not in PATH")
            return False
        
        # Check if docker-compose exists
        if (PROJECT_ROOT / "docker-compose.yml").exists():
            logger.info("✓ docker-compose.yml found")
            
            # Validate docker-compose file
            # Try new 'docker compose' syntax first, then fall back to 'docker-compose'
            try:
                result = subprocess.run(
                    ['docker', 'compose', 'config', '-q'],
                    cwd=PROJECT_ROOT,
                    capture_output=True
                )
            except:
                # Fall back to old syntax
                result = subprocess.run(
                    ['docker-compose', 'config', '-q'],
                    cwd=PROJECT_ROOT,
                    capture_output=True
                )
            
            if result.returncode == 0:
                logger.info("✓ docker-compose.yml is valid")
            else:
                logger.error("✗ docker-compose.yml validation failed")
                return False
        else:
            logger.error("✗ docker-compose.yml not found")
            return False
            
        # Check Dockerfile
        if (PROJECT_ROOT / "Dockerfile").exists():
            logger.info("✓ Dockerfile found")
        else:
            logger.error("✗ Dockerfile not found")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"✗ Docker setup test failed: {e}")
        return False

def test_inference_api_structure():
    """Test inference API module structure."""
    logger.info("\nTesting inference API structure...")
    
    try:
        # Check if inference API directory exists
        inference_api_dir = PROJECT_ROOT / "inference_api"
        if not inference_api_dir.exists():
            logger.error("✗ inference_api directory not found")
            return False
        
        logger.info("✓ inference_api directory found")
        
        # Check key files
        required_files = ['__init__.py', 'main.py']
        for file in required_files:
            if (inference_api_dir / file).exists():
                logger.info(f"✓ {file} found")
            else:
                logger.error(f"✗ {file} not found")
                return False
        
        # Try importing the module
        try:
            import inference_api
            logger.info("✓ inference_api module can be imported")
        except ImportError as e:
            logger.error(f"✗ Cannot import inference_api: {e}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"✗ Inference API structure test failed: {e}")
        return False

def test_deployment_scripts():
    """Test deployment scripts."""
    logger.info("\nTesting deployment scripts...")
    
    try:
        deployment_dir = PROJECT_ROOT / "deployment"
        
        # Check deployment directory
        if not deployment_dir.exists():
            logger.error("✗ deployment directory not found")
            return False
            
        logger.info("✓ deployment directory found")
        
        # Check key scripts
        scripts = ['__init__.py', 'deploy.py', 'README.md']
        for script in scripts:
            if (deployment_dir / script).exists():
                logger.info(f"✓ {script} found")
            else:
                logger.error(f"✗ {script} not found")
                return False
        
        # Test deploy.py syntax
        result = subprocess.run(
            [sys.executable, '-m', 'py_compile', str(deployment_dir / 'deploy.py')],
            capture_output=True
        )
        if result.returncode == 0:
            logger.info("✓ deploy.py syntax is valid")
        else:
            logger.error("✗ deploy.py has syntax errors")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"✗ Deployment scripts test failed: {e}")
        return False

def test_environment_template():
    """Test environment configuration template."""
    logger.info("\nTesting environment configuration...")
    
    try:
        env_example = PROJECT_ROOT / "config" / "env.example"
        
        if env_example.exists():
            logger.info("✓ env.example found")
            
            # Check key environment variables
            with open(env_example, 'r') as f:
                content = f.read()
                
            required_vars = [
                'DEPLOYMENT_MODE',
                'INFERENCE_API_URL',
                'SECRET_KEY',
                'MATHPIX_APP_ID',
                'MATHPIX_APP_KEY'
            ]
            
            for var in required_vars:
                if var in content:
                    logger.info(f"✓ {var} defined in env.example")
                else:
                    logger.error(f"✗ {var} missing from env.example")
                    return False
        else:
            logger.error("✗ env.example not found")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"✗ Environment template test failed: {e}")
        return False

def test_web_interface_compatibility():
    """Test web interface compatibility with deployment modes."""
    logger.info("\nTesting web interface compatibility...")
    
    try:
        # Check if certificate generator supports deployment modes
        cert_gen_path = PROJECT_ROOT / "web_interface" / "certificate_generator.py"
        
        if cert_gen_path.exists():
            with open(cert_gen_path, 'r') as f:
                content = f.read()
                
            # Check for deployment mode support
            if 'deployment_mode' in content and '_generate_remote' in content:
                logger.info("✓ Web interface supports hybrid/cloud deployment")
            else:
                logger.warning("⚠ Web interface may not fully support remote inference")
        else:
            logger.error("✗ certificate_generator.py not found")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"✗ Web interface compatibility test failed: {e}")
        return False

def run_all_tests():
    """Run all deployment tests."""
    logger.info("=" * 60)
    logger.info("FM-LLM Solver Deployment Configuration Test")
    logger.info("=" * 60)
    
    tests = [
        ("Configuration", test_configuration),
        ("Docker Setup", test_docker_setup),
        ("Inference API Structure", test_inference_api_structure),
        ("Deployment Scripts", test_deployment_scripts),
        ("Environment Template", test_environment_template),
        ("Web Interface Compatibility", test_web_interface_compatibility)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n✓ All deployment tests passed! Ready for deployment.")
        return 0
    else:
        logger.error(f"\n✗ {total - passed} tests failed. Please fix issues before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests()) 