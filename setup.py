"""
Setup configuration for FM-LLM Solver.

Install with: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read version from package
version_file = Path(__file__).parent / "fm_llm_solver" / "__init__.py"
version = "1.0.0"
if version_file.exists():
    with open(version_file) as f:
        for line in f:
            if line.startswith("__version__"):
                version = line.split("=")[1].strip().strip('"').strip("'")
                break

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, encoding="utf-8") as f:
        long_description = f.read()

# Core requirements (minimal for basic functionality)
install_requires = [
    "pydantic>=2.0.0",
    "PyYAML>=6.0",
    "numpy>=1.24.0",
    "sympy>=1.12",
    "torch>=2.0.0",
    "transformers>=4.35.0",
    "Flask>=3.0.0",
    "SQLAlchemy>=2.0.0",
    "requests>=2.31.0",
    "tqdm>=4.66.0",
]

# Optional dependencies
extras_require = {
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "black>=23.12.0",
        "isort>=5.13.0",
        "flake8>=6.1.0",
        "mypy>=1.7.0",
        "pre-commit>=3.5.0",
    ],
    "gpu": [
        "faiss-gpu>=1.7.4",
        "flash-attn>=2.3.0",
    ],
    "web": [
        "Flask-SQLAlchemy>=3.1.0",
        "Flask-Login>=0.6.3",
        "Flask-Limiter>=3.5.0",
        "Flask-CORS>=4.0.0",
        "Flask-Migrate>=4.0.0",
        "redis>=5.0.0",
        "gunicorn>=21.2.0",
    ],
    "api": [
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "python-multipart>=0.0.6",
    ],
    "rag": [
        "faiss-cpu>=1.7.4",
        "sentence-transformers>=2.2.0",
        "langchain>=0.1.0",
        "pypdf>=3.17.0",
    ],
    "monitoring": [
        "prometheus-client>=0.19.0",
        "psutil>=5.9.0",
        "GPUtil>=1.4.0",
    ],
    "all": [
        # Include everything
    ],
}

# Combine all extras for 'all'
all_extras = []
for extra_deps in extras_require.values():
    if isinstance(extra_deps, list):
        all_extras.extend(extra_deps)
extras_require["all"] = list(set(all_extras))

setup(
    name="fm-llm-solver",
    version=version,
    author="Patrick Allen Cooper",
    author_email="patrick.cooper@colorado.edu",
    description="Barrier certificate generation for dynamical systems using Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/FM-LLM-Solver",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/FM-LLM-Solver/issues",
        "Documentation": "https://fm-llm-solver.readthedocs.io",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(exclude=["tests", "tests.*", "scripts", "docs"]),
    package_data={
        "fm_llm_solver": [
            "web/templates/**/*.html",
            "web/static/**/*",
        ],
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "fm-llm-solver=run_application:main",
            "fmllm=run_application:main",
        ],
    },
    zip_safe=False,
) 