FM-LLM Solver Documentation
===========================

Welcome to FM-LLM Solver's documentation! This project provides barrier certificate generation for dynamical systems using Large Language Models.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   INSTALLATION
   USER_GUIDE
   API_REFERENCE
   ARCHITECTURE
   DEVELOPMENT
   EXPERIMENTS

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics:

   MATHEMATICAL_PRIMER
   VERIFICATION
   OPTIMIZATION
   MONITORING
   SECURITY
   FEATURES

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Quick Start
-----------

To get started with FM-LLM Solver:

1. Install the package:

   .. code-block:: bash

      pip install -e .

2. Configure your environment:

   .. code-block:: bash

      cp config/env.example .env
      # Edit .env with your settings

3. Run the application:

   .. code-block:: bash

      fm-llm-solver

Features
--------

- **Barrier Certificate Generation**: Automatically generate barrier certificates for various dynamical systems
- **LLM Integration**: Leverages state-of-the-art language models for mathematical reasoning
- **Web Interface**: User-friendly web interface for system specification and certificate visualization
- **Verification**: Built-in verification of generated certificates
- **Knowledge Base**: RAG-powered system for enhanced certificate generation

Support
-------

For issues and questions:

- GitHub Issues: https://github.com/yourusername/FM-LLM-Solver/issues
- Documentation: https://fm-llm-solver.readthedocs.io 