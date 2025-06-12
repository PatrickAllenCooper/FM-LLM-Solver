# FM-LLM Solver Web Interface

A modern web interface for the FM-LLM Solver barrier certificate generation system, built with Flask and Material 3 Expressive design.

## Features

### 🎯 Core Functionality
- **Interactive Query Interface**: Submit system descriptions through an intuitive web form
- **Multiple Model Support**: Choose between base models, fine-tuned variants, and specialized discrete/continuous models
- **Real-time Processing**: Live progress updates with background task processing
- **Comprehensive Verification**: View detailed verification results including numerical, symbolic, and SOS checks
- **Query History**: Browse and search through previous queries with pagination
- **Database Recording**: All queries and results are automatically recorded to SQLite database

### 🎨 User Experience
- **Material 3 Design**: Modern, accessible interface following Google's Material Design 3 principles
- **Responsive Layout**: Works seamlessly on desktop, tablet, and mobile devices
- **Progress Tracking**: Real-time status updates with visual progress indicators
- **Interactive Results**: Copy certificates, view detailed breakdowns, and explore verification feedback
- **Rich Feedback**: Clear success/error states with actionable error messages

### 🔧 Technical Features
- **Modular Architecture**: Clean separation between services, models, and presentation
- **Background Processing**: Non-blocking certificate generation and verification
- **Configuration-Driven**: Uses existing `config.yaml` for all settings
- **Database Integration**: SQLite database with comprehensive query logging
- **RESTful API**: JSON endpoints for programmatic access

## Quick Start

### Prerequisites

1. **Project Setup**: Ensure the main FM-LLM Solver project is properly configured
2. **Knowledge Base**: Build at least one knowledge base using `knowledge_base/knowledge_base_builder.py`
3. **Python Environment**: Python 3.10+ with project dependencies installed

### Installation

1. **Install Web Dependencies**:
   ```bash
   pip install -r web_requirements.txt
   ```

2. **Set Environment Variables** (optional):
   ```bash
   export SECRET_KEY="your-secret-key-for-production"
   ```

3. **Launch the Web Interface**:
   ```bash
   python run_web_interface.py
   ```

The interface will be available at `http://127.0.0.1:5000`

### Command Line Options

```bash
# Basic usage
python run_web_interface.py

# Custom configuration
python run_web_interface.py --config my_config.yaml

# Different host/port
python run_web_interface.py --host 0.0.0.0 --port 8080

# Production mode (disable debug)
python run_web_interface.py --no-debug

# Development mode (enable debug)
python run_web_interface.py --debug
```

## Configuration

The web interface uses the existing `config.yaml` file. Add the following section to configure web-specific settings:

```yaml
# Web interface configuration
web_interface:
  host: "127.0.0.1"
  port: 5000
  debug: true
  database_path: "web_interface/instance/app.db"
  # Security settings
  secret_key_env: "SECRET_KEY"  # Environment variable name for secret key
  # CORS settings for API access
  cors_origins: ["http://localhost:3000", "http://127.0.0.1:3000"]
```

## Usage Guide

### 1. Generating Certificates

1. **Navigate to Home Page**: Open the web interface and go to the main page
2. **Enter System Description**: Provide a clear description including:
   - System dynamics (e.g., `dx/dt = -x**3 - y, dy/dt = x - y**3`)
   - Initial set (e.g., `x**2 + y**2 <= 0.1`)
   - Unsafe set (e.g., `x >= 1.5`)
   - State variables (optional, will be inferred)

3. **Select Model Configuration**:
   - **Base Model**: Unmodified language model
   - **Fine-tuned Model**: Specialized for barrier certificates
   - **Discrete/Continuous**: Domain-specific variants (if available)

4. **Configure RAG**: Choose number of context chunks (0-10)
   - 0: No retrieval (faster, less context)
   - 3: Balanced (recommended)
   - 5-10: More context (slower, potentially better results)

5. **Submit and Monitor**: Click "Generate Certificate" and watch real-time progress

### 2. Understanding Results

#### Certificate Display
- Generated certificates are shown in mathematical notation
- Copy button allows easy copying for external use
- Failed generations show appropriate error messages

#### Verification Results
- **Numerical**: Monte Carlo sampling verification
- **Symbolic**: SymPy-based symbolic analysis  
- **SOS**: Sum-of-Squares programming verification
- **Overall**: Combined assessment of all checks

#### Detailed View
- Click "View Full Details" for comprehensive results
- Includes full LLM output, verification details, and metadata
- JSON format for detailed verification information

### 3. Query History

- **Browse Previous Queries**: Access all past queries with pagination
- **Statistics Dashboard**: View success rates and verification statistics
- **Search and Filter**: Find specific queries by status or content
- **Detailed Analysis**: Click any query to view complete results

## Architecture

### Backend Services

- **`certificate_generator.py`**: Handles model loading and certificate generation
- **`verification_service.py`**: Manages certificate verification pipeline
- **`models.py`**: SQLAlchemy database models for data persistence
- **`app.py`**: Main Flask application with routing and API endpoints

### Database Schema

- **QueryLog**: Stores user queries, model configurations, and results
- **VerificationResult**: Detailed verification outcomes and metadata
- **ModelConfiguration**: Available model configurations (extensible)
- **SystemBenchmark**: Predefined test systems (future feature)

### Frontend Design

- **Material 3 Expressive**: Modern design language with accessibility focus
- **Progressive Enhancement**: Works with JavaScript disabled (basic functionality)
- **Responsive Grid**: Adapts to different screen sizes seamlessly
- **Interactive Elements**: Real-time updates without page refreshes

## API Endpoints

### Core Endpoints

- `GET /`: Main interface
- `POST /query`: Submit new barrier certificate generation request
- `GET /task_status/<task_id>`: Check status of background task
- `GET /query/<query_id>`: View detailed results for specific query
- `GET /history`: Browse query history with pagination
- `GET /api/stats`: Application statistics JSON

### Example API Usage

```python
import requests

# Submit a query
response = requests.post('http://localhost:5000/query', json={
    'system_description': 'System Dynamics: dx/dt = -x**3 - y, dy/dt = x - y**3\nInitial Set: x**2 + y**2 <= 0.1\nUnsafe Set: x >= 1.5',
    'model_config': 'finetuned',
    'rag_k': 3
})

task_id = response.json()['task_id']

# Check status
status_response = requests.get(f'http://localhost:5000/task_status/{task_id}')
print(status_response.json())
```

## Troubleshooting

### Common Issues

1. **"No knowledge base found"**
   - Run `python knowledge_base/knowledge_base_builder.py` first
   - Check that FAISS index and metadata files exist

2. **"Model loading failed"**
   - Ensure PyTorch is installed with CUDA support (for GPU)
   - Check that base model name is correct in config
   - Verify adapter paths exist for fine-tuned models

3. **"Verification errors"**
   - Install missing dependencies: `pip install sympy cvxpy`
   - For SOS verification, ensure MOSEK license or install SCS solver

4. **"Database errors"**
   - Check write permissions for database directory
   - Ensure SQLite is available
   - Delete database file to reset if corrupted

### Performance Optimization

1. **Model Loading**: Models are cached after first load
2. **Knowledge Base**: FAISS indexes are loaded once per session
3. **Database**: SQLite with WAL mode for concurrent access
4. **Background Processing**: Prevents UI blocking during generation

### Production Deployment

For production use, consider:

```bash
# Install production server
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 web_interface.app:app

# Or with environment variables
export SECRET_KEY="your-production-secret-key"
export FLASK_ENV="production"
python run_web_interface.py --no-debug --host 0.0.0.0
```

## Development

### Project Structure

```
web_interface/
├── __init__.py              # Module initialization
├── app.py                   # Main Flask application
├── models.py                # Database models
├── certificate_generator.py # Certificate generation service
├── verification_service.py  # Verification service
├── templates/               # Jinja2 templates
│   ├── base.html           # Base template with Material 3 styles
│   ├── index.html          # Main interface
│   ├── query_detail.html   # Detailed query view
│   ├── history.html        # Query history
│   └── about.html          # Project information
└── instance/               # Database and instance files
    └── app.db              # SQLite database
```

### Adding Features

1. **New Model Types**: Extend `certificate_generator.py` with additional model configurations
2. **Custom Verification**: Add verification methods in `verification_service.py`
3. **UI Components**: Follow Material 3 design tokens in `base.html`
4. **API Endpoints**: Add routes in `app.py` following RESTful conventions

### Testing

```bash
# Install test dependencies
pip install pytest flask-testing

# Run tests (when available)
pytest web_interface/tests/

# Manual testing
python run_web_interface.py --debug
```

## Contributing

1. **Code Style**: Follow PEP 8 and existing code patterns
2. **Documentation**: Update this README for significant changes
3. **Testing**: Add tests for new functionality
4. **Material Design**: Maintain consistency with Material 3 principles

## License

This web interface is part of the FM-LLM Solver project and follows the same licensing terms. 