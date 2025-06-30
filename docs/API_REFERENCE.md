# API Reference

## Python API

### Certificate Generation

```python
from inference.generate_certificate import generate_barrier_certificate

result = generate_barrier_certificate(
    system_description: str,
    model_config: str = "finetuned",  # "base" or "finetuned"
    rag_k: int = 3,
    kb_path: str = None,  # Uses default from config
    conversation_id: str = None,
    domain_bounds: dict = None
)

# Returns:
{
    "certificate": str,  # Generated barrier certificate
    "confidence": float,  # Model confidence (0-1)
    "rag_context": list,  # Retrieved documents
    "verification": dict  # Verification results if enabled
}
```

### Verification

```python
from evaluation.verify_certificate import verify_barrier_certificate

result = verify_barrier_certificate(
    dynamics: dict,  # {"x": "expression", "y": "expression"}
    certificate: str,  # e.g., "x**2 + y**2"
    initial_set: str,  # e.g., "x**2 + y**2 <= 1"
    unsafe_set: str,   # e.g., "x >= 2"
    domain_bounds: dict = None
)

# Returns:
{
    "valid": bool,
    "numerical_check": bool,
    "symbolic_check": bool,
    "sos_check": bool,  # If applicable
    "details": dict
}
```

### Knowledge Base

```python
from knowledge_base.knowledge_base_builder import KnowledgeBaseBuilder

# Build knowledge base
kb = KnowledgeBaseBuilder(config)
kb.build(paper_dir="data/fetched_papers/")

# Query knowledge base
results = kb.search(query="barrier certificate", k=3)
```

## REST API

### Authentication

All API endpoints require authentication via API key:

```bash
-H "X-API-Key: your-api-key"
```

### Endpoints

#### Generate Certificate

```http
POST /api/generate
Content-Type: application/json

{
    "system_description": "string",
    "model_config": "base|finetuned",
    "rag_k": 3,
    "domain_bounds": {
        "x": [-2, 2],
        "y": [-2, 2]
    }
}

Response 200:
{
    "certificate": "string",
    "confidence": 0.95,
    "rag_context": [...],
    "request_id": "uuid"
}
```

#### Verify Certificate

```http
POST /api/verify
Content-Type: application/json

{
    "dynamics": {
        "x": "-x**3 - y",
        "y": "x - y**3"
    },
    "certificate": "x**4 + y**4",
    "initial_set": "x**2 + y**2 <= 0.1",
    "unsafe_set": "x >= 1.5"
}

Response 200:
{
    "valid": true,
    "checks": {
        "numerical": true,
        "symbolic": true,
        "sos": false
    }
}
```

#### Status Check

```http
GET /api/status

Response 200:
{
    "status": "healthy",
    "model_loaded": true,
    "kb_loaded": true,
    "gpu_available": true
}
```

#### User Metrics

```http
GET /api/metrics
X-API-Key: your-api-key

Response 200:
{
    "requests_today": 25,
    "requests_remaining": 25,
    "total_requests": 150,
    "success_rate": 0.92
}
```

### Monitoring API

#### Usage Metrics

```http
GET /monitoring/api/metrics/usage?range=month

Response 200:
{
    "total_requests": 1000,
    "successful_generations": 920,
    "active_users": 45,
    "avg_generation_time": 3.2
}
```

#### Health Check

```http
GET /monitoring/health

Response 200:
{
    "status": "healthy",
    "database": "healthy",
    "cpu_percent": 45.2,
    "memory_percent": 62.1
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid input |
| 401 | Unauthorized - Invalid API key |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Model loading |

## Rate Limits

- Default: 50 requests per day per user
- Can be configured in `config.yaml`
- Premium users may have higher limits

## Examples

### Python Client

```python
import requests

API_URL = "http://localhost:5000/api"
API_KEY = "your-api-key"

# Generate certificate
response = requests.post(
    f"{API_URL}/generate",
    headers={"X-API-Key": API_KEY},
    json={
        "system_description": "dx/dt = -x^3 - y, dy/dt = x - y^3",
        "model_config": "finetuned"
    }
)

result = response.json()
print(f"Certificate: {result['certificate']}")
```

### JavaScript Client

```javascript
const API_URL = 'http://localhost:5000/api';
const API_KEY = 'your-api-key';

async function generateCertificate(system) {
    const response = await fetch(`${API_URL}/generate`, {
        method: 'POST',
        headers: {
            'X-API-Key': API_KEY,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            system_description: system,
            model_config: 'finetuned'
        })
    });
    
    return await response.json();
}
``` 