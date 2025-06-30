# FM-LLM Solver Security Usage Guide

This guide explains how to use the security features implemented in the FM-LLM Solver web interface.

## Quick Start

### 1. Initial Setup

```bash
# Install dependencies
pip install -r requirements/requirements.txt

# Initialize security (creates database and admin user)
python scripts/init_security.py
```

During initialization, you'll be prompted to:
- Enter admin email (default: admin@example.com)
- Set admin password (must meet security requirements)
- Optionally create a demo user for testing

### 2. Starting the Application

```bash
python run_web_interface.py
```

The application will start with security features enabled at http://localhost:5000

## User Guide

### Registration

1. Navigate to http://localhost:5000/auth/register
2. Create an account with:
   - Username (letters, numbers, underscores only)
   - Valid email address
   - Strong password (8+ chars, uppercase, lowercase, numbers, special chars)
3. After registration, you'll be automatically logged in

### Login

1. Go to http://localhost:5000/auth/login
2. Enter username/email and password
3. Check "Remember me" for persistent sessions (optional)

### Using the Application

Once logged in, you can:
- Generate barrier certificates (limited to 50 per day)
- View your generation history
- Access your profile and usage statistics

### Profile Management

Access your profile at http://localhost:5000/auth/profile to:
- View usage statistics (requests today/remaining)
- Update email address
- Change password
- Generate/manage API keys

## API Usage

### Generating an API Key

1. Login to your account
2. Go to Profile → API Access tab
3. Click "Generate API Key"
4. Copy and securely store your key (shown only once)

### Using the API

```bash
# Generate certificate via API
curl -X POST http://localhost:5000/api/generate \
  -H "X-API-Key: YOUR_API_KEY_HERE" \
  -H "Content-Type: application/json" \
  -d '{
    "system_description": "x[k+1] = 2*x[k] - 1; Initial set: x in [-0.5, 0.5]; Unsafe set: x > 2 or x < -2",
    "model_config": "finetuned",
    "rag_k": 3
  }'
```

### API Response

```json
{
  "success": true,
  "query_id": 123,
  "certificate": "0.8 * x**2 - 1.0",
  "llm_output": "...",
  "context_chunks": 3
}
```

## Rate Limiting

### Default Limits
- **Regular users**: 50 requests/day
- **Admin users**: 1000 requests/day
- **Demo user**: 10 requests/day

### Checking Your Usage
- View remaining requests in your profile
- API responses include rate limit headers:
  - `X-RateLimit-Limit`: Your daily limit
  - `X-RateLimit-Remaining`: Requests remaining today
  - `X-RateLimit-Reset`: Time when limit resets

### Rate Limit Exceeded
When you exceed your limit:
- Web interface: Redirected to profile with error message
- API: HTTP 429 response with error details

## Security Best Practices

### For Users

1. **Strong Passwords**
   - Use unique passwords for each account
   - Enable password manager
   - Change passwords regularly

2. **API Key Security**
   - Never share your API key
   - Regenerate if compromised
   - Use environment variables in code:
   ```python
   import os
   api_key = os.environ.get('FM_LLM_API_KEY')
   ```

3. **Session Security**
   - Logout when finished
   - Don't use "Remember me" on shared computers
   - Report suspicious activity

### For Administrators

1. **Monitor Security Logs**
   - Access at: http://localhost:5000/auth/admin/security-logs
   - Watch for:
     - Failed login attempts
     - Rate limit violations
     - Blocked IPs

2. **User Management**
   - View users at: http://localhost:5000/auth/admin/users
   - Deactivate suspicious accounts
   - Adjust rate limits as needed

3. **Regular Maintenance**
   - Review security logs weekly
   - Update dependencies monthly
   - Backup database regularly

## Troubleshooting

### Common Issues

**Can't Login**
- Check username/password spelling
- Ensure account is active
- Clear browser cookies
- Check if IP is blocked

**Rate Limit Issues**
- Wait until midnight UTC for reset
- Contact admin for limit increase
- Use API key for higher limits

**API Key Not Working**
- Verify key is correct (no spaces)
- Check if key was revoked
- Ensure proper header format
- Verify account is active

### Security Incident Response

If you suspect a security issue:

1. **For Users**
   - Change password immediately
   - Revoke API keys
   - Contact administrator

2. **For Admins**
   - Check security logs
   - Block suspicious IPs
   - Review affected accounts
   - Document incident

## Configuration

Security settings in `config/config.yaml`:

```yaml
security:
  max_requests_per_day: 50
  password_min_length: 8
  session_lifetime_hours: 24
  brute_force_threshold: 5
  ip_block_duration_hours: 1
```

## Testing Security

Run security tests:

```bash
# Start the application first
python run_web_interface.py

# In another terminal
python tests/test_security.py
```

## Production Deployment

For production environments:

1. **Use HTTPS**
   ```nginx
   server {
       listen 443 ssl;
       ssl_certificate /path/to/cert.pem;
       ssl_certificate_key /path/to/key.pem;
       ...
   }
   ```

2. **Set Strong Secret Key**
   ```bash
   export SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
   ```

3. **Use Production Database**
   - PostgreSQL or MySQL instead of SQLite
   - Regular backups
   - Connection encryption

4. **Enable Monitoring**
   - Set up Prometheus/Grafana
   - Configure alerts for security events
   - Monitor rate limit violations

## Support

For security-related questions or to report vulnerabilities:
- Email: security@fm-llm-solver.example.com
- Do not post security issues publicly
- Include detailed reproduction steps 