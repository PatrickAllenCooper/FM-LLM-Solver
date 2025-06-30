# Security Guide

FM-LLM Solver includes comprehensive security features to protect against common threats and ensure safe operation.

## Features

### Authentication & Authorization
- **User Registration**: Email validation and strong password requirements (8+ chars, mixed case, numbers, special chars)
- **Secure Login**: Session-based authentication with Flask-Login
- **Role-Based Access**: Regular users, premium users, and admins
- **API Keys**: Secure programmatic access with individual keys

### Rate Limiting
- **Default**: 50 requests per day per user
- **Configurable**: Adjust in `config.yaml`
- **Daily Reset**: Counters reset at midnight UTC
- **Violation Tracking**: Monitors and logs rate limit violations

### Protection Mechanisms
- **DDoS Protection**: IP blacklisting and request throttling
- **Brute Force Protection**: 5 failed logins trigger temporary IP block
- **CSRF Protection**: Tokens on all forms
- **XSS Prevention**: Automatic template escaping
- **SQL Injection Prevention**: Parameterized queries via SQLAlchemy
- **Security Headers**: X-Frame-Options, X-Content-Type-Options, CSP

## Setup

### 1. Initialize Security

```bash
python scripts/init_security.py
```

This creates:
- SQLite database with security tables
- Default admin user (admin/admin - change immediately!)
- Security configuration

### 2. Configure Settings

Edit `config/config.yaml`:

```yaml
security:
  rate_limit:
    requests_per_day: 50
    premium_requests_per_day: 200
  
  session:
    secret_key: "your-secret-key-here"  # Generate a strong key
    permanent_session_lifetime: 86400    # 24 hours
  
  password:
    min_length: 8
    require_uppercase: true
    require_lowercase: true
    require_numbers: true
    require_special: true
```

### 3. Start with Security

```bash
python run_web_interface.py
```

## Usage

### Web Interface

1. **Register**: http://localhost:5000/auth/register
2. **Login**: http://localhost:5000/auth/login
3. **Profile**: View/manage API keys at http://localhost:5000/auth/profile

### API Access

```bash
# Get your API key from profile page
curl -X POST http://localhost:5000/api/generate \
  -H "X-API-Key: your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{"system_description": "..."}'
```

### Admin Panel

Access admin features at http://localhost:5000/auth/admin/users (admin login required)

- View all users
- Adjust rate limits
- Activate/deactivate accounts
- Monitor security events

## Best Practices

1. **Change Default Admin Password**: Immediately after setup
2. **Use HTTPS in Production**: Configure SSL/TLS
3. **Regular Backups**: Backup the database regularly
4. **Monitor Logs**: Check security logs for suspicious activity
5. **Update Dependencies**: Keep security packages updated

## Monitoring

View security events in the monitoring dashboard:
- Failed login attempts
- Rate limit violations
- Blocked IPs
- API usage patterns

## Emergency Procedures

### Block an IP
```python
from web_interface.models import IPBlacklist
ip = IPBlacklist(ip_address="1.2.3.4", reason="Manual block")
db.session.add(ip)
db.session.commit()
```

### Reset User Password
```python
from web_interface.models import User
user = User.query.filter_by(username="user").first()
user.set_password("new_password")
db.session.commit()
```

### Disable All Access
Set rate limits to 0 in config and restart the application. 