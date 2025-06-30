# FM-LLM Solver Security Implementation

## Overview

This document outlines the comprehensive security mechanisms implemented in the FM-LLM Solver web interface to prevent abuse, overuse, and attacks.

## Security Features

### 1. User Authentication

- **Registration & Login**: Users must create an account with a strong password
- **Password Requirements**:
  - Minimum 8 characters
  - Must contain uppercase and lowercase letters
  - Must contain numbers
  - Must contain special characters
- **Session Management**: Flask-Login handles secure sessions
- **Remember Me**: Optional persistent sessions with secure cookies

### 2. Rate Limiting

- **Per-User Limits**: 50 requests per day (configurable per user)
- **Daily Reset**: Counters reset at midnight UTC
- **API Rate Limiting**: Same limits apply to API access
- **Customizable Limits**: Admin can set different limits for different user roles

### 3. DDoS Protection

- **IP Blacklisting**: Automatic blocking of suspicious IPs
- **Brute Force Detection**: 5 failed login attempts trigger temporary IP block
- **Request Throttling**: Built-in Flask rate limiting
- **Security Headers**: Multiple headers to prevent common attacks

### 4. Security Headers

All responses include:
- `X-Frame-Options: SAMEORIGIN` - Prevents clickjacking
- `X-Content-Type-Options: nosniff` - Prevents MIME sniffing
- `X-XSS-Protection: 1; mode=block` - XSS protection
- `Content-Security-Policy` - Restricts resource loading
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy` - Disables unnecessary browser features

### 5. Input Validation

- **Server-side validation** for all inputs
- **SQL injection prevention** via SQLAlchemy ORM
- **XSS prevention** through template escaping
- **CSRF protection** with tokens on all forms
- **File upload restrictions** (if implemented)

### 6. API Security

- **API Key Authentication**: Secure token-based access
- **Key Generation**: Cryptographically secure 48-character keys
- **Rate Limiting**: Same daily limits apply
- **HTTPS Enforcement**: Recommended for production

### 7. Logging & Monitoring

- **Security Event Logging**: All security-relevant events logged
- **Failed Login Tracking**: Monitor brute force attempts
- **Rate Limit Violations**: Track and analyze patterns
- **Admin Dashboard**: View security logs and user activity

### 8. User Roles

- **User**: Standard access, 50 requests/day
- **Premium**: Enhanced limits (customizable)
- **Admin**: Full access, 1000 requests/day, user management

## Implementation Details

### Database Schema

New tables added:
- `users`: User accounts with hashed passwords
- `rate_limit_logs`: Track rate limit violations
- `ip_blacklist`: Blocked IP addresses
- `security_logs`: Security event audit trail

### Authentication Flow

1. User registers with email and strong password
2. Password hashed using Werkzeug's PBKDF2
3. Session created on successful login
4. CSRF token required for all state-changing operations

### Rate Limiting Implementation

```python
@rate_limit(max_requests=50)
def protected_route():
    # Automatically checks and enforces rate limits
    pass
```

### API Usage

```bash
# Using API key
curl -X POST http://localhost:5000/api/generate \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"system_description": "..."}'
```

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements/requirements.txt
   ```

2. **Initialize Security**:
   ```bash
   python scripts/init_security.py
   ```

3. **Create Admin Account**: Follow prompts to set admin credentials

4. **Start Application**:
   ```bash
   python run_web_interface.py
   ```

## Security Best Practices

### For Deployment

1. **Use HTTPS**: Always use SSL/TLS in production
2. **Strong Secret Key**: Set a strong SECRET_KEY environment variable
3. **Database Security**: Use PostgreSQL/MySQL instead of SQLite
4. **Firewall Rules**: Restrict access to necessary ports only
5. **Regular Updates**: Keep dependencies updated

### For Users

1. **Strong Passwords**: Use unique, complex passwords
2. **API Key Security**: Keep API keys confidential
3. **Report Issues**: Report suspicious activity to admins

## Additional Protections

### Against Common Attacks

- **SQL Injection**: ORM with parameterized queries
- **XSS**: Template auto-escaping, CSP headers
- **CSRF**: Token validation on forms
- **Session Hijacking**: Secure session cookies
- **Brute Force**: Account lockout, IP blocking
- **DDoS**: Rate limiting, IP blacklisting

### Monitoring & Alerts

- Security logs track all authentication events
- Failed login attempts are monitored
- Rate limit violations are logged
- Admin can view all security events

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

## Emergency Procedures

### If Under Attack

1. **Enable Maintenance Mode**: Temporarily disable the service
2. **Review Logs**: Check security_logs for attack patterns
3. **Block IPs**: Add attacking IPs to blacklist
4. **Increase Security**: Tighten rate limits temporarily
5. **Report**: Document the incident

### Account Recovery

1. Admin can reset user passwords
2. Admin can unlock accounts
3. API keys can be regenerated
4. Session invalidation available

## Future Enhancements

Potential additional security features:
- Two-factor authentication (2FA)
- OAuth2/SAML integration
- Email verification
- Password reset via email
- Geographic access restrictions
- Advanced bot detection
- Web Application Firewall (WAF) integration

## Contact

For security concerns or to report vulnerabilities, contact the development team. 