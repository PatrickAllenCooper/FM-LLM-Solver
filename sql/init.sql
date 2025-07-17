-- FM-LLM Solver Database Initialization Script
-- Enhanced User Account System with Comprehensive Tracking

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- Create enhanced users table
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(80) UNIQUE NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    
    -- Enhanced user profile
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    organization VARCHAR(200),
    job_title VARCHAR(100),
    bio TEXT,
    website VARCHAR(255),
    location VARCHAR(100),
    timezone VARCHAR(50) DEFAULT 'UTC',
    
    -- User status and verification
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    is_premium BOOLEAN DEFAULT FALSE,
    email_verified BOOLEAN DEFAULT FALSE,
    email_verification_token VARCHAR(255),
    password_reset_token VARCHAR(255),
    password_reset_expires DATETIME,
    
    -- Account timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_login DATETIME,
    last_active DATETIME DEFAULT CURRENT_TIMESTAMP,
    profile_updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Subscription and billing
    subscription_type VARCHAR(20) DEFAULT 'free',
    subscription_start DATETIME,
    subscription_end DATETIME,
    billing_email VARCHAR(120),
    
    -- Rate limiting and usage
    daily_request_count INTEGER DEFAULT 0,
    monthly_request_count INTEGER DEFAULT 0,
    total_request_count INTEGER DEFAULT 0,
    last_request_date DATE,
    last_request_month VARCHAR(7),
    
    -- Usage limits based on subscription
    daily_request_limit INTEGER DEFAULT 50,
    monthly_request_limit INTEGER DEFAULT 1000,
    max_concurrent_requests INTEGER DEFAULT 3,
    
    -- API access
    api_key VARCHAR(64) UNIQUE,
    api_key_created DATETIME,
    api_key_last_used DATETIME,
    api_requests_count INTEGER DEFAULT 0,
    
    -- User preferences
    preferred_models JSON,
    default_rag_k INTEGER DEFAULT 3,
    email_notifications BOOLEAN DEFAULT TRUE,
    marketing_emails BOOLEAN DEFAULT FALSE,
    theme_preference VARCHAR(20) DEFAULT 'light',
    
    -- User role and permissions
    role VARCHAR(20) DEFAULT 'user',
    permissions JSON,
    
    -- Privacy and security
    profile_visibility VARCHAR(20) DEFAULT 'private',
    two_factor_enabled BOOLEAN DEFAULT FALSE,
    two_factor_secret VARCHAR(32),
    backup_codes JSON,
    
    -- Activity tracking
    login_count INTEGER DEFAULT 0,
    certificates_generated INTEGER DEFAULT 0,
    successful_verifications INTEGER DEFAULT 0,
    favorite_systems JSON
);

-- Create user activities table
CREATE TABLE IF NOT EXISTS user_activities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Activity details
    activity_type VARCHAR(50) NOT NULL,
    activity_details JSON,
    
    -- Request context
    ip_address VARCHAR(45),
    user_agent TEXT,
    session_id VARCHAR(255),
    
    -- Performance tracking
    response_time_ms INTEGER,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create user sessions table
CREATE TABLE IF NOT EXISTS user_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    
    -- Session details
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME,
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Device/browser info
    ip_address VARCHAR(45),
    user_agent TEXT,
    device_type VARCHAR(50),
    browser VARCHAR(100),
    os VARCHAR(100),
    
    -- Security
    login_method VARCHAR(20),
    is_remembered BOOLEAN DEFAULT FALSE,
    
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create certificate favorites table
CREATE TABLE IF NOT EXISTS certificate_favorites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    query_id INTEGER NOT NULL,
    
    -- Favorite details
    name VARCHAR(200),
    notes TEXT,
    tags JSON,
    
    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Visibility
    is_public BOOLEAN DEFAULT FALSE,
    
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (query_id) REFERENCES query_logs(id) ON DELETE CASCADE,
    UNIQUE(user_id, query_id)
);

-- Enhanced query logs table
CREATE TABLE IF NOT EXISTS query_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    
    -- System and query details
    system_description TEXT NOT NULL,
    system_name VARCHAR(200),
    system_type VARCHAR(50),
    system_dimension INTEGER,
    variables JSON,
    
    -- Model configuration and generation
    model_config JSON,
    model_name VARCHAR(200),
    model_version VARCHAR(50),
    rag_k INTEGER DEFAULT 0,
    temperature REAL DEFAULT 0.7,
    max_tokens INTEGER DEFAULT 512,
    
    -- Results
    generated_certificate TEXT,
    certificate_format VARCHAR(50),
    certificate_complexity INTEGER,
    extraction_method VARCHAR(50),
    
    -- Status and performance
    status VARCHAR(50) DEFAULT 'pending',
    error_message TEXT,
    processing_time REAL,
    processing_start DATETIME,
    processing_end DATETIME,
    total_tokens_used INTEGER,
    cost_estimate REAL,
    
    -- Context and tracking
    conversation_id VARCHAR(36),
    session_id VARCHAR(255),
    ip_address VARCHAR(45),
    user_agent TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- User interaction
    user_rating INTEGER,
    user_feedback TEXT,
    is_favorite BOOLEAN DEFAULT FALSE,
    is_public BOOLEAN DEFAULT FALSE,
    tags JSON,
    
    -- Domain bounds for certificate validity
    certificate_domain_bounds TEXT,
    domain_bounds_conditions TEXT,
    domain_description TEXT,
    
    -- Verification tracking
    verification_requested BOOLEAN DEFAULT FALSE,
    verification_completed BOOLEAN DEFAULT FALSE,
    verification_success BOOLEAN DEFAULT FALSE,
    verification_attempts INTEGER DEFAULT 0,
    
    -- Quality metrics
    confidence_score REAL,
    mathematical_soundness REAL,
    
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- Create rate limit logs table
CREATE TABLE IF NOT EXISTS rate_limit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Request details
    endpoint VARCHAR(200),
    method VARCHAR(10),
    ip_address VARCHAR(45),
    
    -- Rate limit status
    was_blocked BOOLEAN DEFAULT FALSE,
    requests_today INTEGER,
    limit_exceeded_by INTEGER DEFAULT 0,
    
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create IP blacklist table
CREATE TABLE IF NOT EXISTS ip_blacklist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ip_address VARCHAR(45) UNIQUE NOT NULL,
    reason VARCHAR(200),
    blocked_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    blocked_until DATETIME,
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Tracking
    request_count INTEGER DEFAULT 0,
    last_request DATETIME
);

-- Create security logs table
CREATE TABLE IF NOT EXISTS security_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    event_type VARCHAR(50),
    
    -- User info (if applicable)
    user_id INTEGER,
    username VARCHAR(80),
    
    -- Request info
    ip_address VARCHAR(45),
    user_agent TEXT,
    endpoint VARCHAR(200),
    
    -- Event details
    description TEXT,
    severity VARCHAR(20),
    
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- Create conversations table
CREATE TABLE IF NOT EXISTS conversation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id VARCHAR(100) UNIQUE NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- User tracking
    user_id INTEGER,
    
    -- Conversation settings
    model_config VARCHAR(100) NOT NULL,
    rag_k INTEGER DEFAULT 3,
    
    -- Current state
    status VARCHAR(20) DEFAULT 'active',
    system_description TEXT,
    ready_to_generate BOOLEAN DEFAULT FALSE,
    
    -- Domain bounds for certificate validity
    domain_bounds TEXT,
    domain_conditions TEXT,
    
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- Create conversation messages table
CREATE TABLE IF NOT EXISTS conversation_message (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Message content
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    
    -- Metadata
    message_type VARCHAR(30) DEFAULT 'chat',
    processing_time_seconds REAL,
    context_chunks_used INTEGER DEFAULT 0,
    
    FOREIGN KEY (conversation_id) REFERENCES conversation(id) ON DELETE CASCADE
);

-- Create verification results table
CREATE TABLE IF NOT EXISTS verification_result (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id INTEGER NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Individual verification checks
    numerical_check_passed BOOLEAN DEFAULT FALSE,
    symbolic_check_passed BOOLEAN DEFAULT FALSE,
    sos_check_passed BOOLEAN DEFAULT FALSE,
    
    -- Domain bounds verification
    domain_bounds_check_passed BOOLEAN DEFAULT TRUE,
    domain_bounds_violations INTEGER DEFAULT 0,
    
    -- Overall verification result
    overall_success BOOLEAN DEFAULT FALSE,
    
    -- Detailed verification information
    verification_details TEXT,
    
    -- Verification metadata
    verification_time_seconds REAL,
    samples_used INTEGER,
    tolerance_used REAL,
    
    FOREIGN KEY (query_id) REFERENCES query_logs(id) ON DELETE CASCADE
);

-- Create model configurations table
CREATE TABLE IF NOT EXISTS model_configuration (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    
    -- Model settings
    base_model_name VARCHAR(200) NOT NULL,
    adapter_path VARCHAR(500),
    barrier_certificate_type VARCHAR(50),
    
    -- Configuration JSON
    config_json TEXT,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    is_available BOOLEAN DEFAULT TRUE,
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create system benchmarks table
CREATE TABLE IF NOT EXISTS system_benchmark (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    
    -- System definition
    system_dynamics TEXT NOT NULL,
    initial_set TEXT,
    unsafe_set TEXT,
    safe_set TEXT,
    state_variables VARCHAR(200),
    
    -- Domain bounds for barrier certificate validity
    certificate_domain_bounds TEXT,
    domain_bounds_description TEXT,
    
    -- Expected results (if known)
    expected_certificate TEXT,
    expected_verification BOOLEAN,
    
    -- Metadata
    difficulty_level VARCHAR(20),
    system_type VARCHAR(50),
    dimension INTEGER,
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_api_key ON users(api_key);
CREATE INDEX IF NOT EXISTS idx_users_subscription_type ON users(subscription_type);
CREATE INDEX IF NOT EXISTS idx_users_last_active ON users(last_active);

CREATE INDEX IF NOT EXISTS idx_user_activities_user_id ON user_activities(user_id);
CREATE INDEX IF NOT EXISTS idx_user_activities_timestamp ON user_activities(timestamp);
CREATE INDEX IF NOT EXISTS idx_user_activities_type ON user_activities(activity_type);

CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_user_sessions_active ON user_sessions(is_active);

CREATE INDEX IF NOT EXISTS idx_query_logs_user_id ON query_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_query_logs_timestamp ON query_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_query_logs_status ON query_logs(status);
CREATE INDEX IF NOT EXISTS idx_query_logs_model_name ON query_logs(model_name);
CREATE INDEX IF NOT EXISTS idx_query_logs_conversation_id ON query_logs(conversation_id);

CREATE INDEX IF NOT EXISTS idx_certificate_favorites_user_id ON certificate_favorites(user_id);
CREATE INDEX IF NOT EXISTS idx_certificate_favorites_query_id ON certificate_favorites(query_id);

CREATE INDEX IF NOT EXISTS idx_security_logs_timestamp ON security_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_security_logs_event_type ON security_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_security_logs_user_id ON security_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_security_logs_ip_address ON security_logs(ip_address);

CREATE INDEX IF NOT EXISTS idx_rate_limit_logs_user_id ON rate_limit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_rate_limit_logs_timestamp ON rate_limit_logs(timestamp);

CREATE INDEX IF NOT EXISTS idx_ip_blacklist_ip_address ON ip_blacklist(ip_address);
CREATE INDEX IF NOT EXISTS idx_ip_blacklist_active ON ip_blacklist(is_active);

CREATE INDEX IF NOT EXISTS idx_conversation_user_id ON conversation(user_id);
CREATE INDEX IF NOT EXISTS idx_conversation_session_id ON conversation(session_id);
CREATE INDEX IF NOT EXISTS idx_conversation_status ON conversation(status);

CREATE INDEX IF NOT EXISTS idx_conversation_message_conversation_id ON conversation_message(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversation_message_timestamp ON conversation_message(timestamp);

-- Insert default admin user (password: admin123!)
INSERT OR IGNORE INTO users (
    username, email, password_hash, role, is_active, is_verified,
    daily_request_limit, monthly_request_limit, certificates_generated,
    created_at, subscription_type
) VALUES (
    'admin',
    'admin@fm-llm-solver.com', 
    'pbkdf2:sha256:260000$TMK8bGtLzHVT8cxY$a8b7f5c3d2e1f6789456123abc987def654321fedcba9876543210abcdef012',
    'admin',
    1,
    1,
    1000,
    10000,
    0,
    CURRENT_TIMESTAMP,
    'enterprise'
);

-- Insert sample model configurations
INSERT OR IGNORE INTO model_configuration (
    name, description, base_model_name, barrier_certificate_type,
    config_json, is_active, is_available
) VALUES 
(
    'qwen2.5-coder-1.5b',
    'Qwen2.5-Coder 1.5B - Optimized for mathematical reasoning',
    'Qwen/Qwen2.5-Coder-1.5B-Instruct',
    'unified',
    '{"max_tokens": 512, "temperature": 0.7, "top_p": 0.9, "quantization": "4bit"}',
    1, 1
),
(
    'qwen2.5-coder-7b',
    'Qwen2.5-Coder 7B - High-performance mathematical reasoning',
    'Qwen/Qwen2.5-Coder-7B-Instruct',
    'unified',
    '{"max_tokens": 1024, "temperature": 0.6, "top_p": 0.95, "quantization": "4bit"}',
    1, 1
),
(
    'deepseek-coder-6.7b',
    'DeepSeek-Coder 6.7B - Advanced code generation and reasoning',
    'deepseek-ai/deepseek-coder-6.7b-instruct',
    'continuous',
    '{"max_tokens": 512, "temperature": 0.7, "quantization": "4bit"}',
    1, 1
);

-- Insert sample system benchmarks
INSERT OR IGNORE INTO system_benchmark (
    name, description, system_dynamics, initial_set, unsafe_set,
    certificate_domain_bounds, difficulty_level, system_type, dimension
) VALUES 
(
    'Linear Unstable System',
    'Simple 2D linear system with unstable equilibrium',
    'dx/dt = x + y, dy/dt = -x + y',
    'x^2 + y^2 <= 1',
    'x^2 + y^2 >= 4',
    '{"x": [-3, 3], "y": [-3, 3]}',
    'easy',
    'continuous',
    2
),
(
    'Van der Pol Oscillator',
    'Nonlinear oscillator with limit cycle behavior',
    'dx/dt = y, dy/dt = μ(1 - x²)y - x',
    'x^2 + y^2 <= 0.1',
    'x^2 + y^2 >= 4',
    '{"x": [-3, 3], "y": [-3, 3]}',
    'medium',
    'continuous',
    2
),
(
    'Discrete Linear System',
    'Simple discrete-time linear system',
    'x[k+1] = 0.9*x[k] + 0.1*y[k], y[k+1] = -0.1*x[k] + 0.9*y[k]',
    'x^2 + y^2 <= 1',
    'x^2 + y^2 >= 4',
    '{"x": [-3, 3], "y": [-3, 3]}',
    'easy',
    'discrete',
    2
);

-- Create views for common queries
CREATE VIEW IF NOT EXISTS user_activity_summary AS
SELECT 
    u.id as user_id,
    u.username,
    u.subscription_type,
    u.certificates_generated,
    u.successful_verifications,
    COUNT(ua.id) as total_activities,
    MAX(ua.timestamp) as last_activity_time,
    COUNT(CASE WHEN ua.activity_type = 'login' THEN 1 END) as login_count_activities,
    COUNT(CASE WHEN ua.activity_type = 'certificate_generated' THEN 1 END) as certificate_generation_activities
FROM users u
LEFT JOIN user_activities ua ON u.id = ua.user_id
GROUP BY u.id, u.username, u.subscription_type, u.certificates_generated, u.successful_verifications;

CREATE VIEW IF NOT EXISTS daily_user_stats AS
SELECT 
    DATE(timestamp) as date,
    COUNT(DISTINCT user_id) as active_users,
    COUNT(CASE WHEN activity_type = 'certificate_generated' THEN 1 END) as certificates_generated,
    COUNT(CASE WHEN activity_type = 'login' THEN 1 END) as logins,
    AVG(response_time_ms) as avg_response_time_ms
FROM user_activities
WHERE timestamp >= DATE('now', '-30 days')
GROUP BY DATE(timestamp)
ORDER BY date DESC;

-- Create triggers for automatic timestamping
CREATE TRIGGER IF NOT EXISTS update_user_last_active
AFTER INSERT ON user_activities
FOR EACH ROW
BEGIN
    UPDATE users SET last_active = CURRENT_TIMESTAMP WHERE id = NEW.user_id;
END;

CREATE TRIGGER IF NOT EXISTS update_conversation_timestamp
AFTER INSERT ON conversation_message
FOR EACH ROW
BEGIN
    UPDATE conversation SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.conversation_id;
END;

-- Create trigger to automatically update certificate count
CREATE TRIGGER IF NOT EXISTS increment_certificate_count
AFTER UPDATE OF status ON query_logs
FOR EACH ROW
WHEN NEW.status = 'completed' AND OLD.status != 'completed' AND NEW.user_id IS NOT NULL
BEGIN
    UPDATE users 
    SET certificates_generated = certificates_generated + 1 
    WHERE id = NEW.user_id;
END;

-- Create trigger to automatically update verification count
CREATE TRIGGER IF NOT EXISTS increment_verification_count
AFTER INSERT ON verification_result
FOR EACH ROW
WHEN NEW.overall_success = 1
BEGIN
    UPDATE users 
    SET successful_verifications = successful_verifications + 1 
    WHERE id = (SELECT user_id FROM query_logs WHERE id = NEW.query_id);
END;

-- Optimize database settings for performance
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = 10000;
PRAGMA temp_store = memory;
PRAGMA mmap_size = 268435456; -- 256MB 