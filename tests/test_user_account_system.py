"""
Comprehensive test suite for the enhanced user account system.
Tests user authentication, profile management, activity tracking, and certificate history.
"""

import pytest
import json
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Import Flask app and models
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from web_interface.app import app
from web_interface.models import (
    db, User, UserActivity, UserSession, CertificateFavorite, 
    QueryLog, VerificationResult
)
from web_interface.auth import check_password_strength


class TestUserAccountSystem:
    """Test suite for enhanced user account functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client with in-memory database."""
        # Create temporary database
        db_fd, app.config['DATABASE'] = tempfile.mkstemp()
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        app.config['SECRET_KEY'] = 'test-secret-key'
        
        with app.test_client() as client:
            with app.app_context():
                db.create_all()
                yield client
                
        os.close(db_fd)
        os.unlink(app.config['DATABASE'])
    
    @pytest.fixture
    def sample_user(self):
        """Create a sample user for testing."""
        with app.app_context():
            user = User(
                username='testuser',
                email='test@example.com',
                first_name='Test',
                last_name='User',
                organization='Test Org',
                job_title='Developer',
                bio='Test bio',
                subscription_type='free',
                daily_request_limit=50,
                monthly_request_limit=1000
            )
            user.set_password('testpass123!')
            db.session.add(user)
            db.session.commit()
            return user
    
    @pytest.fixture
    def admin_user(self):
        """Create an admin user for testing."""
        with app.app_context():
            admin = User(
                username='admin',
                email='admin@example.com',
                role='admin',
                subscription_type='enterprise',
                daily_request_limit=1000,
                monthly_request_limit=10000
            )
            admin.set_password('admin123!')
            db.session.add(admin)
            db.session.commit()
            return admin

    def test_user_registration(self, client):
        """Test user registration functionality."""
        # Test successful registration
        response = client.post('/auth/register', data={
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password': 'StrongPass123!',
            'confirm_password': 'StrongPass123!'
        })
        assert response.status_code == 302  # Redirect after successful registration
        
        # Verify user was created
        with app.app_context():
            user = User.query.filter_by(username='newuser').first()
            assert user is not None
            assert user.email == 'newuser@example.com'
            assert user.subscription_type == 'free'
            assert user.daily_request_limit == 50
            assert user.check_password('StrongPass123!')
    
    def test_user_registration_validation(self, client):
        """Test user registration input validation."""
        # Test duplicate username
        with app.app_context():
            existing_user = User(username='existing', email='existing@example.com')
            existing_user.set_password('pass123')
            db.session.add(existing_user)
            db.session.commit()
        
        response = client.post('/auth/register', data={
            'username': 'existing',
            'email': 'new@example.com',
            'password': 'NewPass123!',
            'confirm_password': 'NewPass123!'
        })
        assert b'Username already exists' in response.data
        
        # Test weak password
        response = client.post('/auth/register', data={
            'username': 'weakpass',
            'email': 'weak@example.com',
            'password': '123',
            'confirm_password': '123'
        })
        assert b'Password must be at least 8 characters' in response.data
    
    def test_user_login_logout(self, client, sample_user):
        """Test user login and logout functionality."""
        # Test successful login
        response = client.post('/auth/login', data={
            'username': 'testuser',
            'password': 'testpass123!'
        })
        assert response.status_code == 302
        
        # Test access to protected page
        response = client.get('/auth/profile')
        assert response.status_code == 200
        assert b'testuser' in response.data
        
        # Test logout
        response = client.get('/auth/logout')
        assert response.status_code == 302
        
        # Test access to protected page after logout
        response = client.get('/auth/profile')
        assert response.status_code == 302  # Redirect to login
    
    def test_enhanced_user_profile(self, client, sample_user):
        """Test enhanced user profile functionality."""
        # Login first
        client.post('/auth/login', data={
            'username': 'testuser',
            'password': 'testpass123!'
        })
        
        # Test profile view
        response = client.get('/auth/profile')
        assert response.status_code == 200
        assert b'Test User' in response.data
        assert b'Test Org' in response.data
        assert b'Developer' in response.data
        
        # Test profile update
        response = client.post('/auth/update_profile', data={
            'first_name': 'Updated',
            'last_name': 'Name',
            'organization': 'New Org',
            'job_title': 'Senior Developer',
            'bio': 'Updated bio',
            'website': 'https://example.com',
            'location': 'New York',
            'profile_visibility': 'public'
        })
        assert response.status_code == 302
        
        # Verify updates
        with app.app_context():
            user = User.query.filter_by(username='testuser').first()
            assert user.first_name == 'Updated'
            assert user.last_name == 'Name'
            assert user.organization == 'New Org'
            assert user.job_title == 'Senior Developer'
            assert user.website == 'https://example.com'
            assert user.profile_visibility == 'public'
    
    def test_user_activity_tracking(self, client, sample_user):
        """Test user activity tracking functionality."""
        # Login to generate activity
        client.post('/auth/login', data={
            'username': 'testuser',
            'password': 'testpass123!'
        })
        
        with app.app_context():
            # Check that login activity was logged
            activities = UserActivity.query.filter_by(
                user_id=sample_user.id,
                activity_type='login'
            ).all()
            assert len(activities) > 0
            
            activity = activities[0]
            assert activity.success is True
            assert activity.ip_address is not None
            assert activity.activity_details is not None
    
    def test_api_key_management(self, client, sample_user):
        """Test API key generation and management."""
        # Login first
        client.post('/auth/login', data={
            'username': 'testuser',
            'password': 'testpass123!'
        })
        
        # Generate API key
        response = client.post('/auth/api_key', 
                             headers={'X-Requested-With': 'XMLHttpRequest'})
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'api_key' in data
        assert len(data['api_key']) == 64  # Standard API key length
        
        # Verify API key was stored
        with app.app_context():
            user = User.query.filter_by(username='testuser').first()
            assert user.api_key is not None
            assert user.api_key_created is not None
        
        # Test API key revocation
        response = client.post('/auth/revoke_api_key',
                             headers={'X-Requested-With': 'XMLHttpRequest'})
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['success'] is True
        
        # Verify API key was revoked
        with app.app_context():
            user = User.query.filter_by(username='testuser').first()
            assert user.api_key is None
    
    def test_rate_limiting(self, client, sample_user):
        """Test user rate limiting functionality."""
        with app.app_context():
            user = User.query.filter_by(username='testuser').first()
            
            # Test within limits
            assert user.check_rate_limit() is True
            
            # Simulate reaching daily limit
            user.daily_request_count = user.daily_request_limit
            db.session.commit()
            
            assert user.check_rate_limit() is False
            
            # Test request count increment
            original_count = user.daily_request_count
            user.increment_request_count()
            db.session.commit()
            
            assert user.daily_request_count == original_count + 1
            assert user.total_request_count == 1
    
    def test_certificate_generation_tracking(self, client, sample_user):
        """Test certificate generation tracking."""
        with app.app_context():
            # Create a sample certificate generation record
            query_log = QueryLog(
                user_id=sample_user.id,
                system_description='dx/dt = -x, dy/dt = -y',
                system_name='Simple Linear System',
                system_type='continuous',
                system_dimension=2,
                variables=['x', 'y'],
                model_name='test-model',
                model_config={'temperature': 0.7},
                generated_certificate='V(x,y) = x^2 + y^2',
                certificate_format='polynomial',
                certificate_complexity=5,
                status='completed',
                processing_time=2.5,
                user_rating=5,
                tags=['linear', 'stable']
            )
            db.session.add(query_log)
            db.session.commit()
            
            # Test certificate count increment
            original_count = sample_user.certificates_generated
            sample_user.increment_certificate_count()
            db.session.commit()
            
            assert sample_user.certificates_generated == original_count + 1
    
    def test_certificate_favorites(self, client, sample_user):
        """Test certificate favorites functionality."""
        with app.app_context():
            # Create a query log first
            query_log = QueryLog(
                user_id=sample_user.id,
                system_description='Test system',
                generated_certificate='V(x) = x^2',
                status='completed'
            )
            db.session.add(query_log)
            db.session.flush()  # Get the ID
            
            # Create a favorite
            favorite = CertificateFavorite(
                user_id=sample_user.id,
                query_id=query_log.id,
                name='My Favorite Certificate',
                notes='This is a great certificate',
                tags=['favorite', 'quadratic'],
                is_public=False
            )
            db.session.add(favorite)
            db.session.commit()
            
            # Test favorite retrieval
            favorites = CertificateFavorite.query.filter_by(
                user_id=sample_user.id
            ).all()
            assert len(favorites) == 1
            assert favorites[0].name == 'My Favorite Certificate'
            assert 'favorite' in favorites[0].tags
    
    def test_user_statistics(self, client, sample_user):
        """Test user statistics and usage tracking."""
        with app.app_context():
            # Set up some data
            sample_user.certificates_generated = 10
            sample_user.successful_verifications = 8
            sample_user.daily_request_count = 25
            sample_user.monthly_request_count = 300
            sample_user.total_request_count = 1500
            db.session.commit()
            
            # Test usage statistics
            stats = sample_user.get_usage_stats()
            assert stats['certificates_generated'] == 10
            assert stats['successful_verifications'] == 8
            assert stats['daily_requests'] == 25
            assert stats['monthly_requests'] == 300
            assert stats['total_requests'] == 1500
            assert stats['daily_usage_percent'] == 50.0  # 25/50 * 100
            assert stats['monthly_usage_percent'] == 30.0  # 300/1000 * 100
    
    def test_subscription_management(self, client, sample_user):
        """Test subscription status and management."""
        with app.app_context():
            # Test free subscription
            status = sample_user.get_subscription_status()
            assert status['active'] is False
            assert status['type'] == 'free'
            
            # Test active subscription
            sample_user.subscription_type = 'premium'
            sample_user.subscription_start = datetime.utcnow() - timedelta(days=10)
            sample_user.subscription_end = datetime.utcnow() + timedelta(days=20)
            db.session.commit()
            
            status = sample_user.get_subscription_status()
            assert status['active'] is True
            assert status['type'] == 'premium'
            assert status['days_remaining'] == 20
    
    def test_user_session_tracking(self, client, sample_user):
        """Test user session tracking and management."""
        with app.app_context():
            # Create a user session
            session = UserSession(
                user_id=sample_user.id,
                session_token='test-session-token',
                ip_address='127.0.0.1',
                user_agent='Test Browser',
                device_type='desktop',
                browser='Chrome',
                os='Windows',
                login_method='password'
            )
            db.session.add(session)
            db.session.commit()
            
            # Test session retrieval
            sessions = UserSession.query.filter_by(
                user_id=sample_user.id,
                is_active=True
            ).all()
            assert len(sessions) == 1
            assert sessions[0].device_type == 'desktop'
            assert sessions[0].browser == 'Chrome'
    
    def test_verification_tracking(self, client, sample_user):
        """Test verification result tracking."""
        with app.app_context():
            # Create query log and verification result
            query_log = QueryLog(
                user_id=sample_user.id,
                system_description='Test system',
                generated_certificate='V(x) = x^2',
                status='completed'
            )
            db.session.add(query_log)
            db.session.flush()
            
            verification = VerificationResult(
                query_id=query_log.id,
                numerical_check_passed=True,
                symbolic_check_passed=True,
                sos_check_passed=False,
                overall_success=True,
                verification_time_seconds=1.5,
                samples_used=1000
            )
            db.session.add(verification)
            db.session.commit()
            
            # Test verification count increment
            original_count = sample_user.successful_verifications
            sample_user.increment_verification_count()
            db.session.commit()
            
            assert sample_user.successful_verifications == original_count + 1
    
    def test_user_preferences(self, client, sample_user):
        """Test user preferences management."""
        # Login first
        client.post('/auth/login', data={
            'username': 'testuser',
            'password': 'testpass123!'
        })
        
        # Update preferences
        response = client.post('/auth/update_preferences', data={
            'theme_preference': 'dark',
            'default_rag_k': '5',
            'email_notifications': 'on',
            'marketing_emails': ''  # Unchecked checkbox
        })
        assert response.status_code == 302
        
        # Verify preferences were updated
        with app.app_context():
            user = User.query.filter_by(username='testuser').first()
            assert user.theme_preference == 'dark'
            assert user.default_rag_k == 5
            assert user.email_notifications is True
            assert user.marketing_emails is False
    
    def test_admin_functionality(self, client, admin_user):
        """Test admin-specific functionality."""
        # Login as admin
        client.post('/auth/login', data={
            'username': 'admin',
            'password': 'admin123!'
        })
        
        # Test admin profile access
        response = client.get('/auth/profile')
        assert response.status_code == 200
        assert b'Administrator Panel' in response.data
        
        # Verify admin properties
        with app.app_context():
            user = User.query.filter_by(username='admin').first()
            assert user.is_admin is True
            assert user.role == 'admin'
            assert user.subscription_type == 'enterprise'
    
    def test_user_data_export(self, client, sample_user):
        """Test user data export functionality."""
        # Login first
        client.post('/auth/login', data={
            'username': 'testuser',
            'password': 'testpass123!'
        })
        
        # Create some data to export
        with app.app_context():
            query_log = QueryLog(
                user_id=sample_user.id,
                system_description='Test export system',
                generated_certificate='V(x) = x^2',
                status='completed'
            )
            db.session.add(query_log)
            
            activity = UserActivity(
                user_id=sample_user.id,
                activity_type='test_activity',
                activity_details={'test': 'data'}
            )
            db.session.add(activity)
            db.session.commit()
        
        # Test data export
        response = client.get('/auth/export_account')
        assert response.status_code == 200
        assert response.headers['Content-Type'] == 'application/json'
        
        # Verify export contains user data
        data = json.loads(response.data)
        assert 'username' in data
        assert 'queries' in data
        assert 'activities' in data
        assert len(data['queries']) > 0
        assert len(data['activities']) > 0
    
    def test_password_strength_validation(self):
        """Test password strength validation."""
        # Test weak passwords
        is_valid, message = check_password_strength('123')
        assert is_valid is False
        assert 'at least 8 characters' in message
        
        is_valid, message = check_password_strength('password')
        assert is_valid is False
        assert 'uppercase letter' in message
        
        is_valid, message = check_password_strength('Password')
        assert is_valid is False
        assert 'number' in message
        
        # Test strong password
        is_valid, message = check_password_strength('StrongPass123!')
        assert is_valid is True
    
    def test_user_display_properties(self, sample_user):
        """Test user display properties and methods."""
        with app.app_context():
            # Test full name
            assert sample_user.full_name == 'Test User'
            assert sample_user.display_name == 'Test User'
            
            # Test user with no first/last name
            user_no_name = User(username='noname', email='noname@test.com')
            assert user_no_name.full_name == 'noname'
            assert user_no_name.display_name == 'noname'
            
            # Test to_dict method
            user_dict = sample_user.to_dict(include_sensitive=True)
            assert 'username' in user_dict
            assert 'display_name' in user_dict
            assert 'usage_stats' in user_dict
            assert 'subscription_status' in user_dict
            
            user_dict_public = sample_user.to_dict(include_sensitive=False)
            assert 'username' in user_dict_public
            assert 'usage_stats' not in user_dict_public


class TestUserAccountIntegration:
    """Integration tests for user account system with certificate generation."""
    
    @pytest.fixture
    def client(self):
        """Create test client with in-memory database."""
        db_fd, app.config['DATABASE'] = tempfile.mkstemp()
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        app.config['SECRET_KEY'] = 'test-secret-key'
        
        with app.test_client() as client:
            with app.app_context():
                db.create_all()
                yield client
                
        os.close(db_fd)
        os.unlink(app.config['DATABASE'])
    
    def test_certificate_generation_with_user_tracking(self, client):
        """Test certificate generation with full user tracking."""
        # Create and login user
        with app.app_context():
            user = User(username='certuser', email='cert@test.com')
            user.set_password('pass123')
            db.session.add(user)
            db.session.commit()
        
        client.post('/auth/login', data={
            'username': 'certuser',
            'password': 'pass123'
        })
        
        # Mock certificate generation
        with patch('web_interface.certificate_generator.CertificateGenerator') as mock_gen:
            mock_gen.return_value.generate_certificate_with_user_tracking.return_value = {
                'generated_certificate': 'V(x,y) = x^2 + y^2',
                'status': 'completed',
                'processing_time_seconds': 2.5,
                'certificate_format': 'polynomial',
                'certificate_complexity': 5
            }
            
            response = client.post('/generate', data={
                'system_description': 'dx/dt = -x, dy/dt = -y',
                'system_name': 'Test System',
                'model_config': 'qwen2.5-coder-1.5b'
            })
            
            assert response.status_code == 200
            
            # Verify tracking data was created
            with app.app_context():
                user = User.query.filter_by(username='certuser').first()
                queries = QueryLog.query.filter_by(user_id=user.id).all()
                activities = UserActivity.query.filter_by(
                    user_id=user.id,
                    activity_type='certificate_generated'
                ).all()
                
                assert len(queries) > 0
                assert len(activities) > 0
                assert user.certificates_generated > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 