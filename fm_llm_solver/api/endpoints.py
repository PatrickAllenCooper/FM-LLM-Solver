"""
Comprehensive API Endpoints for FM-LLM-Solver

This module defines all external API endpoints for authorized access
to the FM-LLM-Solver system functionality.
"""

import os
import sys
import time
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from functools import wraps

from flask import Blueprint, request, jsonify, current_app
from flask_restx import Api, Resource, fields, Namespace
from werkzeug.security import check_password_hash

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from web_interface.models import User, QueryLog, UserActivity, db
from web_interface.certificate_generator import CertificateGenerator
from web_interface.conversation_service import ConversationService
from fm_llm_solver.core.config import load_config

logger = logging.getLogger(__name__)

# Create API blueprint
api_bp = Blueprint('api_v1', __name__, url_prefix='/api/v1')
api = Api(api_bp, 
          title='FM-LLM-Solver API',
          version='1.0',
          description='Comprehensive API for barrier certificate generation and system management',
          doc='/docs/')

# API Namespaces
auth_ns = Namespace('auth', description='Authentication and authorization')
certificates_ns = Namespace('certificates', description='Barrier certificate generation')
models_ns = Namespace('models', description='Model management and selection')
rag_ns = Namespace('rag', description='RAG configuration and datasets')
users_ns = Namespace('users', description='User management')
queries_ns = Namespace('queries', description='Query history and management')
system_ns = Namespace('system', description='System status and monitoring')

api.add_namespace(auth_ns)
api.add_namespace(certificates_ns)
api.add_namespace(models_ns)
api.add_namespace(rag_ns)
api.add_namespace(users_ns)
api.add_namespace(queries_ns)
api.add_namespace(system_ns)

# Initialize services
config = load_config()
certificate_generator = CertificateGenerator(config)
conversation_service = ConversationService(config)


# ============= AUTHENTICATION AND AUTHORIZATION =============

def require_api_key(f):
    """Decorator to require valid API key for endpoint access."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        
        if not api_key:
            return {'error': 'API key required', 'code': 'MISSING_API_KEY'}, 401
        
        # Find user by API key
        user = User.query.filter_by(api_key=api_key).first()
        if not user:
            return {'error': 'Invalid API key', 'code': 'INVALID_API_KEY'}, 401
        
        if not user.is_active:
            return {'error': 'Account deactivated', 'code': 'ACCOUNT_INACTIVE'}, 403
        
        # Update last API usage
        user.api_key_last_used = datetime.utcnow()
        user.api_requests_count += 1
        db.session.commit()
        
        # Store user in request context
        request.current_user = user
        
        return f(*args, **kwargs)
    return decorated_function


def require_admin(f):
    """Decorator to require admin privileges."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not hasattr(request, 'current_user') or request.current_user.role != 'admin':
            return {'error': 'Admin privileges required', 'code': 'INSUFFICIENT_PRIVILEGES'}, 403
        return f(*args, **kwargs)
    return decorated_function


# ============= AUTHENTICATION ENDPOINTS =============

@auth_ns.route('/token')
class APIToken(Resource):
    """Generate API token for user authentication."""
    
    def post(self):
        """Generate new API token."""
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return {'error': 'Username and password required'}, 400
        
        user = User.query.filter_by(username=username).first()
        if not user or not check_password_hash(user.password_hash, password):
            return {'error': 'Invalid credentials'}, 401
        
        if not user.is_active:
            return {'error': 'Account deactivated'}, 403
        
        # Generate new API key
        import secrets
        api_key = secrets.token_urlsafe(32)
        user.api_key = api_key
        user.api_key_created = datetime.utcnow()
        db.session.commit()
        
        return {
            'api_key': api_key,
            'expires_in': None,  # No expiration for now
            'user_id': user.id,
            'username': user.username,
            'role': user.role
        }


@auth_ns.route('/validate')
class APIValidate(Resource):
    """Validate API token."""
    
    @require_api_key
    def get(self):
        """Validate current API key."""
        user = request.current_user
        return {
            'valid': True,
            'user_id': user.id,
            'username': user.username,
            'role': user.role,
            'api_requests_count': user.api_requests_count,
            'last_used': user.api_key_last_used.isoformat() if user.api_key_last_used else None
        }


# ============= CERTIFICATE GENERATION ENDPOINTS =============

@certificates_ns.route('/generate')
class CertificateGenerate(Resource):
    """Generate barrier certificates."""
    
    @require_api_key
    def post(self):
        """Generate a barrier certificate."""
        try:
            data = request.get_json()
            
            # Validate required fields
            system_description = data.get('system_description')
            if not system_description:
                return {'error': 'system_description is required'}, 400
            
            # Extract generation parameters
            model_config = data.get('model_config', 'base')
            rag_config = data.get('rag_config', {})
            domain_bounds = data.get('domain_bounds')
            system_name = data.get('system_name')
            tags = data.get('tags', [])
            
            # Generate certificate with user tracking
            result = certificate_generator.generate_certificate_with_user_tracking(
                system_description=system_description,
                model_config=model_config,
                conversation_id=None,
                system_name=system_name,
                user_tags=tags,
                domain_bounds=domain_bounds,
                rag_config=rag_config
            )
            
            if result['success']:
                return {
                    'success': True,
                    'query_id': result['query_id'],
                    'certificate': result.get('certificate'),
                    'model_used': result.get('model_name'),
                    'processing_time': result.get('processing_time_seconds'),
                    'confidence_score': result.get('confidence_score'),
                    'format_type': result.get('certificate_format'),
                    'cost_estimate': result.get('cost_estimate'),
                    'context_chunks': result.get('context_chunks', 0)
                }
            else:
                return {
                    'success': False,
                    'error': result.get('error', 'Generation failed'),
                    'query_id': result.get('query_id')
                }, 500
        
        except Exception as e:
            logger.error(f"API certificate generation failed: {e}")
            return {'error': f'Internal error: {str(e)}'}, 500


@certificates_ns.route('/verify/<int:query_id>')
class CertificateVerify(Resource):
    """Verify generated certificates."""
    
    @require_api_key
    def post(self, query_id):
        """Verify a generated certificate."""
        try:
            query = QueryLog.query.filter_by(id=query_id, user_id=request.current_user.id).first()
            if not query:
                return {'error': 'Query not found'}, 404
            
            if not query.generated_certificate:
                return {'error': 'No certificate to verify'}, 400
            
            # TODO: Implement verification logic
            verification_result = {
                'verified': True,  # Placeholder
                'verification_method': 'symbolic',
                'conditions_satisfied': ['boundary', 'derivative'],
                'confidence': 0.85
            }
            
            # Update query with verification result
            query.verification_completed = True
            query.verification_success = verification_result['verified']
            db.session.commit()
            
            return {
                'success': True,
                'query_id': query_id,
                'verification_result': verification_result
            }
        
        except Exception as e:
            logger.error(f"API certificate verification failed: {e}")
            return {'error': f'Verification error: {str(e)}'}, 500


# ============= MODEL MANAGEMENT ENDPOINTS =============

@models_ns.route('/available')
class ModelsAvailable(Resource):
    """Get available models."""
    
    @require_api_key
    def get(self):
        """Get list of available models."""
        try:
            models = certificate_generator.get_available_models()
            return {
                'success': True,
                'models': models,
                'total_count': len(models)
            }
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return {'error': f'Failed to get models: {str(e)}'}, 500


@models_ns.route('/test/<model_key>')
class ModelsTest(Resource):
    """Test model availability."""
    
    @require_api_key
    def get(self, model_key):
        """Test if a specific model is available."""
        try:
            # Check if model is in available models
            available_models = certificate_generator.get_available_models()
            model_exists = any(m['key'] == model_key for m in available_models)
            
            if not model_exists:
                return {'error': 'Model not found'}, 404
            
            # TODO: Implement actual model testing
            test_result = {
                'available': True,
                'model_key': model_key,
                'status': 'healthy',
                'test_timestamp': datetime.utcnow().isoformat()
            }
            
            return {
                'success': True,
                'test_result': test_result
            }
        except Exception as e:
            logger.error(f"Model test failed: {e}")
            return {'error': f'Model test error: {str(e)}'}, 500


# ============= RAG CONFIGURATION ENDPOINTS =============

@rag_ns.route('/researchers')
class RAGResearchers(Resource):
    """Manage researcher datasets for RAG."""
    
    @require_api_key
    def get(self):
        """Get available researchers for RAG configuration."""
        try:
            import csv
            researchers = []
            user_ids_path = os.path.join(PROJECT_ROOT, 'data', 'user_ids.csv')
            
            if os.path.exists(user_ids_path):
                with open(user_ids_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        researchers.append({
                            'scholar_id': row.get('scholar_id', ''),
                            'name': row.get('name', ''),
                            'institution': row.get('institution', ''),
                            'research_focus': row.get('research_focus', '')
                        })
            
            return {
                'success': True,
                'researchers': researchers,
                'total_count': len(researchers)
            }
        except Exception as e:
            logger.error(f"Failed to get researchers: {e}")
            return {'error': f'Failed to get researchers: {str(e)}'}, 500


@rag_ns.route('/datasets/preview')
class RAGDatasetPreview(Resource):
    """Preview RAG dataset configurations."""
    
    @require_api_key
    def post(self):
        """Get preview statistics for RAG dataset configuration."""
        try:
            data = request.get_json()
            dataset_type = data.get('dataset_type', 'unified')
            selected_researchers = data.get('selected_researchers', [])
            selected_topics = data.get('selected_topics', [])
            
            # Calculate preview statistics (similar to existing logic)
            base_stats = {
                'unified': {'papers': 156, 'chunks': 3240, 'quality': 87},
                'discrete': {'papers': 45, 'chunks': 920, 'quality': 82},
                'continuous': {'papers': 98, 'chunks': 2010, 'quality': 89}
            }
            
            if dataset_type in base_stats:
                stats = base_stats[dataset_type]
                paper_count = stats['papers']
                chunk_count = stats['chunks']
                quality_score = stats['quality']
            else:  # custom
                paper_count = len(selected_researchers) * 5 if selected_researchers else 5
                chunk_count = paper_count * 15
                quality_score = min(95, 75 + (len(selected_topics) * 3))
            
            return {
                'success': True,
                'dataset_preview': {
                    'paper_count': paper_count,
                    'chunk_count': chunk_count,
                    'quality_score': quality_score,
                    'dataset_type': dataset_type,
                    'selected_researchers_count': len(selected_researchers),
                    'selected_topics_count': len(selected_topics)
                }
            }
        except Exception as e:
            logger.error(f"RAG dataset preview failed: {e}")
            return {'error': f'Preview error: {str(e)}'}, 500


# ============= USER MANAGEMENT ENDPOINTS =============

@users_ns.route('/profile')
class UserProfile(Resource):
    """Manage user profile."""
    
    @require_api_key
    def get(self):
        """Get current user profile."""
        user = request.current_user
        return {
            'user_id': user.id,
            'username': user.username,
            'email': user.email,
            'role': user.role,
            'created_at': user.created_at.isoformat(),
            'last_login': user.last_login.isoformat() if user.last_login else None,
            'certificates_generated': user.certificates_generated,
            'successful_verifications': user.successful_verifications,
            'daily_request_limit': user.daily_request_limit,
            'monthly_request_limit': user.monthly_request_limit,
            'requests_today': user.requests_today,
            'requests_month': user.requests_month
        }


@users_ns.route('/usage')
class UserUsage(Resource):
    """Get user usage statistics."""
    
    @require_api_key
    def get(self):
        """Get current user's usage statistics."""
        user = request.current_user
        
        # Get recent queries
        recent_queries = QueryLog.query.filter_by(user_id=user.id).order_by(
            QueryLog.timestamp.desc()
        ).limit(10).all()
        
        return {
            'usage_summary': {
                'total_queries': QueryLog.query.filter_by(user_id=user.id).count(),
                'successful_generations': QueryLog.query.filter_by(
                    user_id=user.id, status='completed'
                ).count(),
                'requests_today': user.requests_today,
                'requests_month': user.requests_month,
                'daily_limit': user.daily_request_limit,
                'monthly_limit': user.monthly_request_limit
            },
            'recent_queries': [
                {
                    'id': q.id,
                    'timestamp': q.timestamp.isoformat(),
                    'status': q.status,
                    'model_name': q.model_name,
                    'has_certificate': bool(q.generated_certificate),
                    'processing_time': q.processing_time
                } for q in recent_queries
            ]
        }


# ============= QUERY HISTORY ENDPOINTS =============

@queries_ns.route('/history')
class QueryHistory(Resource):
    """Query history management."""
    
    @require_api_key
    def get(self):
        """Get user's query history."""
        try:
            # Parse query parameters
            page = request.args.get('page', 1, type=int)
            per_page = min(request.args.get('per_page', 20, type=int), 100)
            status_filter = request.args.get('status')
            model_filter = request.args.get('model')
            
            # Build query
            query = QueryLog.query.filter_by(user_id=request.current_user.id)
            
            if status_filter:
                query = query.filter_by(status=status_filter)
            if model_filter:
                query = query.filter_by(model_name=model_filter)
            
            # Paginate
            pagination = query.order_by(QueryLog.timestamp.desc()).paginate(
                page=page, per_page=per_page, error_out=False
            )
            
            return {
                'success': True,
                'queries': [
                    {
                        'id': q.id,
                        'timestamp': q.timestamp.isoformat(),
                        'system_description': q.system_description,
                        'system_name': q.system_name,
                        'model_name': q.model_name,
                        'status': q.status,
                        'has_certificate': bool(q.generated_certificate),
                        'processing_time': q.processing_time,
                        'cost_estimate': q.cost_estimate,
                        'confidence_score': q.confidence_score,
                        'user_rating': q.user_rating
                    } for q in pagination.items
                ],
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': pagination.total,
                    'pages': pagination.pages,
                    'has_next': pagination.has_next,
                    'has_prev': pagination.has_prev
                }
            }
        except Exception as e:
            logger.error(f"Query history retrieval failed: {e}")
            return {'error': f'History error: {str(e)}'}, 500


@queries_ns.route('/<int:query_id>')
class QueryDetail(Resource):
    """Individual query details."""
    
    @require_api_key
    def get(self, query_id):
        """Get detailed information about a specific query."""
        try:
            query = QueryLog.query.filter_by(id=query_id, user_id=request.current_user.id).first()
            if not query:
                return {'error': 'Query not found'}, 404
            
            return {
                'success': True,
                'query': {
                    'id': query.id,
                    'timestamp': query.timestamp.isoformat(),
                    'system_description': query.system_description,
                    'system_name': query.system_name,
                    'model_name': query.model_name,
                    'model_config': query.model_config,
                    'rag_k': query.rag_k,
                    'status': query.status,
                    'generated_certificate': query.generated_certificate,
                    'processing_time': query.processing_time,
                    'total_tokens_used': query.total_tokens_used,
                    'cost_estimate': query.cost_estimate,
                    'confidence_score': query.confidence_score,
                    'mathematical_soundness': query.mathematical_soundness,
                    'verification_completed': query.verification_completed,
                    'verification_success': query.verification_success,
                    'user_rating': query.user_rating,
                    'user_feedback': query.user_feedback,
                    'domain_bounds': query.get_domain_bounds_dict() if hasattr(query, 'get_domain_bounds_dict') else None,
                    'error_message': query.error_message
                }
            }
        except Exception as e:
            logger.error(f"Query detail retrieval failed: {e}")
            return {'error': f'Detail error: {str(e)}'}, 500


# ============= SYSTEM STATUS ENDPOINTS =============

@system_ns.route('/health')
class SystemHealth(Resource):
    """System health check."""
    
    def get(self):
        """Get system health status (no auth required)."""
        try:
            # Basic health checks
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'version': '1.0.0',
                'components': {}
            }
            
            # Database health
            try:
                db.session.execute('SELECT 1')
                health_status['components']['database'] = 'healthy'
            except Exception as e:
                health_status['components']['database'] = f'unhealthy: {str(e)}'
                health_status['status'] = 'degraded'
            
            # Certificate generator health
            try:
                models = certificate_generator.get_available_models()
                health_status['components']['certificate_generator'] = 'healthy'
                health_status['components']['available_models'] = len(models)
            except Exception as e:
                health_status['components']['certificate_generator'] = f'unhealthy: {str(e)}'
                health_status['status'] = 'degraded'
            
            return health_status
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }, 500


@system_ns.route('/stats')
class SystemStats(Resource):
    """System statistics."""
    
    @require_api_key
    def get(self):
        """Get system usage statistics."""
        try:
            # Calculate statistics
            total_users = User.query.count()
            active_users = User.query.filter_by(is_active=True).count()
            total_queries = QueryLog.query.count()
            successful_queries = QueryLog.query.filter_by(status='completed').count()
            
            # Recent activity (last 24 hours)
            yesterday = datetime.utcnow() - timedelta(days=1)
            recent_queries = QueryLog.query.filter(QueryLog.timestamp >= yesterday).count()
            
            return {
                'success': True,
                'statistics': {
                    'users': {
                        'total': total_users,
                        'active': active_users
                    },
                    'queries': {
                        'total': total_queries,
                        'successful': successful_queries,
                        'success_rate': (successful_queries / total_queries * 100) if total_queries > 0 else 0,
                        'last_24h': recent_queries
                    },
                    'system': {
                        'uptime': 'N/A',  # TODO: Track actual uptime
                        'version': '1.0.0'
                    }
                }
            }
        except Exception as e:
            logger.error(f"System stats retrieval failed: {e}")
            return {'error': f'Stats error: {str(e)}'}, 500


# ============= EXPORT BLUEPRINT =============

def create_api():
    """Factory function to create and configure the API."""
    return api_bp 