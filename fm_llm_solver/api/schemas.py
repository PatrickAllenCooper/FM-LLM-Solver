"""
API Schemas for FM-LLM-Solver

This module defines request/response schemas for API validation and documentation.
"""

from flask_restx import fields, Model
from typing import Dict, Any


class APISchemas:
    """API schema definitions for request/response validation."""
    
    @staticmethod
    def get_auth_schemas(api) -> Dict[str, Model]:
        """Authentication schemas."""
        return {
            'token_request': api.model('TokenRequest', {
                'username': fields.String(required=True, description='Username'),
                'password': fields.String(required=True, description='Password')
            }),
            
            'token_response': api.model('TokenResponse', {
                'api_key': fields.String(description='Generated API key'),
                'expires_in': fields.Integer(description='Expiration time in seconds'),
                'user_id': fields.Integer(description='User ID'),
                'username': fields.String(description='Username'),
                'role': fields.String(description='User role')
            }),
            
            'validation_response': api.model('ValidationResponse', {
                'valid': fields.Boolean(description='Whether token is valid'),
                'user_id': fields.Integer(description='User ID'),
                'username': fields.String(description='Username'),
                'role': fields.String(description='User role'),
                'api_requests_count': fields.Integer(description='Total API requests'),
                'last_used': fields.String(description='Last usage timestamp')
            })
        }
    
    @staticmethod
    def get_certificate_schemas(api) -> Dict[str, Model]:
        """Certificate generation schemas."""
        return {
            'rag_config': api.model('RAGConfig', {
                'rag_k': fields.Integer(description='Number of RAG chunks', default=3),
                'dataset_type': fields.String(description='Dataset type', enum=['unified', 'discrete', 'continuous', 'custom']),
                'selected_researchers': fields.List(fields.String, description='Selected researcher IDs'),
                'selected_topics': fields.List(fields.String, description='Selected research topics')
            }),
            
            'generation_request': api.model('CertificateGenerationRequest', {
                'system_description': fields.String(required=True, description='System dynamics description'),
                'model_config': fields.String(description='Model configuration key', default='base'),
                'rag_config': fields.Nested('RAGConfig', description='RAG configuration'),
                'domain_bounds': fields.Raw(description='Domain bounds dictionary'),
                'system_name': fields.String(description='Optional system name'),
                'tags': fields.List(fields.String, description='User-defined tags')
            }),
            
            'generation_response': api.model('CertificateGenerationResponse', {
                'success': fields.Boolean(description='Whether generation succeeded'),
                'query_id': fields.Integer(description='Query ID for tracking'),
                'certificate': fields.String(description='Generated barrier certificate'),
                'model_used': fields.String(description='Model that was used'),
                'processing_time': fields.Float(description='Processing time in seconds'),
                'confidence_score': fields.Float(description='Confidence in the result'),
                'format_type': fields.String(description='Certificate format type'),
                'cost_estimate': fields.Float(description='Estimated cost'),
                'context_chunks': fields.Integer(description='Number of RAG chunks used'),
                'error': fields.String(description='Error message if failed')
            }),
            
            'verification_response': api.model('VerificationResponse', {
                'success': fields.Boolean(description='Whether verification succeeded'),
                'query_id': fields.Integer(description='Query ID'),
                'verification_result': fields.Raw(description='Detailed verification results')
            })
        }
    
    @staticmethod
    def get_model_schemas(api) -> Dict[str, Model]:
        """Model management schemas."""
        return {
            'model_info': api.model('ModelInfo', {
                'key': fields.String(description='Model key'),
                'name': fields.String(description='Model display name'),
                'description': fields.String(description='Model description'),
                'type': fields.String(description='Model type', enum=['base', 'finetuned', 'generic']),
                'barrier_type': fields.String(description='Barrier type', enum=['unified', 'discrete', 'continuous']),
                'provider': fields.String(description='Model provider (for generic models)'),
                'available': fields.Boolean(description='Whether model is available'),
                'cost_per_1k_input': fields.Float(description='Cost per 1K input tokens'),
                'cost_per_1k_output': fields.Float(description='Cost per 1K output tokens')
            }),
            
            'models_response': api.model('ModelsResponse', {
                'success': fields.Boolean(description='Whether request succeeded'),
                'models': fields.List(fields.Nested('ModelInfo'), description='Available models'),
                'total_count': fields.Integer(description='Total number of models')
            }),
            
            'model_test_response': api.model('ModelTestResponse', {
                'success': fields.Boolean(description='Whether test succeeded'),
                'test_result': fields.Raw(description='Test result details')
            })
        }
    
    @staticmethod
    def get_rag_schemas(api) -> Dict[str, Model]:
        """RAG configuration schemas."""
        return {
            'researcher_info': api.model('ResearcherInfo', {
                'scholar_id': fields.String(description='Google Scholar ID'),
                'name': fields.String(description='Researcher name'),
                'institution': fields.String(description='Institution'),
                'research_focus': fields.String(description='Research focus areas')
            }),
            
            'researchers_response': api.model('ResearchersResponse', {
                'success': fields.Boolean(description='Whether request succeeded'),
                'researchers': fields.List(fields.Nested('ResearcherInfo'), description='Available researchers'),
                'total_count': fields.Integer(description='Total number of researchers')
            }),
            
            'dataset_preview_request': api.model('DatasetPreviewRequest', {
                'dataset_type': fields.String(description='Dataset type', enum=['unified', 'discrete', 'continuous', 'custom']),
                'selected_researchers': fields.List(fields.String, description='Selected researcher IDs'),
                'selected_topics': fields.List(fields.String, description='Selected topics')
            }),
            
            'dataset_preview': api.model('DatasetPreview', {
                'paper_count': fields.Integer(description='Number of papers'),
                'chunk_count': fields.Integer(description='Number of chunks'),
                'quality_score': fields.Integer(description='Quality score (0-100)'),
                'dataset_type': fields.String(description='Dataset type'),
                'selected_researchers_count': fields.Integer(description='Number of selected researchers'),
                'selected_topics_count': fields.Integer(description='Number of selected topics')
            }),
            
            'dataset_preview_response': api.model('DatasetPreviewResponse', {
                'success': fields.Boolean(description='Whether request succeeded'),
                'dataset_preview': fields.Nested('DatasetPreview', description='Dataset preview statistics')
            })
        }
    
    @staticmethod
    def get_user_schemas(api) -> Dict[str, Model]:
        """User management schemas."""
        return {
            'user_profile': api.model('UserProfile', {
                'user_id': fields.Integer(description='User ID'),
                'username': fields.String(description='Username'),
                'email': fields.String(description='Email address'),
                'role': fields.String(description='User role'),
                'created_at': fields.String(description='Account creation date'),
                'last_login': fields.String(description='Last login timestamp'),
                'certificates_generated': fields.Integer(description='Total certificates generated'),
                'successful_verifications': fields.Integer(description='Successful verifications'),
                'daily_request_limit': fields.Integer(description='Daily request limit'),
                'monthly_request_limit': fields.Integer(description='Monthly request limit'),
                'requests_today': fields.Integer(description='Requests made today'),
                'requests_month': fields.Integer(description='Requests made this month')
            }),
            
            'usage_summary': api.model('UsageSummary', {
                'total_queries': fields.Integer(description='Total queries made'),
                'successful_generations': fields.Integer(description='Successful generations'),
                'requests_today': fields.Integer(description='Requests today'),
                'requests_month': fields.Integer(description='Requests this month'),
                'daily_limit': fields.Integer(description='Daily limit'),
                'monthly_limit': fields.Integer(description='Monthly limit')
            }),
            
            'query_summary': api.model('QuerySummary', {
                'id': fields.Integer(description='Query ID'),
                'timestamp': fields.String(description='Query timestamp'),
                'status': fields.String(description='Query status'),
                'model_name': fields.String(description='Model used'),
                'has_certificate': fields.Boolean(description='Whether certificate was generated'),
                'processing_time': fields.Float(description='Processing time')
            }),
            
            'usage_response': api.model('UsageResponse', {
                'usage_summary': fields.Nested('UsageSummary', description='Usage summary'),
                'recent_queries': fields.List(fields.Nested('QuerySummary'), description='Recent queries')
            })
        }
    
    @staticmethod
    def get_query_schemas(api) -> Dict[str, Model]:
        """Query history schemas."""
        return {
            'query_item': api.model('QueryItem', {
                'id': fields.Integer(description='Query ID'),
                'timestamp': fields.String(description='Query timestamp'),
                'system_description': fields.String(description='System description'),
                'system_name': fields.String(description='System name'),
                'model_name': fields.String(description='Model used'),
                'status': fields.String(description='Query status'),
                'has_certificate': fields.Boolean(description='Whether certificate was generated'),
                'processing_time': fields.Float(description='Processing time'),
                'cost_estimate': fields.Float(description='Cost estimate'),
                'confidence_score': fields.Float(description='Confidence score'),
                'user_rating': fields.Integer(description='User rating (1-5)')
            }),
            
            'pagination_info': api.model('PaginationInfo', {
                'page': fields.Integer(description='Current page'),
                'per_page': fields.Integer(description='Items per page'),
                'total': fields.Integer(description='Total items'),
                'pages': fields.Integer(description='Total pages'),
                'has_next': fields.Boolean(description='Whether there is a next page'),
                'has_prev': fields.Boolean(description='Whether there is a previous page')
            }),
            
            'history_response': api.model('HistoryResponse', {
                'success': fields.Boolean(description='Whether request succeeded'),
                'queries': fields.List(fields.Nested('QueryItem'), description='Query list'),
                'pagination': fields.Nested('PaginationInfo', description='Pagination information')
            }),
            
            'query_detail': api.model('QueryDetail', {
                'id': fields.Integer(description='Query ID'),
                'timestamp': fields.String(description='Query timestamp'),
                'system_description': fields.String(description='System description'),
                'system_name': fields.String(description='System name'),
                'model_name': fields.String(description='Model used'),
                'model_config': fields.Raw(description='Model configuration'),
                'rag_k': fields.Integer(description='RAG chunks used'),
                'status': fields.String(description='Query status'),
                'generated_certificate': fields.String(description='Generated certificate'),
                'processing_time': fields.Float(description='Processing time'),
                'total_tokens_used': fields.Integer(description='Total tokens used'),
                'cost_estimate': fields.Float(description='Cost estimate'),
                'confidence_score': fields.Float(description='Confidence score'),
                'mathematical_soundness': fields.Float(description='Mathematical soundness score'),
                'verification_completed': fields.Boolean(description='Whether verification was completed'),
                'verification_success': fields.Boolean(description='Whether verification succeeded'),
                'user_rating': fields.Integer(description='User rating (1-5)'),
                'user_feedback': fields.String(description='User feedback'),
                'domain_bounds': fields.Raw(description='Domain bounds'),
                'error_message': fields.String(description='Error message if failed')
            }),
            
            'detail_response': api.model('DetailResponse', {
                'success': fields.Boolean(description='Whether request succeeded'),
                'query': fields.Nested('QueryDetail', description='Query details')
            })
        }
    
    @staticmethod
    def get_system_schemas(api) -> Dict[str, Model]:
        """System status schemas."""
        return {
            'health_response': api.model('HealthResponse', {
                'status': fields.String(description='Overall system status'),
                'timestamp': fields.String(description='Health check timestamp'),
                'version': fields.String(description='System version'),
                'components': fields.Raw(description='Component health status')
            }),
            
            'user_stats': api.model('UserStats', {
                'total': fields.Integer(description='Total users'),
                'active': fields.Integer(description='Active users')
            }),
            
            'query_stats': api.model('QueryStats', {
                'total': fields.Integer(description='Total queries'),
                'successful': fields.Integer(description='Successful queries'),
                'success_rate': fields.Float(description='Success rate percentage'),
                'last_24h': fields.Integer(description='Queries in last 24 hours')
            }),
            
            'system_info': api.model('SystemInfo', {
                'uptime': fields.String(description='System uptime'),
                'version': fields.String(description='System version')
            }),
            
            'statistics': api.model('Statistics', {
                'users': fields.Nested('UserStats', description='User statistics'),
                'queries': fields.Nested('QueryStats', description='Query statistics'),
                'system': fields.Nested('SystemInfo', description='System information')
            }),
            
            'stats_response': api.model('StatsResponse', {
                'success': fields.Boolean(description='Whether request succeeded'),
                'statistics': fields.Nested('Statistics', description='System statistics')
            })
        }
    
    @staticmethod
    def get_error_schemas(api) -> Dict[str, Model]:
        """Error response schemas."""
        return {
            'error_response': api.model('ErrorResponse', {
                'error': fields.String(description='Error message'),
                'code': fields.String(description='Error code'),
                'timestamp': fields.String(description='Error timestamp'),
                'details': fields.Raw(description='Additional error details')
            })
        } 