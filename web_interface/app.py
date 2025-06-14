from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, abort
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
import sys
import json
import traceback
import threading
import queue
import uuid

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.config_loader import load_config
from web_interface.models import init_db, QueryLog, VerificationResult
from web_interface.certificate_generator import CertificateGenerator
from web_interface.verification_service import VerificationService

app = Flask(__name__)

# Load configuration
config = load_config()
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Database configuration
db_path = config.get('web_interface', {}).get('database_path', 'web_interface/instance/app.db')
# Convert to absolute path to avoid SQLite issues
if not os.path.isabs(db_path):
    db_path = os.path.join(PROJECT_ROOT, db_path)
os.makedirs(os.path.dirname(db_path), exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db = init_db(app)

# Add custom Jinja2 filters
@app.template_filter('fromjson')
def fromjson_filter(value):
    """Parse JSON string in templates."""
    if value is None:
        return {}
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}

@app.template_filter('tojson')
def tojson_filter(value):
    """Convert Python object to JSON string in templates."""
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return '{}'

# Initialize services
certificate_generator = CertificateGenerator(config)
verification_service = VerificationService(config)

# Store for background tasks
background_tasks = {}

@app.route('/')
def index():
    """Main interface for querying models."""
    # Get available model configurations
    model_configs = certificate_generator.get_available_models()
    recent_queries = db.session.query(QueryLog).order_by(QueryLog.timestamp.desc()).limit(10).all()
    
    return render_template('index.html', 
                         model_configs=model_configs, 
                         recent_queries=recent_queries)

@app.route('/query', methods=['POST'])
def submit_query():
    """Handle query submission."""
    try:
        data = request.get_json()
        
        # Validate input
        system_description = data.get('system_description', '').strip()
        model_config = data.get('model_config', 'base')
        rag_k = int(data.get('rag_k', 3))
        
        if not system_description:
            return jsonify({'error': 'System description is required'}), 400
        
        # Create query log entry
        query = QueryLog(
            system_description=system_description,
            model_config=model_config,
            rag_k=rag_k,
            status='processing'
        )
        db.session.add(query)
        db.session.commit()
        
        # Start background task for generation and verification
        task_id = str(uuid.uuid4())
        
        # Initialize task tracking BEFORE starting thread to avoid race condition
        background_tasks[task_id] = {
            'status': 'processing',
            'query_id': query.id,
            'progress': 0
        }
        
        task_thread = threading.Thread(
            target=process_query_background,
            args=(task_id, query.id, system_description, model_config, rag_k)
        )
        task_thread.daemon = True
        task_thread.start()
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'query_id': query.id,
            'message': 'Query submitted successfully'
        })
        
    except Exception as e:
        app.logger.error(f"Error submitting query: {str(e)}")
        return jsonify({'error': f'Failed to submit query: {str(e)}'}), 500

def process_query_background(task_id, query_id, system_description, model_config, rag_k):
    """Background task to process query generation and verification."""
    # Ensure we have Flask application context for database operations
    with app.app_context():
        try:
            # Check if task still exists (in case of cleanup)
            if task_id not in background_tasks:
                app.logger.warning(f"Task {task_id} not found in background_tasks, aborting")
                return
            
            # Start timing
            from datetime import datetime
            start_time = datetime.utcnow()
            
            # Update query with start time
            query = db.session.get(QueryLog, query_id)
            if query:
                query.processing_start = start_time
                db.session.commit()
            
            # Update task progress
            background_tasks[task_id]['progress'] = 10
            background_tasks[task_id]['status'] = 'generating'
            
            # Generate certificate
            generation_result = certificate_generator.generate_certificate(
                system_description, model_config, rag_k
            )
            
            background_tasks[task_id]['progress'] = 50
            
            # Update query with generation result
            query = db.session.get(QueryLog, query_id)
            if query is None:
                app.logger.error(f"Query {query_id} not found in database")
                background_tasks[task_id]['status'] = 'failed'
                background_tasks[task_id]['error'] = 'Query not found in database'
                return
            
            query.llm_output = generation_result.get('llm_output', '')
            query.generated_certificate = generation_result.get('certificate', '')
            query.context_chunks = generation_result.get('context_chunks', 0)
            
            if not generation_result.get('success'):
                query.status = 'failed'
                query.error_message = generation_result.get('error', 'Generation failed')
                query.processing_end = datetime.utcnow()
                db.session.commit()
                background_tasks[task_id]['status'] = 'failed'
                background_tasks[task_id]['error'] = query.error_message
                return
            
            background_tasks[task_id]['status'] = 'verifying'
            background_tasks[task_id]['progress'] = 70
            
            # Verify certificate if generated successfully
            if query.generated_certificate:
                verification_result = verification_service.verify_certificate(
                    query.generated_certificate, system_description
                )
                
                # Create verification result record
                verification = VerificationResult(
                    query_id=query_id,
                    numerical_check_passed=verification_result.get('numerical_passed', False),
                    symbolic_check_passed=verification_result.get('symbolic_passed', False),
                    sos_check_passed=verification_result.get('sos_passed', False),
                    verification_details=json.dumps(verification_result.get('details', {})),
                    overall_success=verification_result.get('overall_success', False)
                )
                db.session.add(verification)
                
                query.verification_summary = json.dumps({
                    'numerical': verification_result.get('numerical_passed', False),
                    'symbolic': verification_result.get('symbolic_passed', False),
                    'sos': verification_result.get('sos_passed', False),
                    'overall': verification_result.get('overall_success', False)
                })
            
            query.status = 'completed'
            query.processing_end = datetime.utcnow()
            db.session.commit()
            
            background_tasks[task_id]['status'] = 'completed'
            background_tasks[task_id]['progress'] = 100
            
        except Exception as e:
            app.logger.error(f"Background task error: {str(e)}")
            app.logger.error(traceback.format_exc())
            
            try:
                # Update query with error (with additional error handling)
                query = db.session.get(QueryLog, query_id)
                if query:
                    query.status = 'failed'
                    query.error_message = str(e)
                    query.processing_end = datetime.utcnow()
                    db.session.commit()
                
                # Update background task status
                if task_id in background_tasks:
                    background_tasks[task_id]['status'] = 'failed'
                    background_tasks[task_id]['error'] = str(e)
                    
            except Exception as inner_e:
                app.logger.error(f"Error updating failed task status: {str(inner_e)}")

@app.route('/task_status/<task_id>')
def get_task_status(task_id):
    """Get status of background task."""
    task = background_tasks.get(task_id)
    if not task:
        return jsonify({
            'status': 'not_found',
            'error': 'Task not found. It may have completed and been cleaned up.'
        }), 404
    
    response = {
        'status': task['status'],
        'progress': task.get('progress', 0)
    }
    
    if task['status'] == 'failed':
        response['error'] = task.get('error', 'Unknown error')
    elif task['status'] == 'completed':
        # Get query results
        query = db.session.get(QueryLog, task['query_id'])
        if query:
            response['query'] = {
                'id': query.id,
                'generated_certificate': query.generated_certificate,
                'verification_summary': json.loads(query.verification_summary) if query.verification_summary else None
            }
        
        # Clean up completed task after a delay to allow final status check
        cleanup_completed_task(task_id)
    
    return jsonify(response)

def cleanup_completed_task(task_id, delay_seconds=30):
    """Clean up completed task after a delay."""
    def cleanup():
        import time
        time.sleep(delay_seconds)
        if task_id in background_tasks:
            task_status = background_tasks[task_id].get('status')
            if task_status in ['completed', 'failed']:
                del background_tasks[task_id]
                app.logger.info(f"Cleaned up background task: {task_id}")
    
    cleanup_thread = threading.Thread(target=cleanup)
    cleanup_thread.daemon = True
    cleanup_thread.start()

@app.route('/query/<int:query_id>')
def view_query(query_id):
    """View detailed results for a specific query."""
    query = db.session.get(QueryLog, query_id)
    if query is None:
        abort(404)
    verification = db.session.query(VerificationResult).filter_by(query_id=query_id).first()
    
    verification_details = None
    if verification and verification.verification_details:
        try:
            verification_details = json.loads(verification.verification_details)
        except json.JSONDecodeError:
            verification_details = None
    
    return render_template('query_detail.html', 
                         query=query, 
                         verification=verification,
                         verification_details=verification_details)

@app.route('/history')
def query_history():
    """View query history with pagination."""
    page = request.args.get('page', 1, type=int)
    queries = db.paginate(
        db.select(QueryLog).order_by(QueryLog.timestamp.desc()),
        page=page, per_page=20, error_out=False
    )
    
    return render_template('history.html', queries=queries)

@app.route('/api/stats')
def get_stats():
    """Get application statistics."""
    total_queries = db.session.query(QueryLog).count()
    successful_queries = db.session.query(QueryLog).filter_by(status='completed').count()
    failed_queries = db.session.query(QueryLog).filter_by(status='failed').count()
    
    # Verification statistics
    verifications = db.session.query(VerificationResult).all()
    numerical_passed = sum(1 for v in verifications if v.numerical_check_passed)
    symbolic_passed = sum(1 for v in verifications if v.symbolic_check_passed)
    sos_passed = sum(1 for v in verifications if v.sos_check_passed)
    overall_passed = sum(1 for v in verifications if v.overall_success)
    
    return jsonify({
        'total_queries': total_queries,
        'successful_queries': successful_queries,
        'failed_queries': failed_queries,
        'success_rate': (successful_queries / total_queries * 100) if total_queries > 0 else 0,
        'verification_stats': {
            'total_verified': len(verifications),
            'numerical_passed': numerical_passed,
            'symbolic_passed': symbolic_passed,
            'sos_passed': sos_passed,
            'overall_passed': overall_passed
        }
    })

@app.route('/about')
def about():
    """About page with project information."""
    return render_template('about.html')

@app.route('/favicon.ico')
def favicon():
    """Handle favicon requests."""
    # Return a simple response to prevent 404 errors
    # In production, you might want to serve an actual favicon file
    from flask import Response
    return Response(status=204)  # No Content

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    # Configuration from config.yaml
    web_config = config.get('web_interface', {})
    host = web_config.get('host', '127.0.0.1')
    port = web_config.get('port', 5000)
    debug = web_config.get('debug', True)
    
    app.run(host=host, port=port, debug=debug) 