"""
Main routes for FM-LLM Solver web interface.

Handles the primary user-facing web interface for certificate generation.
"""

import json
import uuid
import threading
from datetime import datetime
from typing import Dict, Any, Optional

from flask import (
    Blueprint,
    render_template,
    request,
    jsonify,
    current_app,
    redirect,
    url_for,
    flash,
)
from flask_login import login_required, current_user

from fm_llm_solver.core.logging import get_logger
from fm_llm_solver.web.models import (
    db,
    QueryLog,
    VerificationResult,
)
from fm_llm_solver.web.utils import rate_limit, validate_input

main_bp = Blueprint("main", __name__)
logger = get_logger(__name__)

# Background task storage
background_tasks: Dict[str, Dict[str, Any]] = {}


@main_bp.route("/")
def index():
    """Main interface for barrier certificate generation."""
    try:
        # Get available model configurations
        model_configs = getattr(current_app, "certificate_generator", None)
        if model_configs and hasattr(model_configs, "get_available_models"):
            model_configs = model_configs.get_available_models()
        else:
            # Fallback model configurations
            model_configs = [
                {
                    "key": "base",
                    "name": "Base Model",
                    "description": "Standard barrier certificate generation",
                },
                {
                    "key": "finetuned",
                    "name": "Fine-tuned Model",
                    "description": "Optimized for barrier certificates",
                },
            ]

        # Get recent queries for logged-in users
        recent_queries = []
        if current_user.is_authenticated:
            recent_queries = (
                db.session.query(QueryLog)
                .filter_by(user_id=current_user.id)
                .order_by(QueryLog.timestamp.desc())
                .limit(10)
                .all()
            )

        return render_template(
            "index.html", model_configs=model_configs, recent_queries=recent_queries
        )

    except Exception as e:
        logger.error(f"Error loading index page: {e}")
        flash("An error occurred loading the page. Please try again.", "error")
        return render_template("index.html", model_configs=[], recent_queries=[])


@main_bp.route("/query", methods=["POST"])
@login_required
@rate_limit(max_requests=50)
def submit_query():
    """Handle certificate generation query submission."""
    try:
        data = request.get_json()

        # Validate input
        system_description = validate_input(data.get("system_description", "").strip())
        model_config = data.get("model_config", "base")
        rag_k = int(data.get("rag_k", 3))
        domain_bounds = data.get("domain_bounds")

        # Verification parameter overrides
        verif_params = {
            "num_samples_lie": data.get("num_samples_lie"),
            "num_samples_boundary": data.get("num_samples_boundary"),
            "numerical_tolerance": data.get("numerical_tolerance"),
            "attempt_sos": data.get("attempt_sos"),
            "attempt_optimization": data.get("attempt_optimization"),
            "optimization_max_iter": data.get("optimization_max_iter"),
            "optimization_pop_size": data.get("optimization_pop_size"),
        }

        if not system_description:
            return jsonify({"error": "System description is required"}), 400

        # Create query log entry
        query = QueryLog(
            user_id=current_user.id,
            system_description=system_description,
            model_config=model_config,
            rag_k=rag_k,
            status="processing",
        )

        # Store domain bounds if provided
        if domain_bounds:
            query.set_domain_bounds_dict(domain_bounds)

        db.session.add(query)
        db.session.commit()

        # Start background task
        task_id = str(uuid.uuid4())
        background_tasks[task_id] = {"status": "processing", "query_id": query.id, "progress": 0}

        # Start processing thread
        task_thread = threading.Thread(
            target=_process_query_background,
            args=(
                task_id,
                query.id,
                system_description,
                model_config,
                rag_k,
                verif_params,
                domain_bounds,
            ),
            daemon=True,
        )
        task_thread.start()

        return jsonify(
            {
                "success": True,
                "task_id": task_id,
                "query_id": query.id,
                "message": "Query submitted successfully",
            }
        )

    except ValueError as e:
        logger.warning(f"Invalid input: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error submitting query: {e}")
        return jsonify({"error": "Failed to submit query. Please try again."}), 500


@main_bp.route("/task_status/<task_id>")
def get_task_status(task_id: str):
    """Get status of a background task."""
    try:
        if task_id not in background_tasks:
            return jsonify({"status": "not_found"}), 404

        task = background_tasks[task_id]

        # If task is completed, include query data
        if task["status"] == "completed" and "query_id" in task:
            query = db.session.get(QueryLog, task["query_id"])
            if query:
                return jsonify(
                    {"status": "completed", "progress": 100, "query": _serialize_query(query)}
                )

        return jsonify(task)

    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@main_bp.route("/query/<int:query_id>")
@login_required
def view_query(query_id: int):
    """View detailed query results."""
    try:
        query = db.session.get(QueryLog, query_id)

        if not query:
            flash("Query not found.", "error")
            return redirect(url_for("main.index"))

        # Check user access
        if query.user_id != current_user.id and not current_user.is_admin:
            flash("You do not have permission to view this query.", "error")
            return redirect(url_for("main.index"))

        # Get verification results
        verification = db.session.query(VerificationResult).filter_by(query_id=query_id).first()

        return render_template("query_detail.html", query=query, verification=verification)

    except Exception as e:
        logger.error(f"Error viewing query {query_id}: {e}")
        flash("An error occurred loading the query details.", "error")
        return redirect(url_for("main.index"))


@main_bp.route("/history")
@login_required
def query_history():
    """View user's query history."""
    try:
        page = request.args.get("page", 1, type=int)
        per_page = 20

        queries = (
            db.session.query(QueryLog)
            .filter_by(user_id=current_user.id)
            .order_by(QueryLog.timestamp.desc())
            .paginate(page=page, per_page=per_page, error_out=False)
        )

        return render_template("history.html", queries=queries)

    except Exception as e:
        logger.error(f"Error loading history: {e}")
        flash("An error occurred loading your query history.", "error")
        return render_template("history.html", queries=[])


def _process_query_background(
    task_id: str,
    query_id: int,
    system_description: str,
    model_config: str,
    rag_k: int,
    verif_params: Dict[str, Any],
    domain_bounds: Optional[Dict[str, Any]] = None,
):
    """Background task to process query generation and verification."""
    with current_app.app_context():
        try:
            # Check if task still exists
            if task_id not in background_tasks:
                logger.warning(f"Task {task_id} not found, aborting")
                return

            start_time = datetime.utcnow()

            # Update query with start time
            query = db.session.get(QueryLog, query_id)
            if query:
                query.processing_start = start_time
                db.session.commit()

            # Update progress: starting generation
            background_tasks[task_id].update({"progress": 10, "status": "generating"})

            # Generate certificate
            generator = getattr(current_app, "certificate_generator", None)
            if generator and hasattr(generator, "generate_certificate"):
                generation_result = generator.generate_certificate(
                    system_description, model_config, rag_k, domain_bounds
                )
            else:
                # Fallback for missing generator
                generation_result = {
                    "success": False,
                    "error": "Certificate generator not available",
                    "certificate": "",
                    "llm_output": "",
                    "context_chunks": [],
                }

            background_tasks[task_id]["progress"] = 50

            # Update query with generation result
            if query:
                query.llm_output = generation_result.get("llm_output", "")
                query.generated_certificate = generation_result.get("certificate", "")
                query.context_chunks = json.dumps(generation_result.get("context_chunks", []))

                if not generation_result.get("success"):
                    query.status = "failed"
                    query.error_message = generation_result.get("error", "Generation failed")
                    db.session.commit()

                    background_tasks[task_id].update(
                        {
                            "status": "failed",
                            "error": generation_result.get("error", "Generation failed"),
                        }
                    )
                    return

                db.session.commit()

            # Start verification
            background_tasks[task_id].update({"progress": 70, "status": "verifying"})

            verifier = getattr(current_app, "verifier", None)
            if verifier and hasattr(verifier, "verify_certificate"):
                verification_result = verifier.verify_certificate(
                    generation_result["certificate"],
                    system_description,
                    verif_params,
                    domain_bounds,
                )
            else:
                # Fallback for missing verifier
                verification_result = {
                    "numerical_passed": False,
                    "symbolic_passed": False,
                    "sos_passed": False,
                    "overall_success": False,
                    "details": {"error": "Verification service not available"},
                }

            # Create verification result record
            verification = VerificationResult(
                query_id=query_id,
                numerical_check_passed=verification_result.get("numerical_passed", False),
                symbolic_check_passed=verification_result.get("symbolic_passed", False),
                sos_check_passed=verification_result.get("sos_passed", False),
                verification_details=json.dumps(verification_result.get("details", {})),
                overall_success=verification_result.get("overall_success", False),
            )
            db.session.add(verification)

            # Update query
            if query:
                query.verification_summary = json.dumps(
                    {
                        "numerical": verification_result.get("numerical_passed", False),
                        "symbolic": verification_result.get("symbolic_passed", False),
                        "sos": verification_result.get("sos_passed", False),
                        "overall": verification_result.get("overall_success", False),
                    }
                )
                query.status = "completed"
                query.processing_end = datetime.utcnow()
                query.user_decision = "pending"

            db.session.commit()

            # Mark task as completed
            background_tasks[task_id].update({"status": "completed", "progress": 100})

            # Schedule cleanup
            threading.Timer(30.0, lambda: background_tasks.pop(task_id, None)).start()

        except Exception as e:
            logger.error(f"Background processing error for task {task_id}: {e}")

            # Update query with error
            try:
                query = db.session.get(QueryLog, query_id)
                if query:
                    query.status = "failed"
                    query.error_message = str(e)
                    query.processing_end = datetime.utcnow()
                    db.session.commit()

                background_tasks[task_id] = {"status": "failed", "error": str(e)}

            except Exception as inner_e:
                logger.error(f"Error updating failed task: {inner_e}")


def _serialize_query(query: QueryLog) -> Dict[str, Any]:
    """Serialize a query object for JSON response."""
    verification_result = db.session.query(VerificationResult).filter_by(query_id=query.id).first()

    return {
        "id": query.id,
        "system_description": query.system_description,
        "certificate": query.generated_certificate,
        "model_config": query.model_config,
        "rag_k": query.rag_k,
        "status": query.status,
        "timestamp": query.timestamp.isoformat(),
        "verification": (
            {
                "overall_success": (
                    verification_result.overall_success if verification_result else False
                ),
                "numerical_passed": (
                    verification_result.numerical_check_passed if verification_result else False
                ),
                "symbolic_passed": (
                    verification_result.symbolic_check_passed if verification_result else False
                ),
                "sos_passed": (
                    verification_result.sos_check_passed if verification_result else False
                ),
            }
            if verification_result
            else None
        ),
    }
