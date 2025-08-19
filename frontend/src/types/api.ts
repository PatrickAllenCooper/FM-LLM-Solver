export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  timestamp: string;
}

export interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    total_pages: number;
    has_next: boolean;
    has_prev: boolean;
  };
}

export interface User {
  id: string;
  email: string;
  role: 'admin' | 'researcher' | 'viewer';
  created_at: string;
  updated_at: string;
  last_login_at?: string;
}

export interface AuthResponse {
  token: string;
  user: {
    id: string;
    email: string;
    role: string;
  };
  expires_at: string;
}

export interface SystemSpec {
  id: string;
  name: string;
  description?: string;
  system_type: 'continuous' | 'discrete' | 'hybrid';
  dimension: number;
  dynamics_json: any;
  constraints_json?: any;
  initial_set_json?: any;
  unsafe_set_json?: any;
  created_by: string;
  created_at: string;
  spec_version: string;
  spec_hash: string;
}

export interface AcceptanceResult {
  accepted: boolean;
  margin?: number;
  counterexample?: {
    state: Record<string, number>;
    violation_type: string;
    violation_magnitude: number;
  };
  acceptance_method: 'symbolic' | 'numerical' | 'smt' | 'mathematical';
  duration_ms: number;
  solver_output?: string;
  technical_details?: {
    conditions_checked: string[];
    sampling_method: 'uniform' | 'sobol' | 'lhs' | 'adaptive';
    sample_count: number;
    domain_coverage: Record<string, { min: number; max: number }>;
    violation_analysis: {
      total_violations: number;
      violation_points: Array<{
        point: Record<string, number>;
        condition: string;
        value: number;
        severity: 'minor' | 'moderate' | 'severe';
      }>;
    };
    margin_breakdown: {
      positivity_margin?: number;
      decreasing_margin?: number;
      separation_margin?: number;
      invariant_margin?: number;
    };
    numerical_parameters: {
      tolerance: number;
      max_iterations: number;
      convergence_threshold: number;
    };
    stage_results: {
      stage_a_passed: boolean;
      stage_b_enabled: boolean;
      stage_b_passed?: boolean;
    };
  };
}

export interface Candidate {
  id: string;
  system_spec_id: string;
  certificate_type: 'lyapunov' | 'barrier' | 'inductive_invariant';
  generation_method: 'llm' | 'sos' | 'sdp' | 'quadratic_template' | 'conversational';
  llm_provider?: string;
  llm_model?: string;
  llm_mode?: 'direct_expression' | 'basis_coeffs' | 'structure_constraints';
  llm_config_json?: any;
  candidate_expression: string;
  candidate_json: any;
  acceptance_status: 'pending' | 'accepted' | 'failed' | 'timeout';
  margin?: number;
  created_by: string;
  created_at: string;
  accepted_at?: string;
  generation_duration_ms?: number;
  acceptance_duration_ms?: number;
  system_name?: string; // Joined from system_specs
  counterexamples?: Counterexample[];
  acceptance_result?: AcceptanceResult;
  // Conversational mode specific fields
  conversation_id?: string;
  conversation_context?: string;
  conversation_summary?: ConversationSummary;
}

// Conversational mode interfaces
export interface ConversationMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  metadata?: {
    token_count?: number;
    model_used?: string;
    message_type?: 'question' | 'insight' | 'approach' | 'refinement' | 'final';
  };
}

export interface ConversationSummary {
  key_insights: string[];
  mathematical_approaches_discussed: string[];
  final_approach_rationale: string;
  conversation_summary: string;
  total_tokens_used: number;
  summarization_timestamp: string;
}

export interface Conversation {
  id: string;
  system_spec_id: string;
  certificate_type: 'lyapunov' | 'barrier' | 'inductive_invariant';
  status: 'active' | 'summarized' | 'published' | 'abandoned';
  messages: ConversationMessage[];
  summary?: ConversationSummary;
  final_certificate_id?: string;
  created_by: string;
  created_at: string;
  updated_at: string;
  published_at?: string;
  token_count: number;
  message_count: number;
  // UI specific fields
  message_preview?: string;
}

// Request interfaces for conversational mode
export interface StartConversationRequest {
  system_spec_id: string;
  certificate_type: 'lyapunov' | 'barrier' | 'inductive_invariant';
  initial_message?: string;
}

export interface SendMessageRequest {
  message: string;
  request_insights?: boolean;
  force_summarize?: boolean;
}

export interface PublishCertificateFromConversationRequest {
  conversation_id: string;
  final_instructions?: string;
}

export interface Counterexample {
  id: string;
  candidate_id: string;
  x_json: any;
  context: string;
  violation_metrics_json?: any;
  created_at: string;
}

export interface LLMConfig {
  provider: 'anthropic';
  model: string;
  temperature: number;
  max_tokens: number;
  max_attempts: number;
  mode: 'direct_expression' | 'basis_coeffs' | 'structure_constraints';
  timeout_ms: number;
}

export interface SystemSpecRequest {
  name: string;
  description?: string;
  system_type: 'continuous' | 'discrete' | 'hybrid';
  dimension: number;
  dynamics: {
    type: 'polynomial' | 'nonlinear' | 'linear' | 'piecewise';
    variables: string[];
    equations: string[];
    domain?: {
      bounds?: Record<string, { min?: number; max?: number }>;
      constraints?: string[];
    };
  };
  constraints?: any;
  initial_set?: any;
  unsafe_set?: any;
}

export interface CertificateGenerationRequest {
  system_spec_id: string;
  certificate_type: 'lyapunov' | 'barrier' | 'inductive_invariant';
  generation_method: 'llm' | 'sos' | 'sdp' | 'quadratic_template';
  llm_config?: LLMConfig;
  baseline_comparison?: boolean;
}

// Form types
export interface LoginForm {
  email: string;
  password: string;
}

export interface RegisterForm {
  email: string;
  password: string;
  confirmPassword: string;
  role?: 'admin' | 'researcher' | 'viewer';
}

export interface ChangePasswordForm {
  currentPassword: string;
  newPassword: string;
  confirmNewPassword: string;
}
