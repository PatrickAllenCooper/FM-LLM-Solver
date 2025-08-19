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

export interface Candidate {
  id: string;
  system_spec_id: string;
  certificate_type: 'lyapunov' | 'barrier' | 'inductive_invariant';
  generation_method: 'llm' | 'sos' | 'sdp' | 'quadratic_template';
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
