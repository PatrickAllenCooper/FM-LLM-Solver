export interface User {
  id: string;
  email: string;
  password_hash: string;
  role: 'admin' | 'researcher' | 'viewer';
  created_at: Date;
  updated_at: Date;
  last_login_at?: Date;
}

export interface SystemSpec {
  id: string;
  owner_user_id: string;
  name: string;
  description?: string;
  system_type: 'continuous' | 'discrete' | 'hybrid';
  dimension: number;
  dynamics_json: Record<string, any>; // Mathematical system representation
  constraints_json?: Record<string, any>; // System constraints
  initial_set_json?: Record<string, any>; // Initial conditions
  safe_set_json?: Record<string, any>; // Safe regions
  unsafe_set_json?: Record<string, any>; // Unsafe regions
  created_by: string;
  created_at: Date;
  updated_at: Date;
  spec_version: string;
  hash: string; // For reproducibility
}

export interface Candidate {
  id: string;
  system_spec_id: string;
  certificate_type: 'lyapunov' | 'barrier' | 'inductive_invariant';
  generation_method: 'llm' | 'sos' | 'sdp' | 'quadratic_template';
  llm_provider?: string;
  llm_model?: string;
  llm_mode?: 'direct_expression' | 'basis_coeffs' | 'structure_constraints';
  llm_config_json?: Record<string, any>;
  candidate_expression: string; // Mathematical expression
  candidate_json: Record<string, any>; // Raw candidate data
  acceptance_status: 'pending' | 'accepted' | 'failed' | 'timeout';
  margin?: number;
  created_by: string;
  created_at: Date;
  updated_at: Date;
  accepted_at?: Date;
  generation_duration_ms?: number;
  acceptance_duration_ms?: number;
}

export interface Counterexample {
  id: string;
  candidate_id: string;
  x_json: Record<string, any>; // State where certificate fails
  context: string; // Human-readable explanation
  violation_metrics_json?: Record<string, any>; // Quantitative violation data
  created_at: Date;
}

export interface AuditEvent {
  id: string;
  user_id: string;
  action: string; // e.g., 'create_candidate', 'verify_certificate'
  entity_type: string; // e.g., 'candidate', 'system_spec'
  entity_id: string;
  at: Date;
  ip?: string;
  user_agent?: string;
}

export interface ExperimentRun {
  id: string;
  name: string;
  description?: string;
  system_spec_id: string;
  config_json: Record<string, any>; // Experiment configuration
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  created_by: string;
  created_at: Date;
  started_at?: Date;
  completed_at?: Date;
  total_attempts: number;
  successful_attempts: number;
  baseline_comparison_json?: Record<string, any>;
}

// For database operations
export type CreateUser = Omit<User, 'id' | 'created_at' | 'updated_at'>;
export type UpdateUser = Partial<Pick<User, 'email' | 'role' | 'last_login_at'>>;

export type CreateSystemSpec = Omit<SystemSpec, 'id' | 'created_at' | 'updated_at' | 'hash'>;
export type UpdateSystemSpec = Partial<Pick<SystemSpec, 'name' | 'description'>>;

export type CreateCandidate = Omit<Candidate, 'id' | 'created_at' | 'updated_at'>;
export type UpdateCandidate = Partial<Pick<Candidate, 'acceptance_status' | 'margin' | 'accepted_at' | 'generation_duration_ms' | 'acceptance_duration_ms'>>;

export type CreateCounterexample = Omit<Counterexample, 'id' | 'created_at'>;
export type CreateAuditEvent = Omit<AuditEvent, 'id' | 'at'>;
export type CreateExperimentRun = Omit<ExperimentRun, 'id' | 'created_at' | 'started_at' | 'completed_at'>;
