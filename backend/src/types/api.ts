import { z } from 'zod';

// Request/Response types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  timestamp: string;
}

export interface AuthRequest {
  email: string;
  password: string;
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

// LLM Configuration
export const LLMConfigSchema = z.object({
  provider: z.literal('anthropic'),
  model: z.string().default('claude-sonnet-4-20250514'),
  temperature: z.number().min(0).max(1).default(0.0),
  max_tokens: z.number().min(1).max(4096).default(2048),
  max_attempts: z.number().min(1).max(10).default(3),
  mode: z.enum(['direct_expression', 'basis_coeffs', 'structure_constraints']),
  timeout_ms: z.number().default(30000),
});

export type LLMConfig = z.infer<typeof LLMConfigSchema>;

// System specification schemas
export const DynamicsSchema = z.object({
  type: z.enum(['polynomial', 'nonlinear', 'linear', 'piecewise']),
  variables: z.array(z.string()),
  equations: z.array(z.string()), // Mathematical expressions
  domain: z.object({
    bounds: z.record(z.object({
      min: z.number().optional(),
      max: z.number().optional(),
    })).optional(),
    constraints: z.array(z.string()).optional(), // Inequality constraints
  }).optional(),
});

export const SystemSpecRequestSchema = z.object({
  name: z.string().min(1).max(255),
  description: z.string().optional(),
  system_type: z.enum(['continuous', 'discrete', 'hybrid']),
  dimension: z.number().min(1).max(20),
  dynamics: DynamicsSchema,
  constraints: z.record(z.any()).optional(),
  initial_set: z.record(z.any()).optional(),
  unsafe_set: z.record(z.any()).optional(),
});

export type SystemSpecRequest = z.infer<typeof SystemSpecRequestSchema>;

// Certificate generation request  
export const CertificateGenerationRequestSchema = z.object({
  system_spec_id: z.string().min(1), // Accept Firestore IDs, not just UUIDs
  certificate_type: z.enum(['lyapunov', 'barrier', 'inductive_invariant']),
  generation_method: z.enum(['llm', 'sos', 'sdp', 'quadratic_template']),
  llm_config: LLMConfigSchema.optional(),
  baseline_comparison: z.boolean().default(false),
});

export type CertificateGenerationRequest = z.infer<typeof CertificateGenerationRequestSchema>;

// Certificate response from LLM
export const LLMCertificateResponseSchema = z.object({
  certificate_type: z.enum(['lyapunov', 'barrier', 'inductive_invariant']),
  expression: z.string(), // Mathematical expression as string
  variables: z.array(z.string()),
  domain: z.object({
    bounds: z.record(z.object({
      min: z.number().optional(),
      max: z.number().optional(),
    })).optional(),
    description: z.string().optional(),
  }).optional(),
  properties: z.object({
    positive_definite: z.boolean().optional(),
    negative_definite: z.boolean().optional(),
    decreasing_along_trajectories: z.boolean().optional(),
    separates_safe_unsafe: z.boolean().optional(),
  }).optional(),
  reasoning: z.string().optional(),
  confidence: z.number().min(0).max(1).optional(),
});

export type LLMCertificateResponse = z.infer<typeof LLMCertificateResponseSchema>;

// Acceptance result
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
}

// Experiment configuration
export const ExperimentConfigSchema = z.object({
  name: z.string().min(1).max(255),
  description: z.string().optional(),
  system_specs: z.array(z.string().uuid()),
  llm_configs: z.array(LLMConfigSchema),
  baseline_methods: z.array(z.enum(['sos', 'sdp', 'quadratic_template'])),
  budget_per_spec: z.object({
    max_total_attempts: z.number().min(1).default(10),
    max_time_minutes: z.number().min(1).default(60),
  }),
  acceptance_config: z.object({
    timeout_seconds: z.number().min(1).default(30),
    numerical_precision: z.number().default(1e-6),
  }),
});

export type ExperimentConfig = z.infer<typeof ExperimentConfigSchema>;

// Pagination
export interface PaginationParams {
  page: number;
  limit: number;
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
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

// Error types
export interface ValidationError {
  field: string;
  message: string;
  value?: any;
}

export interface ApiError {
  code: string;
  message: string;
  details?: ValidationError[];
}
