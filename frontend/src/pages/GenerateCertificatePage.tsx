import { useState, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useQuery, useMutation } from '@tanstack/react-query';
import { useNavigate, Link } from 'react-router-dom';
import { 
  ArrowLeftIcon,
  PlayIcon,
  InformationCircleIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';

import { CertificateGenerationRequest } from '@/types/api';
import { api } from '@/services/api';

// Form validation schema
const CertificateGenerationSchema = z.object({
  system_spec_id: z.string().min(1, 'Please select a system specification'),
  certificate_type: z.enum(['lyapunov', 'barrier', 'inductive_invariant']),
  generation_method: z.enum(['llm', 'sos', 'sdp', 'quadratic_template']),
  baseline_comparison: z.boolean().default(false),
  llm_config: z.object({
    provider: z.literal('anthropic').default('anthropic'),
    model: z.string().default('claude-3-5-sonnet-20241022'),
    temperature: z.number().min(0).max(1).default(0.0),
    max_tokens: z.number().min(1).max(4096).default(2048),
    max_attempts: z.number().min(1).max(10).default(3),
    mode: z.enum(['direct_expression', 'basis_coeffs', 'structure_constraints']).default('direct_expression'),
    timeout_ms: z.number().default(30000),
  }).optional(),
});

type CertificateGenerationForm = z.infer<typeof CertificateGenerationSchema>;

const CERTIFICATE_TYPE_INFO = {
  lyapunov: {
    title: 'Lyapunov Function',
    description: 'Proves stability by showing energy decreases along trajectories',
    useCases: ['System stability', 'Convergence analysis', 'Equilibrium verification']
  },
  barrier: {
    title: 'Barrier Certificate',
    description: 'Proves safety by maintaining separation between safe and unsafe regions',
    useCases: ['Safety verification', 'Invariant maintenance', 'Reachability analysis']
  },
  inductive_invariant: {
    title: 'Inductive Invariant',
    description: 'Proves properties that hold initially and are preserved over time',
    useCases: ['Property verification', 'Temporal logic', 'State space constraints']
  },
};

const METHOD_INFO = {
  llm: {
    title: 'LLM Generated',
    description: 'Use large language models to generate certificate candidates',
    pros: ['Novel approaches', 'Handles complex systems', 'Fast generation'],
    cons: ['May require verification', 'Quality varies'],
  },
  sos: {
    title: 'Sum of Squares (SOS)',
    description: 'Classical method using polynomial optimization',
    pros: ['Mathematically rigorous', 'Well-established', 'Deterministic'],
    cons: ['Limited to polynomial systems', 'Computationally expensive'],
  },
  sdp: {
    title: 'Semidefinite Programming (SDP)',
    description: 'Convex optimization approach for certificate synthesis',
    pros: ['Global optimality', 'Efficient for certain problems'],
    cons: ['Scalability issues', 'Numerical precision'],
  },
  quadratic_template: {
    title: 'Quadratic Template',
    description: 'Template-based approach using quadratic forms',
    pros: ['Fast computation', 'Good for linear systems'],
    cons: ['Limited expressiveness', 'May not find complex certificates'],
  },
};

const LLM_MODES = {
  direct_expression: 'Direct Expression - Generate complete mathematical expressions',
  basis_coeffs: 'Basis Coefficients - Generate coefficients for predefined basis functions',
  structure_constraints: 'Structure Constraints - Generate with structural constraints and templates',
};

export default function GenerateCertificatePage() {
  const navigate = useNavigate();
  const [showAdvanced, setShowAdvanced] = useState(false);

  const form = useForm<CertificateGenerationForm>({
    resolver: zodResolver(CertificateGenerationSchema),
    defaultValues: {
      system_spec_id: '',
      certificate_type: 'lyapunov',
      generation_method: 'llm',
      baseline_comparison: false,
      llm_config: {
        provider: 'anthropic',
        model: 'claude-3-5-sonnet-20241022',
        temperature: 0.0,
        max_tokens: 2048,
        max_attempts: 3,
        mode: 'direct_expression',
        timeout_ms: 30000,
      },
    },
    mode: 'all', // Validate on change and blur to keep form state accurate
  });

  // Fetch available system specifications
  const { data: systemSpecsResponse, isLoading: specsLoading } = useQuery({
    queryKey: ['system-specs'],
    queryFn: async () => {
      const response = await api.getSystemSpecs();
      return response; // This is PaginatedResponse<SystemSpec>
    },
  });

  const systemSpecs = systemSpecsResponse?.data || []; // Extract the actual SystemSpec[] array

  // Certificate generation mutation
  const generateMutation = useMutation({
    mutationFn: async (data: CertificateGenerationRequest) => {
      return await api.generateCertificate(data);
    },
    onSuccess: (certificate) => {
      toast.success('Certificate generation started!');
      navigate(`/certificates/${certificate.id}`);
    },
    onError: (error: any) => {
      toast.error(error?.response?.data?.error || 'Failed to start certificate generation');
    },
  });

  const onSubmit = (data: CertificateGenerationForm) => {
    const payload: CertificateGenerationRequest = {
      system_spec_id: data.system_spec_id,
      certificate_type: data.certificate_type,
      generation_method: data.generation_method,
      baseline_comparison: data.baseline_comparison,
    };

    // Only include LLM config for LLM generation method
    if (data.generation_method === 'llm' && data.llm_config) {
      payload.llm_config = data.llm_config;
    }

    generateMutation.mutate(payload);
  };

  const selectedMethod = form.watch('generation_method');
  const selectedType = form.watch('certificate_type');
  const isLLMMethod = selectedMethod === 'llm';

  // Force re-validation when system specs load
  useEffect(() => {
    if (systemSpecs.length > 0) {
      form.trigger(); // Re-validate entire form when data loads
    }
  }, [systemSpecs.length, form]);

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Header */}
      <div className="flex items-center space-x-4">
        <Link to="/certificates" className="btn btn-outline">
          <ArrowLeftIcon className="w-4 h-4 mr-2" />
          Back to Certificates
        </Link>
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Generate Certificate</h1>
          <p className="mt-1 text-gray-600">
            Create a new formal verification certificate for your system
          </p>
        </div>
      </div>

      <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
        {/* System Selection */}
        <div className="card">
          <div className="card-body">
            <h2 className="text-xl font-medium text-gray-900 mb-4">System Specification</h2>
            
            {specsLoading ? (
              <div className="flex items-center justify-center py-8">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                <span className="ml-3 text-gray-600">Loading system specifications...</span>
              </div>
            ) : !systemSpecs?.length ? (
              <div className="text-center py-8">
                <ExclamationTriangleIcon className="mx-auto h-10 w-10 text-yellow-400" />
                <h3 className="mt-2 text-sm font-medium text-gray-900">No system specifications found</h3>
                <p className="mt-1 text-sm text-gray-500">
                  You need to create a system specification first before generating certificates.
                </p>
                <div className="mt-4">
                  <Link to="/system-specs/create" className="btn btn-primary">
                    Create System Specification
                  </Link>
                </div>
              </div>
            ) : (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Select System Specification *
                </label>
                <select
                  {...form.register('system_spec_id')}
                  className="input"
                >
                  <option value="">Select a system specification...</option>
                  {systemSpecs.map((spec) => (
                    <option key={spec.id} value={spec.id}>
                      {spec.name} ({spec.system_type}, {spec.dimension}D)
                    </option>
                  ))}
                </select>
                {form.formState.errors.system_spec_id && (
                  <p className="mt-1 text-sm text-red-600">
                    {form.formState.errors.system_spec_id.message}
                  </p>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Certificate Type Selection */}
        <div className="card">
          <div className="card-body">
            <h2 className="text-xl font-medium text-gray-900 mb-4">Certificate Type</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {Object.entries(CERTIFICATE_TYPE_INFO).map(([type, info]) => (
                <label
                  key={type}
                  className={`relative flex cursor-pointer rounded-lg border p-4 focus:outline-none ${
                    selectedType === type
                      ? 'border-blue-600 ring-2 ring-blue-600 bg-blue-50'
                      : 'border-gray-300 hover:border-gray-400'
                  }`}
                >
                  <input
                    type="radio"
                    value={type}
                    {...form.register('certificate_type')}
                    className="sr-only"
                  />
                  <div className="flex-1">
                    <div className="flex items-center">
                      <div className="text-sm">
                        <p className="font-medium text-gray-900">{info.title}</p>
                        <p className="text-gray-500 mt-1">{info.description}</p>
                        <div className="mt-2">
                          <p className="text-xs font-medium text-gray-700">Use Cases:</p>
                          <ul className="text-xs text-gray-600 mt-1">
                            {info.useCases.map((useCase, index) => (
                              <li key={index}>• {useCase}</li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    </div>
                  </div>
                </label>
              ))}
            </div>
          </div>
        </div>

        {/* Generation Method Selection */}
        <div className="card">
          <div className="card-body">
            <h2 className="text-xl font-medium text-gray-900 mb-4">Generation Method</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {Object.entries(METHOD_INFO).map(([method, info]) => (
                <label
                  key={method}
                  className={`relative flex cursor-pointer rounded-lg border p-4 focus:outline-none ${
                    selectedMethod === method
                      ? 'border-blue-600 ring-2 ring-blue-600 bg-blue-50'
                      : 'border-gray-300 hover:border-gray-400'
                  }`}
                >
                  <input
                    type="radio"
                    value={method}
                    {...form.register('generation_method')}
                    className="sr-only"
                  />
                  <div className="flex-1">
                    <div className="flex items-center">
                      <div className="text-sm">
                        <p className="font-medium text-gray-900">{info.title}</p>
                        <p className="text-gray-500 mt-1">{info.description}</p>
                        <div className="mt-2 grid grid-cols-1 gap-1">
                          <div>
                            <span className="text-xs font-medium text-green-700">Pros: </span>
                            <span className="text-xs text-green-600">{info.pros.join(', ')}</span>
                          </div>
                          <div>
                            <span className="text-xs font-medium text-red-700">Cons: </span>
                            <span className="text-xs text-red-600">{info.cons.join(', ')}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </label>
              ))}
            </div>
          </div>
        </div>

        {/* LLM Configuration (only for LLM method) */}
        {isLLMMethod && (
          <div className="card">
            <div className="card-body">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-medium text-gray-900">LLM Configuration</h2>
                <button
                  type="button"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="btn btn-outline btn-sm"
                >
                  <InformationCircleIcon className="w-4 h-4 mr-1" />
                  {showAdvanced ? 'Hide' : 'Show'} Advanced
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Model
                  </label>
                  <select {...form.register('llm_config.model')} className="input">
                    <option value="claude-3-5-sonnet-20241022">Claude 3.5 Sonnet</option>
                    <option value="claude-3-haiku-20240307">Claude 3 Haiku</option>
                    <option value="claude-3-opus-20240229">Claude 3 Opus</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Generation Mode
                  </label>
                  <select {...form.register('llm_config.mode')} className="input">
                    {Object.entries(LLM_MODES).map(([mode, description]) => (
                      <option key={mode} value={mode}>
                        {description.split(' - ')[0]}
                      </option>
                    ))}
                  </select>
                  <p className="mt-1 text-xs text-gray-500">
                    {LLM_MODES[form.watch('llm_config.mode') as keyof typeof LLM_MODES]}
                  </p>
                </div>

                {showAdvanced && (
                  <>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Temperature
                      </label>
                      <input
                        type="number"
                        step="0.1"
                        min="0"
                        max="1"
                        {...form.register('llm_config.temperature', { valueAsNumber: true })}
                        className="input"
                      />
                      <p className="mt-1 text-xs text-gray-500">
                        Controls randomness (0 = deterministic, 1 = very random)
                      </p>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Max Tokens
                      </label>
                      <input
                        type="number"
                        min="100"
                        max="4096"
                        {...form.register('llm_config.max_tokens', { valueAsNumber: true })}
                        className="input"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Max Attempts
                      </label>
                      <input
                        type="number"
                        min="1"
                        max="10"
                        {...form.register('llm_config.max_attempts', { valueAsNumber: true })}
                        className="input"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Timeout (ms)
                      </label>
                      <input
                        type="number"
                        min="1000"
                        max="120000"
                        step="1000"
                        {...form.register('llm_config.timeout_ms', { valueAsNumber: true })}
                        className="input"
                      />
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Additional Options */}
        <div className="card">
          <div className="card-body">
            <h2 className="text-xl font-medium text-gray-900 mb-4">Additional Options</h2>
            
            <div className="flex items-center">
              <input
                type="checkbox"
                id="baseline_comparison"
                {...form.register('baseline_comparison')}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <label htmlFor="baseline_comparison" className="ml-2 block text-sm text-gray-900">
                Run baseline comparison
              </label>
            </div>
            <p className="mt-1 text-sm text-gray-500">
              Also generate certificates using traditional methods for comparison
            </p>
          </div>
        </div>

        {/* Generate Button */}
        <div className="flex justify-end space-x-4">
          <Link to="/certificates" className="btn btn-outline">
            Cancel
          </Link>
          <button
            type="submit"
            disabled={generateMutation.isPending || !systemSpecs?.length}
            className="btn btn-primary"
          >
            {generateMutation.isPending ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                Generating...
              </>
            ) : (
              <>
                <PlayIcon className="w-4 h-4 mr-2" />
                Generate Certificate
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
