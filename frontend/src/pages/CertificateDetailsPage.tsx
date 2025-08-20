import { useState } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  ArrowLeftIcon, 
  CheckCircleIcon, 
  XCircleIcon, 
  ClockIcon,
  DocumentDuplicateIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  ArrowDownTrayIcon,
  Cog8ToothIcon,
  ChartBarIcon,
  BeakerIcon,
  CalculatorIcon
} from '@heroicons/react/24/outline';
import { clsx } from 'clsx';
import { format } from 'date-fns';
import toast from 'react-hot-toast';


import { api } from '@/services/api';

const STATUS_COLORS = {
  pending: 'bg-yellow-100 text-yellow-800 border-yellow-200',
  accepted: 'bg-green-100 text-green-800 border-green-200',
  failed: 'bg-red-100 text-red-800 border-red-200',
  timeout: 'bg-gray-100 text-gray-800 border-gray-200',
};

const STATUS_ICONS = {
  pending: ClockIcon,
  accepted: CheckCircleIcon,
  failed: XCircleIcon,
  timeout: ClockIcon,
};

const METHOD_LABELS = {
  llm: 'LLM Generated',
  sos: 'SOS Baseline',
  sdp: 'SDP Baseline',
  quadratic_template: 'Quadratic Template',
  conversational: 'Conversational Mode',
};

const TYPE_LABELS = {
  lyapunov: 'Lyapunov Function',
  barrier: 'Barrier Certificate',
  inductive_invariant: 'Inductive Invariant',
};

export default function CertificateDetailsPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);
  const [showParameterControls, setShowParameterControls] = useState(false);
  
  // Parameter form state
  const [sampleCount, setSampleCount] = useState(10000);
  const [samplingMethod, setSamplingMethod] = useState<'uniform' | 'sobol' | 'lhs' | 'adaptive'>('uniform');
  const [tolerance, setTolerance] = useState(1e-6);
  const [enableStageB, setEnableStageB] = useState(false);

  const { data: certificate, isLoading, error } = useQuery({
    queryKey: ['certificate', id],
    queryFn: async () => {
      return await api.getCandidateById(id!);
    },
    enabled: !!id,
  });

  // Re-run acceptance check mutation
  const rerunMutation = useMutation({
    mutationFn: async (params: {
      sample_count: number;
      sampling_method: 'uniform' | 'sobol' | 'lhs' | 'adaptive';
      tolerance: number;
      enable_stage_b: boolean;
    }) => {
      return await api.rerunAcceptance(id!, params);
    },
    onSuccess: (data) => {
      toast.success('Acceptance check completed with new parameters!');
      // Invalidate and refetch the certificate to get updated results
      queryClient.invalidateQueries({ queryKey: ['certificate', id] });
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.error || 'Failed to re-run acceptance check');
    },
  });

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success('Copied to clipboard!');
  };

  const exportCertificate = () => {
    if (!certificate) return;
    
    const exportData = {
      id: certificate.id,
      certificate_type: certificate.certificate_type,
      expression: certificate.candidate_expression,
      generation_method: certificate.generation_method,
      acceptance_status: certificate.acceptance_status,
      margin: certificate.margin,
      system_spec_id: certificate.system_spec_id,
      system_name: certificate.system_name,
      created_at: certificate.created_at,
      accepted_at: certificate.accepted_at,
      llm_config: certificate.llm_config_json,
      candidate_data: certificate.candidate_json,
      counterexamples: certificate.counterexamples,
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `certificate-${certificate.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (isLoading) {
    return (
      <div className="space-y-8">
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span className="ml-3 text-gray-600">Loading certificate details...</span>
        </div>
      </div>
    );
  }

  if (error || !certificate) {
    return (
      <div className="space-y-8">
        <div className="text-center py-12">
          <ExclamationTriangleIcon className="mx-auto h-12 w-12 text-red-400" />
          <h3 className="mt-2 text-lg font-medium text-gray-900">Certificate not found</h3>
          <p className="mt-1 text-gray-500">
            The certificate you're looking for doesn't exist or has been deleted.
          </p>
          <div className="mt-6">
            <button
              onClick={() => navigate('/certificates')}
              className="btn btn-primary"
            >
              Back to Certificates
            </button>
          </div>
        </div>
      </div>
    );
  }

  const StatusIcon = STATUS_ICONS[certificate.acceptance_status];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex items-center space-x-4">
          <Link
            to="/certificates"
            className="btn btn-outline"
          >
            <ArrowLeftIcon className="w-4 h-4 mr-2" />
            Back to Certificates
          </Link>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">
              {TYPE_LABELS[certificate.certificate_type]}
            </h1>
            <p className="mt-1 text-gray-600">
              Certificate ID: {certificate.id}
            </p>
          </div>
        </div>
        
        <div className="flex items-center space-x-3">
          <button
            onClick={exportCertificate}
            className="btn btn-outline"
          >
            <ArrowDownTrayIcon className="w-4 h-4 mr-2" />
            Export
          </button>
          
          <span className={clsx(
            'inline-flex items-center px-3 py-1 rounded-full text-sm font-medium border',
            STATUS_COLORS[certificate.acceptance_status]
          )}>
            <StatusIcon className="w-4 h-4 mr-2" />
            {certificate.acceptance_status}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Content */}
        <div className="lg:col-span-2 space-y-6">
          {/* Certificate Expression */}
          <div className="card">
            <div className="card-body">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-medium text-gray-900">Certificate Expression</h2>
                <button
                  onClick={() => copyToClipboard(certificate.candidate_expression)}
                  className="btn btn-outline btn-sm"
                >
                  <DocumentDuplicateIcon className="w-4 h-4 mr-1" />
                  Copy
                </button>
              </div>
              
              <div className="bg-gray-50 rounded-lg p-4">
                <pre className="text-sm text-gray-800 whitespace-pre-wrap break-all">
                  {certificate.candidate_expression}
                </pre>
              </div>
            </div>
          </div>

          {/* Acceptance Results */}
          {certificate.acceptance_status !== 'pending' && (
            <div className="card">
              <div className="card-body">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-medium text-gray-900">Acceptance Results</h2>
                  <button
                    onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
                    className="btn btn-outline btn-sm"
                  >
                    <CalculatorIcon className="w-4 h-4 mr-1" />
                    {showTechnicalDetails ? 'Hide' : 'Show'} Technical Details
                  </button>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <dt className="text-sm font-medium text-gray-500">Status</dt>
                    <dd className="mt-1">
                      <span className={clsx(
                        'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium',
                        STATUS_COLORS[certificate.acceptance_status].replace('border-', 'bg-').replace('bg-bg-', 'bg-')
                      )}>
                        <StatusIcon className="w-3 h-3 mr-1" />
                        {certificate.acceptance_status}
                      </span>
                    </dd>
                  </div>

                  {certificate.margin !== null && certificate.margin !== undefined && (
                    <div>
                      <dt className="text-sm font-medium text-gray-500">Safety Margin</dt>
                      <dd className="mt-1 text-sm text-gray-900">
                        {certificate.margin.toFixed(8)}
                      </dd>
                    </div>
                  )}

                  {certificate.acceptance_duration_ms && (
                    <div>
                      <dt className="text-sm font-medium text-gray-500">Verification Time</dt>
                      <dd className="mt-1 text-sm text-gray-900">
                        {certificate.acceptance_duration_ms}ms
                      </dd>
                    </div>
                  )}

                  {certificate.accepted_at && (
                    <div>
                      <dt className="text-sm font-medium text-gray-500">Accepted At</dt>
                      <dd className="mt-1 text-sm text-gray-900">
                        {format(new Date(certificate.accepted_at), 'PPp')}
                      </dd>
                    </div>
                  )}
                </div>

                {/* Technical Details Section for Experimental Work - Show for ALL certificates */}
                {showTechnicalDetails && (
                  <div className="mt-8 space-y-6">
                    <div className="border-t border-gray-200 pt-6">
                      <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
                        <BeakerIcon className="w-5 h-5 mr-2 text-blue-600" />
                        Technical Acceptance Analysis
                      </h3>

                      {/* Conditions Checked */}
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                        <div className="bg-blue-50 rounded-xl p-4">
                          <h4 className="font-medium text-blue-900 mb-3">Mathematical Conditions Verified</h4>
                          <ul className="space-y-2">
                            {certificate.acceptance_result?.technical_details?.conditions_checked?.map((condition, idx) => (
                              <li key={idx} className="text-blue-800 text-sm flex items-start">
                                <CheckCircleIcon className="w-4 h-4 mr-2 mt-0.5 flex-shrink-0" />
                                {condition}
                              </li>
                            )) || (
                              <li className="text-blue-800 text-sm">
                                Technical details being generated...
                              </li>
                            )}
                          </ul>
                        </div>

                        <div className="bg-green-50 rounded-xl p-4">
                          <h4 className="font-medium text-green-900 mb-3">Numerical Method Details</h4>
                          <div className="space-y-2 text-sm">
                            <div className="flex justify-between">
                              <span className="text-green-800">Sampling Method:</span>
                              <span className="text-green-900 font-medium capitalize">
                                {certificate.acceptance_result?.technical_details?.sampling_method || 'N/A'}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-green-800">Sample Count:</span>
                              <span className="text-green-900 font-medium">
                                {certificate.acceptance_result?.technical_details?.sample_count?.toLocaleString() || 'N/A'}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-green-800">Tolerance:</span>
                              <span className="text-green-900 font-medium">
                                {certificate.acceptance_result?.technical_details?.numerical_parameters?.tolerance?.toExponential(2) || 'N/A'}
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Stage Results */}
                      <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl p-6 mb-6">
                        <h4 className="font-medium text-indigo-900 mb-4 flex items-center">
                          <Cog8ToothIcon className="w-5 h-5 mr-2" />
                          Two-Stage Acceptance Protocol Results
                        </h4>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          <div className="bg-white rounded-lg p-4">
                            <div className="flex items-center justify-between mb-2">
                              <span className="text-indigo-800 font-medium">Stage A (Numerical)</span>
                              {certificate.acceptance_result.technical_details.stage_results.stage_a_passed ? (
                                <CheckCircleIcon className="w-5 h-5 text-green-600" />
                              ) : (
                                <XCircleIcon className="w-5 h-5 text-red-600" />
                              )}
                            </div>
                            <p className="text-indigo-700 text-sm">
                              Numerical sampling and margin validation
                            </p>
                          </div>
                          <div className="bg-white rounded-lg p-4">
                            <div className="flex items-center justify-between mb-2">
                              <span className="text-indigo-800 font-medium">Stage B (Formal)</span>
                              {certificate.acceptance_result.technical_details.stage_results.stage_b_enabled ? (
                                certificate.acceptance_result.technical_details.stage_results.stage_b_passed ? (
                                  <CheckCircleIcon className="w-5 h-5 text-green-600" />
                                ) : (
                                  <XCircleIcon className="w-5 h-5 text-red-600" />
                                )
                              ) : (
                                <ClockIcon className="w-5 h-5 text-gray-400" />
                              )}
                            </div>
                            <p className="text-indigo-700 text-sm">
                              {certificate.acceptance_result.technical_details.stage_results.stage_b_enabled 
                                ? 'SOS/SMT formal verification'
                                : 'Not enabled (planned enhancement)'}
                            </p>
                          </div>
                        </div>
                      </div>

                      {/* Margin Breakdown */}
                      {certificate.acceptance_result.technical_details.margin_breakdown && (
                        <div className="bg-orange-50 rounded-xl p-6 mb-6">
                          <h4 className="font-medium text-orange-900 mb-4 flex items-center">
                            <ChartBarIcon className="w-5 h-5 mr-2" />
                            Detailed Margin Analysis
                          </h4>
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            {certificate.acceptance_result.technical_details.margin_breakdown.positivity_margin !== undefined && (
                              <div className="bg-white rounded-lg p-3">
                                <div className="text-orange-800 text-sm font-medium">Positivity Margin</div>
                                <div className="text-orange-900 text-lg font-bold">
                                  {certificate.acceptance_result.technical_details.margin_breakdown.positivity_margin.toFixed(6)}
                                </div>
                                <div className="text-orange-700 text-xs">V(x) {'>'} 0 margin</div>
                              </div>
                            )}
                            {certificate.acceptance_result.technical_details.margin_breakdown.decreasing_margin !== undefined && (
                              <div className="bg-white rounded-lg p-3">
                                <div className="text-orange-800 text-sm font-medium">Decreasing Margin</div>
                                <div className="text-orange-900 text-lg font-bold">
                                  {certificate.acceptance_result.technical_details.margin_breakdown.decreasing_margin.toFixed(6)}
                                </div>
                                <div className="text-orange-700 text-xs">dV/dt {'≤'} 0 margin</div>
                              </div>
                            )}
                            {certificate.acceptance_result.technical_details.margin_breakdown.separation_margin !== undefined && (
                              <div className="bg-white rounded-lg p-3">
                                <div className="text-orange-800 text-sm font-medium">Separation Margin</div>
                                <div className="text-orange-900 text-lg font-bold">
                                  {certificate.acceptance_result.technical_details.margin_breakdown.separation_margin.toFixed(6)}
                                </div>
                                <div className="text-orange-700 text-xs">Safe/unsafe separation</div>
                              </div>
                            )}
                          </div>
                        </div>
                      )}

                      {/* Violation Analysis */}
                      {certificate.acceptance_result.technical_details.violation_analysis.total_violations > 0 && (
                        <div className="bg-red-50 rounded-xl p-6 mb-6">
                          <h4 className="font-medium text-red-900 mb-4 flex items-center">
                            <ExclamationTriangleIcon className="w-5 h-5 mr-2" />
                            Violation Analysis ({certificate.acceptance_result.technical_details.violation_analysis.total_violations} violations found)
                          </h4>
                          <div className="space-y-3">
                            {certificate.acceptance_result.technical_details.violation_analysis.violation_points.slice(0, 5).map((violation, idx) => (
                              <div key={idx} className="bg-white rounded-lg p-3 border-l-4 border-red-300">
                                <div className="flex items-center justify-between mb-2">
                                  <span className="text-red-800 font-medium text-sm capitalize">
                                    {violation.condition.replace('_', ' ')} Violation
                                  </span>
                                  <span className={clsx(
                                    'px-2 py-1 rounded text-xs font-medium',
                                    violation.severity === 'severe' ? 'bg-red-100 text-red-800' :
                                    violation.severity === 'moderate' ? 'bg-yellow-100 text-yellow-800' :
                                    'bg-gray-100 text-gray-800'
                                  )}>
                                    {violation.severity}
                                  </span>
                                </div>
                                <div className="text-red-700 text-sm">
                                  <strong>Point:</strong> {Object.entries(violation.point).map(([varName, val]) => 
                                    `${varName}=${(val as number).toFixed(4)}`).join(', ')}
                                </div>
                                <div className="text-red-700 text-sm">
                                  <strong>Value:</strong> {violation.value.toFixed(6)}
                                </div>
                              </div>
                            ))}
                            {certificate.acceptance_result.technical_details.violation_analysis.total_violations > 5 && (
                              <div className="text-red-700 text-sm text-center">
                                ... and {certificate.acceptance_result.technical_details.violation_analysis.total_violations - 5} more violations
                              </div>
                            )}
                          </div>
                        </div>
                      )}

                      {/* Parameter Controls for Re-running */}
                      <div className="bg-gray-50 rounded-xl p-6">
                        <div className="flex items-center justify-between mb-4">
                          <h4 className="font-medium text-gray-900 flex items-center">
                            <Cog8ToothIcon className="w-5 h-5 mr-2" />
                            Experimental Parameter Controls
                          </h4>
                          <button
                            onClick={() => setShowParameterControls(!showParameterControls)}
                            className="btn btn-outline btn-sm"
                          >
                            {showParameterControls ? 'Hide' : 'Show'} Parameter Controls
                          </button>
                        </div>
                        
                        {showParameterControls && (
                          <div className="bg-white rounded-lg p-4 border border-gray-200">
                            <p className="text-gray-700 text-sm mb-4">
                              Adjust numerical parameters and re-run acceptance checking for experimental analysis.
                            </p>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                              <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                  Sample Count
                                </label>
                                <select 
                                  className="input input-sm"
                                  value={sampleCount}
                                  onChange={(e) => setSampleCount(Number(e.target.value))}
                                >
                                  <option value="1000">1,000 (Default)</option>
                                  <option value="5000">5,000 (Higher precision)</option>
                                  <option value="10000">10,000 (Maximum precision)</option>
                                </select>
                              </div>
                              <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                  Sampling Method
                                </label>
                                <select 
                                  className="input input-sm"
                                  value={samplingMethod}
                                  onChange={(e) => setSamplingMethod(e.target.value as 'uniform' | 'sobol' | 'lhs' | 'adaptive')}
                                >
                                  <option value="uniform">Uniform (Default)</option>
                                  <option value="sobol">Sobol (Low-discrepancy)</option>
                                  <option value="lhs">Latin Hypercube</option>
                                  <option value="adaptive">Adaptive Refinement</option>
                                </select>
                              </div>
                              <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                  Numerical Tolerance
                                </label>
                                <select 
                                  className="input input-sm"
                                  value={tolerance}
                                  onChange={(e) => setTolerance(Number(e.target.value))}
                                >
                                  <option value="1e-6">10⁻⁶ (Default)</option>
                                  <option value="1e-8">10⁻⁸ (Higher precision)</option>
                                  <option value="1e-10">10⁻¹⁰ (Maximum precision)</option>
                                </select>
                              </div>
                              <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                  Enable Stage B
                                </label>
                                <select 
                                  className="input input-sm"
                                  value={enableStageB.toString()}
                                  onChange={(e) => setEnableStageB(e.target.value === 'true')}
                                >
                                  <option value="false">Disabled (Stage A only)</option>
                                  <option value="true">Enabled (SOS/SMT verification)</option>
                                </select>
                              </div>
                            </div>
                            <button
                              className="btn btn-primary btn-sm"
                              disabled={rerunMutation.isPending}
                              onClick={() => {
                                rerunMutation.mutate({
                                  sample_count: sampleCount,
                                  sampling_method: samplingMethod,
                                  tolerance: tolerance,
                                  enable_stage_b: enableStageB,
                                });
                              }}
                            >
                              <BeakerIcon className="w-4 h-4 mr-1" />
                              {rerunMutation.isPending ? 'Re-running...' : 'Re-run Acceptance Check'}
                            </button>
                          </div>
                        )}
                        
                        {/* Current Parameters Display */}
                        <div className="mt-4 bg-white rounded-lg p-4 border border-gray-200">
                          <h5 className="font-medium text-gray-900 mb-3">Current Analysis Parameters</h5>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                            <div>
                              <div className="text-gray-500">Samples</div>
                              <div className="font-medium">
                                {certificate.acceptance_result.technical_details.sample_count.toLocaleString()}
                              </div>
                            </div>
                            <div>
                              <div className="text-gray-500">Method</div>
                              <div className="font-medium capitalize">
                                {certificate.acceptance_result.technical_details.sampling_method}
                              </div>
                            </div>
                            <div>
                              <div className="text-gray-500">Tolerance</div>
                              <div className="font-medium">
                                {certificate.acceptance_result.technical_details.numerical_parameters.tolerance.toExponential(0)}
                              </div>
                            </div>
                            <div>
                              <div className="text-gray-500">Violations</div>
                              <div className="font-medium">
                                {certificate.acceptance_result.technical_details.violation_analysis.total_violations}
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Parameter Controls for Re-running */}
                      <div className="bg-gray-50 rounded-xl p-6">
                        <div className="flex items-center justify-between mb-4">
                          <h4 className="font-medium text-gray-900 flex items-center">
                            <Cog8ToothIcon className="w-5 h-5 mr-2" />
                            Experimental Parameter Controls
                          </h4>
                          <button
                            onClick={() => setShowParameterControls(!showParameterControls)}
                            className="btn btn-outline btn-sm"
                          >
                            {showParameterControls ? 'Hide' : 'Show'} Parameter Controls
                          </button>
                        </div>
                        
                        {showParameterControls && (
                          <div className="bg-white rounded-lg p-4 border border-gray-200">
                            <p className="text-gray-700 text-sm mb-4">
                              Adjust numerical parameters and re-run acceptance checking for experimental analysis.
                            </p>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                              <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                  Sample Count
                                </label>
                                <select 
                                  className="input input-sm"
                                  value={sampleCount}
                                  onChange={(e) => setSampleCount(Number(e.target.value))}
                                >
                                  <option value="1000">1,000 (Default)</option>
                                  <option value="5000">5,000 (Higher precision)</option>
                                  <option value="10000">10,000 (Maximum precision)</option>
                                </select>
                              </div>
                              <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                  Sampling Method
                                </label>
                                <select 
                                  className="input input-sm"
                                  value={samplingMethod}
                                  onChange={(e) => setSamplingMethod(e.target.value as 'uniform' | 'sobol' | 'lhs' | 'adaptive')}
                                >
                                  <option value="uniform">Uniform (Default)</option>
                                  <option value="sobol">Sobol (Low-discrepancy)</option>
                                  <option value="lhs">Latin Hypercube</option>
                                  <option value="adaptive">Adaptive Refinement</option>
                                </select>
                              </div>
                              <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                  Numerical Tolerance
                                </label>
                                <select 
                                  className="input input-sm"
                                  value={tolerance}
                                  onChange={(e) => setTolerance(Number(e.target.value))}
                                >
                                  <option value="1e-6">10⁻⁶ (Default)</option>
                                  <option value="1e-8">10⁻⁸ (Higher precision)</option>
                                  <option value="1e-10">10⁻¹⁰ (Maximum precision)</option>
                                </select>
                              </div>
                              <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                  Enable Stage B
                                </label>
                                <select 
                                  className="input input-sm"
                                  value={enableStageB.toString()}
                                  onChange={(e) => setEnableStageB(e.target.value === 'true')}
                                >
                                  <option value="false">Disabled (Stage A only)</option>
                                  <option value="true">Enabled (SOS/SMT verification)</option>
                                </select>
                              </div>
                            </div>
                            <button
                              className="btn btn-primary btn-sm"
                              disabled={rerunMutation.isPending}
                              onClick={() => {
                                rerunMutation.mutate({
                                  sample_count: sampleCount,
                                  sampling_method: samplingMethod,
                                  tolerance: tolerance,
                                  enable_stage_b: enableStageB,
                                });
                              }}
                            >
                              <BeakerIcon className="w-4 h-4 mr-1" />
                              {rerunMutation.isPending ? 'Re-running...' : 'Re-run Acceptance Check'}
                            </button>
                          </div>
                        )}
                        
                        {/* Current Parameters Display */}
                        <div className="mt-4 bg-white rounded-lg p-4 border border-gray-200">
                          <h5 className="font-medium text-gray-900 mb-3">Current Analysis Parameters</h5>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                            <div>
                              <div className="text-gray-500">Samples</div>
                              <div className="font-medium">
                                {certificate.acceptance_result?.technical_details?.sample_count?.toLocaleString() || 'N/A'}
                              </div>
                            </div>
                            <div>
                              <div className="text-gray-500">Method</div>
                              <div className="font-medium capitalize">
                                {certificate.acceptance_result?.technical_details?.sampling_method || 'N/A'}
                              </div>
                            </div>
                            <div>
                              <div className="text-gray-500">Tolerance</div>
                              <div className="font-medium">
                                {certificate.acceptance_result?.technical_details?.numerical_parameters?.tolerance?.toExponential(0) || 'N/A'}
                              </div>
                            </div>
                            <div>
                              <div className="text-gray-500">Violations</div>
                              <div className="font-medium">
                                {certificate.acceptance_result?.technical_details?.violation_analysis?.total_violations || 0}
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Counterexamples */}
          {certificate.counterexamples && certificate.counterexamples.length > 0 && (
            <div className="card">
              <div className="card-body">
                <h2 className="text-lg font-medium text-gray-900 mb-4">Counterexamples</h2>
                
                <div className="space-y-4">
                  {certificate.counterexamples.map((counterexample, index) => (
                    <div key={counterexample.id} className="bg-red-50 border border-red-200 rounded-lg p-4">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <h3 className="text-sm font-medium text-red-900">
                            Counterexample {index + 1}
                          </h3>
                          <p className="mt-1 text-xs text-red-600">
                            {counterexample.context}
                          </p>
                        </div>
                      </div>
                      
                      <div className="mt-3">
                        <dt className="text-xs font-medium text-red-700">State Values:</dt>
                        <dd className="mt-1">
                          <pre className="text-xs text-red-800 bg-red-100 rounded p-2 overflow-auto">
                            {JSON.stringify(counterexample.x_json, null, 2)}
                          </pre>
                        </dd>
                      </div>

                      {counterexample.violation_metrics_json && (
                        <div className="mt-3">
                          <dt className="text-xs font-medium text-red-700">Violation Metrics:</dt>
                          <dd className="mt-1">
                            <pre className="text-xs text-red-800 bg-red-100 rounded p-2 overflow-auto">
                              {JSON.stringify(counterexample.violation_metrics_json, null, 2)}
                            </pre>
                          </dd>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* LLM Configuration (for LLM-generated certificates) */}
          {certificate.generation_method === 'llm' && certificate.llm_config_json && (
            <div className="card">
              <div className="card-body">
                <h2 className="text-lg font-medium text-gray-900 mb-4">LLM Configuration</h2>
                
                <div className="bg-blue-50 rounded-lg p-4">
                  <pre className="text-sm text-blue-800 whitespace-pre-wrap">
                    {JSON.stringify(certificate.llm_config_json, null, 2)}
                  </pre>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Basic Information */}
          <div className="card">
            <div className="card-body">
              <h2 className="text-lg font-medium text-gray-900 mb-4">Basic Information</h2>
              
              <dl className="space-y-3">
                <div>
                  <dt className="text-sm font-medium text-gray-500">Type</dt>
                  <dd className="mt-1 text-sm text-gray-900">
                    {TYPE_LABELS[certificate.certificate_type]}
                  </dd>
                </div>

                <div>
                  <dt className="text-sm font-medium text-gray-500">Generation Method</dt>
                  <dd className="mt-1">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                      {METHOD_LABELS[certificate.generation_method]}
                    </span>
                  </dd>
                </div>

                <div>
                  <dt className="text-sm font-medium text-gray-500">System</dt>
                  <dd className="mt-1 text-sm text-gray-900">
                    {certificate.system_name || 'Unknown System'}
                  </dd>
                </div>

                <div>
                  <dt className="text-sm font-medium text-gray-500">Created</dt>
                  <dd className="mt-1 text-sm text-gray-900">
                    {format(new Date(certificate.created_at), 'PPp')}
                  </dd>
                </div>

                {certificate.generation_duration_ms && (
                  <div>
                    <dt className="text-sm font-medium text-gray-500">Generation Time</dt>
                    <dd className="mt-1 text-sm text-gray-900">
                      {certificate.generation_duration_ms}ms
                    </dd>
                  </div>
                )}

                {certificate.llm_model && (
                  <div>
                    <dt className="text-sm font-medium text-gray-500">LLM Model</dt>
                    <dd className="mt-1 text-sm text-gray-900">
                      {certificate.llm_model}
                    </dd>
                  </div>
                )}

                {certificate.llm_mode && (
                  <div>
                    <dt className="text-sm font-medium text-gray-500">LLM Mode</dt>
                    <dd className="mt-1 text-sm text-gray-900 capitalize">
                      {certificate.llm_mode.replace('_', ' ')}
                    </dd>
                  </div>
                )}
              </dl>
            </div>
          </div>

          {/* Actions */}
          <div className="card">
            <div className="card-body">
              <h2 className="text-lg font-medium text-gray-900 mb-4">Actions</h2>
              
              <div className="space-y-3">
                <Link
                  to={`/system-specs/${certificate.system_spec_id}`}
                  className="btn btn-outline w-full justify-center"
                >
                  <InformationCircleIcon className="w-4 h-4 mr-2" />
                  View System Spec
                </Link>
                
                <button
                  onClick={() => copyToClipboard(certificate.candidate_expression)}
                  className="btn btn-outline w-full justify-center"
                >
                  <DocumentDuplicateIcon className="w-4 h-4 mr-2" />
                  Copy Expression
                </button>
                
                <button
                  onClick={exportCertificate}
                  className="btn btn-outline w-full justify-center"
                >
                  <ArrowDownTrayIcon className="w-4 h-4 mr-2" />
                  Export Certificate
                </button>
              </div>
            </div>
          </div>

          {/* Debug Info */}
          {certificate.candidate_json && (
            <div className="card">
              <div className="card-body">
                <h2 className="text-lg font-medium text-gray-900 mb-4">Raw Certificate Data</h2>
                
                <details className="group">
                  <summary className="cursor-pointer text-sm text-blue-600 hover:text-blue-700">
                    Show raw JSON data
                  </summary>
                  <div className="mt-3 bg-gray-50 rounded-lg p-3">
                    <pre className="text-xs text-gray-700 whitespace-pre-wrap break-all">
                      {JSON.stringify(certificate.candidate_json, null, 2)}
                    </pre>
                  </div>
                </details>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
