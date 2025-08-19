import { useParams, Link, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { 
  ArrowLeftIcon, 
  CheckCircleIcon, 
  XCircleIcon, 
  ClockIcon,
  DocumentDuplicateIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  ArrowDownTrayIcon
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
};

const TYPE_LABELS = {
  lyapunov: 'Lyapunov Function',
  barrier: 'Barrier Certificate',
  inductive_invariant: 'Inductive Invariant',
};

export default function CertificateDetailsPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const { data: certificate, isLoading, error } = useQuery({
    queryKey: ['certificate', id],
    queryFn: async () => {
      return await api.getCandidateById(id!);
    },
    enabled: !!id,
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
      verified_at: certificate.verified_at,
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

          {/* Verification Results */}
          {certificate.acceptance_status !== 'pending' && (
            <div className="card">
              <div className="card-body">
                <h2 className="text-lg font-medium text-gray-900 mb-4">Verification Results</h2>
                
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
