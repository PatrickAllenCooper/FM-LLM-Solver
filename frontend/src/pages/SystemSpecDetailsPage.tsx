import { useParams, Link, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { 
  ArrowLeftIcon, 
  DocumentTextIcon,
  BeakerIcon,
  CubeIcon,
  AdjustmentsHorizontalIcon,
  MapIcon,
  ExclamationTriangleIcon,
  DocumentDuplicateIcon
} from '@heroicons/react/24/outline';
import { clsx } from 'clsx';
import { format } from 'date-fns';
import toast from 'react-hot-toast';

import { api } from '@/services/api';

const TYPE_COLORS = {
  continuous: 'bg-blue-100 text-blue-800 border-blue-200',
  discrete: 'bg-purple-100 text-purple-800 border-purple-200', 
  hybrid: 'bg-green-100 text-green-800 border-green-200',
};

const TYPE_ICONS = {
  continuous: AdjustmentsHorizontalIcon,
  discrete: CubeIcon,
  hybrid: MapIcon,
};

export default function SystemSpecDetailsPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const { data: systemSpec, isLoading, error } = useQuery({
    queryKey: ['system-spec', id],
    queryFn: async () => {
      if (!id) throw new Error('System specification ID is required');
      return await api.getSystemSpecById(id);
    },
    enabled: !!id,
  });

  const copyToClipboard = async (text: string, label: string) => {
    try {
      await navigator.clipboard.writeText(text);
      toast.success(`${label} copied to clipboard!`);
    } catch (error) {
      toast.error('Failed to copy to clipboard');
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-8">
        <div className="flex items-center space-x-4">
          <div className="h-8 w-8 bg-gray-200 rounded animate-pulse"></div>
          <div className="h-8 w-64 bg-gray-200 rounded animate-pulse"></div>
        </div>
        <div className="card">
          <div className="p-8">
            <div className="space-y-4">
              <div className="h-4 w-full bg-gray-200 rounded animate-pulse"></div>
              <div className="h-4 w-3/4 bg-gray-200 rounded animate-pulse"></div>
              <div className="h-4 w-1/2 bg-gray-200 rounded animate-pulse"></div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error || !systemSpec) {
    return (
      <div className="space-y-8">
        <div className="flex items-center space-x-4">
          <button
            onClick={() => navigate('/system-specs')}
            className="btn-secondary inline-flex items-center"
          >
            <ArrowLeftIcon className="w-4 h-4 mr-2" />
            Back to System Specs
          </button>
        </div>
        
        <div className="card">
          <div className="p-8 text-center">
            <ExclamationTriangleIcon className="w-16 h-16 text-red-500 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              System Specification Not Found
            </h3>
            <p className="text-gray-500 mb-6">
              The system specification you're looking for doesn't exist or you don't have permission to view it.
            </p>
            <Link to="/system-specs" className="btn-primary">
              Return to System Specs
            </Link>
          </div>
        </div>
      </div>
    );
  }

  const TypeIcon = TYPE_ICONS[systemSpec.system_type] || DocumentTextIcon;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <button
            onClick={() => navigate('/system-specs')}
            className="btn-secondary inline-flex items-center"
          >
            <ArrowLeftIcon className="w-4 h-4 mr-2" />
            Back to System Specs
          </button>
          
          <div className="flex items-center space-x-3">
            <TypeIcon className="w-8 h-8 text-gray-600" />
            <div>
              <h1 className="academic-header">{systemSpec.name}</h1>
              <div className="flex items-center space-x-2 mt-1">
                <span className={clsx(
                  'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium capitalize border',
                  TYPE_COLORS[systemSpec.system_type]
                )}>
                  {systemSpec.system_type}
                </span>
                <span className="text-sm text-gray-500">•</span>
                <span className="text-sm text-gray-500">{systemSpec.dimension}D System</span>
                <span className="text-sm text-gray-500">•</span>
                <span className="text-sm text-gray-500">v{systemSpec.spec_version}</span>
              </div>
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          <Link
            to={`/certificates/generate?system_spec_id=${systemSpec.id}`}
            className="btn-primary inline-flex items-center"
          >
            <BeakerIcon className="w-4 h-4 mr-2" />
            Generate Certificate
          </Link>
        </div>
      </div>

      {/* Basic Information */}
      <div className="card">
        <div className="card-header">
          <h2 className="academic-subheader mb-0">Basic Information</h2>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="form-label">Name</label>
              <div className="flex items-center space-x-2">
                <p className="text-gray-900">{systemSpec.name}</p>
                <button
                  onClick={() => copyToClipboard(systemSpec.name, 'System name')}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <DocumentDuplicateIcon className="w-4 h-4" />
                </button>
              </div>
            </div>

            <div>
              <label className="form-label">System Type</label>
              <div className="flex items-center space-x-2">
                <span className={clsx(
                  'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium capitalize border',
                  TYPE_COLORS[systemSpec.system_type]
                )}>
                  {systemSpec.system_type}
                </span>
              </div>
            </div>

            <div>
              <label className="form-label">Dimension</label>
              <p className="text-gray-900">{systemSpec.dimension}D</p>
            </div>

            <div>
              <label className="form-label">Created</label>
              <p className="text-gray-900">{format(new Date(systemSpec.created_at), 'MMM d, yyyy \'at\' h:mm a')}</p>
            </div>

            {systemSpec.description && (
              <div className="md:col-span-2">
                <label className="form-label">Description</label>
                <p className="text-gray-900">{systemSpec.description}</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* System Dynamics */}
      <div className="card">
        <div className="card-header">
          <h2 className="academic-subheader mb-0">System Dynamics</h2>
        </div>
        <div className="card-body">
          {systemSpec.dynamics_json && (
            <div className="space-y-6">
              {/* Variables */}
              {systemSpec.dynamics_json.variables && (
                <div>
                  <label className="form-label">Variables</label>
                  <div className="flex flex-wrap gap-2">
                    {systemSpec.dynamics_json.variables.map((variable: string, index: number) => (
                      <span
                        key={index}
                        className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800"
                      >
                        {variable}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Equations */}
              {systemSpec.dynamics_json.equations && (
                <div>
                  <label className="form-label">Differential Equations</label>
                  <div className="space-y-2">
                    {systemSpec.dynamics_json.equations.map((equation: string, index: number) => (
                      <div key={index} className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg font-mono text-sm">
                        <span className="text-gray-500">
                          d{systemSpec.dynamics_json.variables?.[index] || `x${index + 1}`}/dt =
                        </span>
                        <span className="text-gray-900">{equation}</span>
                        <button
                          onClick={() => copyToClipboard(equation, `Equation ${index + 1}`)}
                          className="text-gray-400 hover:text-gray-600"
                        >
                          <DocumentDuplicateIcon className="w-4 h-4" />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Domain */}
              {systemSpec.dynamics_json.domain && (
                <div>
                  <label className="form-label">Domain Constraints</label>
                  <div className="space-y-3">
                    {/* Variable Bounds */}
                    {systemSpec.dynamics_json.domain.bounds && (
                      <div>
                        <h4 className="text-sm font-medium text-gray-700 mb-2">Variable Bounds</h4>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                          {Object.entries(systemSpec.dynamics_json.domain.bounds).map(([variable, bounds]: [string, any]) => (
                            <div key={variable} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                              <span className="font-mono text-sm">{variable}:</span>
                              <span className="font-mono text-sm">
                                [{bounds.min || bounds[0]}, {bounds.max || bounds[1]}]
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* General Constraints */}
                    {systemSpec.dynamics_json.domain.constraints && (
                      <div>
                        <h4 className="text-sm font-medium text-gray-700 mb-2">General Constraints</h4>
                        <div className="space-y-1">
                          {systemSpec.dynamics_json.domain.constraints.map((constraint: string, index: number) => (
                            <div key={index} className="flex items-center space-x-2 p-2 bg-gray-50 rounded font-mono text-sm">
                              <span>{constraint}</span>
                              <button
                                onClick={() => copyToClipboard(constraint, `Constraint ${index + 1}`)}
                                className="text-gray-400 hover:text-gray-600"
                              >
                                <DocumentDuplicateIcon className="w-4 h-4" />
                              </button>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Sets and Constraints */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Initial Set */}
        {systemSpec.initial_set_json && (
          <div className="card">
            <div className="card-header">
              <h2 className="academic-subheader mb-0">Initial Set</h2>
            </div>
            <div className="card-body">
              <div className="space-y-3">
                <div>
                  <label className="form-label">Type</label>
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                    {systemSpec.initial_set_json.type}
                  </span>
                </div>
                <div>
                  <label className="form-label">Definition</label>
                  <pre className="bg-gray-50 p-3 rounded-lg text-sm font-mono overflow-x-auto">
                    {JSON.stringify(systemSpec.initial_set_json, null, 2)}
                  </pre>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Unsafe Set */}
        {systemSpec.unsafe_set_json && (
          <div className="card">
            <div className="card-header">
              <h2 className="academic-subheader mb-0">Unsafe Set</h2>
            </div>
            <div className="card-body">
              <div className="space-y-3">
                <div>
                  <label className="form-label">Type</label>
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                    {systemSpec.unsafe_set_json.type}
                  </span>
                </div>
                <div>
                  <label className="form-label">Definition</label>
                  <pre className="bg-gray-50 p-3 rounded-lg text-sm font-mono overflow-x-auto">
                    {JSON.stringify(systemSpec.unsafe_set_json, null, 2)}
                  </pre>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Constraints */}
      {systemSpec.constraints_json && Object.keys(systemSpec.constraints_json).length > 0 && (
        <div className="card">
          <div className="card-header">
            <h2 className="academic-subheader mb-0">Additional Constraints</h2>
          </div>
          <div className="card-body">
            <pre className="bg-gray-50 p-4 rounded-lg text-sm font-mono overflow-x-auto">
              {JSON.stringify(systemSpec.constraints_json, null, 2)}
            </pre>
          </div>
        </div>
      )}

      {/* Metadata */}
      <div className="card">
        <div className="card-header">
          <h2 className="academic-subheader mb-0">Metadata</h2>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <label className="form-label">System ID</label>
              <div className="flex items-center space-x-2">
                <code className="text-sm text-gray-600 bg-gray-100 px-2 py-1 rounded">
                  {systemSpec.id}
                </code>
                <button
                  onClick={() => copyToClipboard(systemSpec.id, 'System ID')}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <DocumentDuplicateIcon className="w-4 h-4" />
                </button>
              </div>
            </div>

            <div>
              <label className="form-label">Version</label>
              <p className="text-gray-900">v{systemSpec.spec_version}</p>
            </div>

            <div>
              <label className="form-label">Hash</label>
              <div className="flex items-center space-x-2">
                <code className="text-sm text-gray-600 bg-gray-100 px-2 py-1 rounded truncate max-w-32">
                  {(systemSpec as any).hash || 'N/A'}
                </code>
                <button
                  onClick={() => copyToClipboard((systemSpec as any).hash || '', 'System hash')}
                  className="text-gray-400 hover:text-gray-600"
                  disabled={!(systemSpec as any).hash}
                >
                  <DocumentDuplicateIcon className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex justify-center space-x-4 pt-4">
        <Link
          to={`/certificates/generate?system_spec_id=${systemSpec.id}`}
          className="btn-primary inline-flex items-center"
        >
          <BeakerIcon className="w-5 h-5 mr-2" />
          Generate Certificate
        </Link>
        <Link
          to="/system-specs"
          className="btn-secondary inline-flex items-center"
        >
          <DocumentTextIcon className="w-5 h-5 mr-2" />
          View All System Specs
        </Link>
      </div>
    </div>
  );
}
