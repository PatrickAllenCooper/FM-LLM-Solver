import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { 
  EyeIcon, 
  BeakerIcon, 
  CheckCircleIcon, 
  XCircleIcon, 
  ClockIcon,
  FunnelIcon,
  PlusIcon,
  MagnifyingGlassIcon
} from '@heroicons/react/24/outline';
import { clsx } from 'clsx';
import { format } from 'date-fns';


import { api } from '@/services/api';

interface CertificatesFilters {
  certificate_type?: 'lyapunov' | 'barrier' | 'inductive_invariant' | '';
  generation_method?: 'llm' | 'sos' | 'sdp' | 'quadratic_template' | 'conversational' | '';
  acceptance_status?: 'pending' | 'accepted' | 'failed' | 'timeout' | '';
  search?: string;
}

const STATUS_COLORS = {
  pending: 'bg-yellow-100 text-yellow-800',
  accepted: 'bg-green-100 text-green-800',
  failed: 'bg-red-100 text-red-800',
  timeout: 'bg-gray-100 text-gray-800',
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

export default function CertificatesPage() {
  const [page, setPage] = useState(1);
  const [filters, setFilters] = useState<CertificatesFilters>({});
  const [showFilters, setShowFilters] = useState(false);

  const { data: certificatesData, isLoading, error } = useQuery({
    queryKey: ['certificates', page, filters],
    queryFn: async () => {
      const searchParams = {
        page,
        limit: 20,
        ...Object.fromEntries(
          Object.entries(filters).filter(([_, value]) => value !== '' && value !== undefined)
        ),
      };

      return await api.getCandidates(searchParams);
    },
  });

  const handleFilterChange = (key: keyof CertificatesFilters, value: string) => {
    setFilters(prev => ({ ...prev, [key]: value }));
    setPage(1); // Reset to first page when filtering
  };

  const clearFilters = () => {
    setFilters({});
    setPage(1);
  };

  return (
    <div className="space-y-8">
      <div className="cu-gradient-light rounded-3xl p-8 border border-primary-200">
        <div className="flex justify-between items-start">
          <div>
            <h1 className="academic-header text-3xl">Certificates</h1>
            <p className="academic-body">
              View and manage generated Lyapunov functions and barrier certificates.
            </p>
            <div className="mt-3 text-xs text-primary-700 font-medium">
              {certificatesData?.pagination.total || 0} certificates • CU Boulder Research
            </div>
          </div>
        
          <Link
            to="/certificates/generate"
            className="btn-primary shadow-lg"
          >
            <PlusIcon className="w-4 h-4 mr-2" />
            Generate Certificate
          </Link>
        </div>
      </div>

      {/* Filters */}
      <div className="card">
        <div className="card-body">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-medium text-gray-900">Filters & Search</h2>
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="btn btn-outline btn-sm"
            >
              <FunnelIcon className="w-4 h-4 mr-1" />
              {showFilters ? 'Hide' : 'Show'} Filters
            </button>
          </div>

          {/* Search */}
          <div className="relative mb-4">
            <MagnifyingGlassIcon className="w-5 h-5 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
            <input
              type="text"
              placeholder="Search certificates..."
              value={filters.search || ''}
              onChange={(e) => handleFilterChange('search', e.target.value)}
              className="input pl-10"
            />
          </div>

          {showFilters && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Certificate Type
                </label>
                <select
                  value={filters.certificate_type || ''}
                  onChange={(e) => handleFilterChange('certificate_type', e.target.value)}
                  className="input"
                >
                  <option value="">All Types</option>
                  <option value="lyapunov">Lyapunov Function</option>
                  <option value="barrier">Barrier Certificate</option>
                  <option value="inductive_invariant">Inductive Invariant</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Generation Method
                </label>
                <select
                  value={filters.generation_method || ''}
                  onChange={(e) => handleFilterChange('generation_method', e.target.value)}
                  className="input"
                >
                  <option value="">All Methods</option>
                  <option value="llm">LLM Generated</option>
                  <option value="sos">SOS Baseline</option>
                  <option value="sdp">SDP Baseline</option>
                  <option value="quadratic_template">Quadratic Template</option>
                  <option value="conversational">Conversational Mode</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Acceptance Status
                </label>
                <select
                  value={filters.acceptance_status || ''}
                  onChange={(e) => handleFilterChange('acceptance_status', e.target.value)}
                  className="input"
                >
                  <option value="">All Statuses</option>
                  <option value="pending">Pending</option>
                  <option value="accepted">Accepted</option>
                  <option value="failed">Failed</option>
                  <option value="timeout">Timeout</option>
                </select>
              </div>
            </div>
          )}

          {Object.values(filters).some(v => v) && (
            <div className="flex items-center justify-between mt-4 pt-4 border-t border-gray-200">
              <p className="text-sm text-gray-600">
                {certificatesData?.pagination.total || 0} certificates found
              </p>
              <button
                onClick={clearFilters}
                className="text-sm text-blue-600 hover:text-blue-700"
              >
                Clear all filters
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Results */}
      {isLoading ? (
        <div className="card">
          <div className="card-body">
            <div className="flex items-center justify-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              <span className="ml-3 text-gray-600">Loading certificates...</span>
            </div>
          </div>
        </div>
      ) : error ? (
        <div className="card">
          <div className="card-body">
            <div className="text-center py-8">
              <XCircleIcon className="mx-auto h-12 w-12 text-red-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">Error loading certificates</h3>
              <p className="mt-1 text-sm text-gray-500">
                {error instanceof Error ? error.message : 'An unknown error occurred'}
              </p>
            </div>
          </div>
        </div>
      ) : !certificatesData?.data.length ? (
        <div className="card">
          <div className="card-body">
            <div className="text-center py-8">
              <BeakerIcon className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">No certificates found</h3>
              <p className="mt-1 text-sm text-gray-500">
                Get started by generating your first certificate.
              </p>
              <div className="mt-6">
                <Link to="/certificates/generate" className="btn btn-primary">
                  <PlusIcon className="w-4 h-4 mr-2" />
                  Generate Certificate
                </Link>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          {/* Certificates List */}
          <div className="card">
            <div className="overflow-hidden">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Certificate
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      System
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Method
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Margin
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Created
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Generated by
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {certificatesData?.data.map((certificate) => {
                    const StatusIcon = STATUS_ICONS[certificate.acceptance_status];
                    
                    return (
                      <tr key={certificate.id} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div>
                            <div className="text-sm font-medium text-gray-900">
                              {TYPE_LABELS[certificate.certificate_type]}
                            </div>
                            <div className="text-sm text-gray-500 truncate max-w-xs">
                              {certificate.candidate_expression || 'Expression not available'}
                            </div>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm text-gray-900">
                            {certificate.system_name || 'Unknown System'}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                            {METHOD_LABELS[certificate.generation_method]}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={clsx(
                            'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium',
                            STATUS_COLORS[certificate.acceptance_status]
                          )}>
                            <StatusIcon className="w-3 h-3 mr-1" />
                            {certificate.acceptance_status}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {certificate.margin !== null && certificate.margin !== undefined
                            ? certificate.margin.toFixed(6)
                            : '—'
                          }
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {format(new Date(certificate.created_at), 'MMM d, yyyy')}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {certificate.created_by_email || 'Unknown'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                          <Link
                            to={`/certificates/${certificate.id}`}
                            className="text-blue-600 hover:text-blue-900 inline-flex items-center"
                          >
                            <EyeIcon className="w-4 h-4 mr-1" />
                            View
                          </Link>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Pagination */}
          {certificatesData && certificatesData.pagination.total_pages > 1 && (
            <div className="card">
              <div className="card-body">
                <div className="flex items-center justify-between">
                  <div className="text-sm text-gray-700">
                    Showing{' '}
                    <span className="font-medium">
                      {(page - 1) * 20 + 1}
                    </span>{' '}
                    to{' '}
                    <span className="font-medium">
                      {Math.min(page * 20, certificatesData.pagination.total)}
                    </span>{' '}
                    of{' '}
                    <span className="font-medium">
                      {certificatesData.pagination.total}
                    </span>{' '}
                    results
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => setPage(Math.max(1, page - 1))}
                      disabled={!certificatesData.pagination.has_prev}
                      className="btn btn-outline btn-sm disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Previous
                    </button>
                    
                    <span className="text-sm text-gray-700">
                      Page {page} of {certificatesData.pagination.total_pages}
                    </span>
                    
                    <button
                      onClick={() => setPage(Math.min(certificatesData.pagination.total_pages, page + 1))}
                      disabled={!certificatesData.pagination.has_next}
                      className="btn btn-outline btn-sm disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Next
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
