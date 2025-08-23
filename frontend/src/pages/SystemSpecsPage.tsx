import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { 
  PlusIcon, 
  DocumentTextIcon, 
  EyeIcon,
  XCircleIcon,
  MagnifyingGlassIcon,
  FunnelIcon,
  BeakerIcon
} from '@heroicons/react/24/outline';
import { clsx } from 'clsx';
import { format } from 'date-fns';


import { api } from '@/services/api';

interface SystemSpecsFilters {
  system_type?: 'continuous' | 'discrete' | 'hybrid' | '';
  search?: string;
}

const TYPE_COLORS = {
  continuous: 'bg-blue-100 text-blue-800',
  discrete: 'bg-purple-100 text-purple-800',
  hybrid: 'bg-green-100 text-green-800',
};

export default function SystemSpecsPage() {
  const [page, setPage] = useState(1);
  const [filters, setFilters] = useState<SystemSpecsFilters>({});
  const [showFilters, setShowFilters] = useState(false);

  const { data: systemSpecsData, isLoading, error } = useQuery({
    queryKey: ['system-specs', page, filters],
    queryFn: async () => {
      const searchParams = {
        page,
        limit: 20,
        ...Object.fromEntries(
          Object.entries(filters).filter(([_, value]) => value !== '' && value !== undefined)
        ),
      };

      return await api.getSystemSpecs(searchParams);
    },
  });

  const handleFilterChange = (key: keyof SystemSpecsFilters, value: string) => {
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
        <div className="flex items-center justify-between">
          <div>
            <h1 className="academic-header text-3xl">System Specifications</h1>
            <p className="academic-body">
              Define and manage dynamical systems for formal verification.
            </p>
            <div className="mt-3 text-xs text-primary-700 font-medium">
              {systemSpecsData?.pagination.total || 0} specifications • CU Boulder Research
            </div>
          </div>
          <Link
            to="/system-specs/create"
            className="btn-primary shadow-lg"
          >
            <PlusIcon className="w-5 h-5 mr-2" />
            Create System Spec
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
              placeholder="Search system specifications..."
              value={filters.search || ''}
              onChange={(e) => handleFilterChange('search', e.target.value)}
              className="input pl-10"
            />
          </div>

          {showFilters && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  System Type
                </label>
                <select
                  value={filters.system_type || ''}
                  onChange={(e) => handleFilterChange('system_type', e.target.value)}
                  className="input"
                >
                  <option value="">All Types</option>
                  <option value="continuous">Continuous</option>
                  <option value="discrete">Discrete</option>
                  <option value="hybrid">Hybrid</option>
                </select>
              </div>
            </div>
          )}

          {Object.values(filters).some(v => v) && (
            <div className="flex items-center justify-between mt-4 pt-4 border-t border-gray-200">
              <p className="text-sm text-gray-600">
                {systemSpecsData?.pagination.total || 0} system specifications found
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
              <span className="ml-3 text-gray-600">Loading system specifications...</span>
            </div>
          </div>
        </div>
      ) : error ? (
        <div className="card">
          <div className="card-body">
            <div className="text-center py-8">
              <XCircleIcon className="mx-auto h-12 w-12 text-red-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">Error loading system specifications</h3>
              <p className="mt-1 text-sm text-gray-500">
                {error instanceof Error ? error.message : 'An unknown error occurred'}
              </p>
            </div>
          </div>
        </div>
      ) : !systemSpecsData?.data.length ? (
        <div className="card">
          <div className="card-body">
            <div className="text-center py-8">
              <DocumentTextIcon className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">No system specifications found</h3>
              <p className="mt-1 text-sm text-gray-500">
                Get started by creating your first system specification.
              </p>
              <div className="mt-6">
                <Link to="/system-specs/create" className="btn btn-primary">
                  <PlusIcon className="w-4 h-4 mr-2" />
                  Create System Spec
                </Link>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          {/* System Specs List */}
          <div className="card">
            <div className="overflow-hidden">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      System Name
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Type
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Dimension
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Created
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Generated by
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Version
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {systemSpecsData?.data.map((spec) => (
                    <tr key={spec.id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div>
                          <div className="text-sm font-medium text-gray-900">
                            {spec.name}
                          </div>
                          {spec.description && (
                            <div className="text-sm text-gray-500 truncate max-w-xs">
                              {spec.description}
                            </div>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={clsx(
                          'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium capitalize',
                          TYPE_COLORS[spec.system_type]
                        )}>
                          {spec.system_type}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {spec.dimension}D
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {format(new Date(spec.created_at), 'MMM d, yyyy')}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {spec.created_by_email || 'Unknown'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        v{spec.spec_version}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium space-x-3">
                        <Link
                          to={`/system-specs/${spec.id}`}
                          className="text-blue-600 hover:text-blue-900 inline-flex items-center"
                        >
                          <EyeIcon className="w-4 h-4 mr-1" />
                          View
                        </Link>
                        <Link
                          to={`/certificates/generate?system_spec_id=${spec.id}`}
                          className="text-green-600 hover:text-green-900 inline-flex items-center"
                        >
                          <BeakerIcon className="w-4 h-4 mr-1" />
                          Generate
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Pagination */}
          {systemSpecsData && systemSpecsData.pagination.total_pages > 1 && (
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
                      {Math.min(page * 20, systemSpecsData.pagination.total)}
                    </span>{' '}
                    of{' '}
                    <span className="font-medium">
                      {systemSpecsData.pagination.total}
                    </span>{' '}
                    results
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => setPage(Math.max(1, page - 1))}
                      disabled={!systemSpecsData.pagination.has_prev}
                      className="btn btn-outline btn-sm disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Previous
                    </button>
                    
                    <span className="text-sm text-gray-700">
                      Page {page} of {systemSpecsData.pagination.total_pages}
                    </span>
                    
                    <button
                      onClick={() => setPage(Math.min(systemSpecsData.pagination.total_pages, page + 1))}
                      disabled={!systemSpecsData.pagination.has_next}
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
