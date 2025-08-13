import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import {
  DocumentTextIcon,
  CpuChipIcon,
  BeakerIcon,
  CheckCircleIcon,
  XCircleIcon,
  ClockIcon,
  PlusIcon,
} from '@heroicons/react/24/outline';
import { apiService } from '@/services/api';
import { useAuthStore } from '@/stores/auth.store';

export default function DashboardPage() {
  const { user } = useAuthStore();

  // Fetch dashboard data
  const { data: systemSpecs } = useQuery({
    queryKey: ['system-specs', { limit: 5 }],
    queryFn: () => apiService.getSystemSpecs({ limit: 5 }),
  });

  const { data: candidates } = useQuery({
    queryKey: ['candidates', { limit: 10 }],
    queryFn: () => apiService.getCandidates({ limit: 10 }),
  });

  // Calculate statistics
  const totalSpecs = systemSpecs?.pagination.total || 0;
  const totalCandidates = candidates?.pagination.total || 0;
  const verifiedCandidates = candidates?.data.filter(c => c.verification_status === 'verified').length || 0;
  const failedCandidates = candidates?.data.filter(c => c.verification_status === 'failed').length || 0;
  const pendingCandidates = candidates?.data.filter(c => c.verification_status === 'pending').length || 0;

  const stats = [
    {
      name: 'System Specifications',
      value: totalSpecs,
      icon: DocumentTextIcon,
      color: 'text-blue-600',
      bgColor: 'bg-blue-100',
    },
    {
      name: 'Total Certificates',
      value: totalCandidates,
      icon: CpuChipIcon,
      color: 'text-purple-600',
      bgColor: 'bg-purple-100',
    },
    {
      name: 'Verified',
      value: verifiedCandidates,
      icon: CheckCircleIcon,
      color: 'text-green-600',
      bgColor: 'bg-green-100',
    },
    {
      name: 'Failed',
      value: failedCandidates,
      icon: XCircleIcon,
      color: 'text-red-600',
      bgColor: 'bg-red-100',
    },
  ];

  const recentCandidates = candidates?.data.slice(0, 5) || [];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="mt-2 text-gray-600">
          Welcome back, {user?.email}. Here's an overview of your formal verification activities.
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => (
          <div key={stat.name} className="card">
            <div className="card-body">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <div className={`w-8 h-8 ${stat.bgColor} rounded-md flex items-center justify-center`}>
                    <stat.icon className={`w-5 h-5 ${stat.color}`} />
                  </div>
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">
                      {stat.name}
                    </dt>
                    <dd className="text-2xl font-bold text-gray-900">
                      {stat.value}
                    </dd>
                  </dl>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Quick Actions */}
      {user?.role !== 'viewer' && (
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-medium text-gray-900">Quick Actions</h2>
          </div>
          <div className="card-body">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
              <Link
                to="/system-specs/create"
                className="relative group bg-white p-6 focus-within:ring-2 focus-within:ring-inset focus-within:ring-primary-500 rounded-lg border-2 border-dashed border-gray-300 hover:border-gray-400"
              >
                <div>
                  <span className="rounded-lg inline-flex p-3 bg-primary-100 text-primary-600 group-hover:bg-primary-200">
                    <DocumentTextIcon className="w-6 h-6" />
                  </span>
                </div>
                <div className="mt-4">
                  <h3 className="text-lg font-medium text-gray-900">
                    Create System Spec
                  </h3>
                  <p className="mt-2 text-sm text-gray-500">
                    Define a new dynamical system for analysis
                  </p>
                </div>
              </Link>

              <Link
                to="/certificates"
                className="relative group bg-white p-6 focus-within:ring-2 focus-within:ring-inset focus-within:ring-primary-500 rounded-lg border-2 border-dashed border-gray-300 hover:border-gray-400"
              >
                <div>
                  <span className="rounded-lg inline-flex p-3 bg-green-100 text-green-600 group-hover:bg-green-200">
                    <CpuChipIcon className="w-6 h-6" />
                  </span>
                </div>
                <div className="mt-4">
                  <h3 className="text-lg font-medium text-gray-900">
                    Generate Certificate
                  </h3>
                  <p className="mt-2 text-sm text-gray-500">
                    Use LLM to generate Lyapunov or barrier functions
                  </p>
                </div>
              </Link>

              <Link
                to="/experiments"
                className="relative group bg-white p-6 focus-within:ring-2 focus-within:ring-inset focus-within:ring-primary-500 rounded-lg border-2 border-dashed border-gray-300 hover:border-gray-400"
              >
                <div>
                  <span className="rounded-lg inline-flex p-3 bg-purple-100 text-purple-600 group-hover:bg-purple-200">
                    <BeakerIcon className="w-6 h-6" />
                  </span>
                </div>
                <div className="mt-4">
                  <h3 className="text-lg font-medium text-gray-900">
                    Run Experiment
                  </h3>
                  <p className="mt-2 text-sm text-gray-500">
                    Compare LLM vs baseline methods
                  </p>
                </div>
              </Link>
            </div>
          </div>
        </div>
      )}

      {/* Recent Activity */}
      <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
        {/* Recent System Specs */}
        <div className="card">
          <div className="card-header flex items-center justify-between">
            <h2 className="text-lg font-medium text-gray-900">Recent System Specs</h2>
            <Link
              to="/system-specs"
              className="text-sm font-medium text-primary-600 hover:text-primary-500"
            >
              View all
            </Link>
          </div>
          <div className="card-body">
            {systemSpecs?.data.length ? (
              <ul className="space-y-3">
                {systemSpecs.data.map((spec) => (
                  <li key={spec.id} className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <DocumentTextIcon className="w-5 h-5 text-gray-400" />
                      <div>
                        <p className="text-sm font-medium text-gray-900">{spec.name}</p>
                        <p className="text-xs text-gray-500">
                          {spec.system_type} • {spec.dimension}D
                        </p>
                      </div>
                    </div>
                    <span className="text-xs text-gray-500">
                      {new Date(spec.created_at).toLocaleDateString()}
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-gray-500">No system specifications yet.</p>
            )}
          </div>
        </div>

        {/* Recent Certificates */}
        <div className="card">
          <div className="card-header flex items-center justify-between">
            <h2 className="text-lg font-medium text-gray-900">Recent Certificates</h2>
            <Link
              to="/certificates"
              className="text-sm font-medium text-primary-600 hover:text-primary-500"
            >
              View all
            </Link>
          </div>
          <div className="card-body">
            {recentCandidates.length ? (
              <ul className="space-y-3">
                {recentCandidates.map((candidate) => (
                  <li key={candidate.id} className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <CpuChipIcon className="w-5 h-5 text-gray-400" />
                      <div>
                        <p className="text-sm font-medium text-gray-900 capitalize">
                          {candidate.certificate_type}
                        </p>
                        <p className="text-xs text-gray-500">
                          {candidate.generation_method} • {candidate.system_name}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span
                        className={`status-badge ${
                          candidate.verification_status === 'verified'
                            ? 'status-verified'
                            : candidate.verification_status === 'failed'
                            ? 'status-failed'
                            : candidate.verification_status === 'pending'
                            ? 'status-pending'
                            : 'status-timeout'
                        }`}
                      >
                        {candidate.verification_status}
                      </span>
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-gray-500">No certificates generated yet.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
