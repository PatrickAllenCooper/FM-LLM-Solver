import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import {
  DocumentTextIcon,
  CpuChipIcon,
  BeakerIcon,
  CheckCircleIcon,
  XCircleIcon,
} from '@heroicons/react/24/outline';
import { api } from '@/services/api';
import { useAuthStore } from '@/stores/auth.store';

export default function DashboardPage() {
  const { user } = useAuthStore();

  // Fetch dashboard data
  const { data: systemSpecs } = useQuery({
    queryKey: ['system-specs', { limit: 5 }],
    queryFn: () => api.getSystemSpecs({ limit: 5 }),
  });

  const { data: candidates } = useQuery({
    queryKey: ['candidates', { limit: 10 }],
    queryFn: () => api.getCandidates({ limit: 10 }),
  });

  // Calculate statistics
  const totalSpecs = systemSpecs?.pagination.total || 0;
  const totalCandidates = candidates?.pagination.total || 0;
  const verifiedCandidates = candidates?.data.filter(c => c.verification_status === 'verified').length || 0;
  const failedCandidates = candidates?.data.filter(c => c.verification_status === 'failed').length || 0;


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
      <div className="cu-gradient-light rounded-3xl p-8 border border-primary-200">
        <h1 className="academic-header text-3xl">Dashboard</h1>
        <p className="academic-body text-lg">
          Welcome back, {user?.email}. Here's an overview of your formal verification activities.
        </p>
        <div className="mt-4 text-xs text-primary-700 font-medium">
          University of Colorado Boulder • FM-LLM Solver
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => (
          <div key={stat.name} className="surface-elevated p-6 hover:elevation-4 transition-all duration-300 hover:-translate-y-1">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className={`w-12 h-12 ${stat.bgColor} rounded-2xl flex items-center justify-center shadow-md`}>
                  <stat.icon className={`w-6 h-6 ${stat.color}`} />
                </div>
              </div>
              <div className="ml-6 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-600 truncate">
                    {stat.name}
                  </dt>
                  <dd className="text-3xl font-bold text-gray-900 mt-1">
                    {stat.value}
                  </dd>
                </dl>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Quick Actions */}
      {user?.role !== 'viewer' && (
        <div className="card">
          <div className="card-header">
            <h2 className="academic-subheader">Quick Actions</h2>
            <p className="academic-body text-sm">Start your formal verification workflow</p>
          </div>
          <div className="card-body">
            <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
              <Link
                to="/system-specs/create"
                className="relative group surface-outlined p-6 transition-all duration-300 hover:elevation-3 hover:-translate-y-1 focus-within:ring-2 focus-within:ring-primary-500"
              >
                <div>
                  <span className="rounded-2xl inline-flex p-4 cu-gradient text-cu-black shadow-md group-hover:shadow-lg">
                    <DocumentTextIcon className="w-6 h-6" />
                  </span>
                </div>
                <div className="mt-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">
                    Create System Spec
                  </h3>
                  <p className="academic-body text-sm">
                    Define a new dynamical system for analysis
                  </p>
                </div>
              </Link>

              <Link
                to="/certificates"
                className="relative group surface-outlined p-6 transition-all duration-300 hover:elevation-3 hover:-translate-y-1 focus-within:ring-2 focus-within:ring-primary-500"
              >
                <div>
                  <span className="rounded-2xl inline-flex p-4 bg-gradient-to-br from-success-400 to-success-600 text-white shadow-md group-hover:shadow-lg">
                    <CpuChipIcon className="w-6 h-6" />
                  </span>
                </div>
                <div className="mt-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">
                    Generate Certificate
                  </h3>
                  <p className="academic-body text-sm">
                    Use LLM to generate Lyapunov or barrier functions
                  </p>
                </div>
              </Link>

              <Link
                to="/experiments"
                className="relative group surface-outlined p-6 transition-all duration-300 hover:elevation-3 hover:-translate-y-1 focus-within:ring-2 focus-within:ring-primary-500"
              >
                <div>
                  <span className="rounded-2xl inline-flex p-4 bg-gradient-to-br from-purple-400 to-purple-600 text-white shadow-md group-hover:shadow-lg">
                    <BeakerIcon className="w-6 h-6" />
                  </span>
                </div>
                <div className="mt-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">
                    Run Experiment
                  </h3>
                  <p className="academic-body text-sm">
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
            <h2 className="academic-subheader mb-0">Recent System Specs</h2>
            <Link
              to="/system-specs"
              className="text-sm font-medium text-primary-600 hover:text-primary-700 transition-colors duration-200"
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
            <h2 className="academic-subheader mb-0">Recent Certificates</h2>
            <Link
              to="/certificates"
              className="text-sm font-medium text-primary-600 hover:text-primary-700 transition-colors duration-200"
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
