import { Link } from 'react-router-dom';
import { PlusIcon, DocumentTextIcon } from '@heroicons/react/24/outline';

export default function SystemSpecsPage() {
  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">System Specifications</h1>
          <p className="mt-2 text-gray-600">
            Define and manage dynamical systems for formal verification.
          </p>
        </div>
        <Link
          to="/system-specs/create"
          className="btn-primary"
        >
          <PlusIcon className="w-5 h-5 mr-2" />
          Create System Spec
        </Link>
      </div>

      <div className="card">
        <div className="card-body text-center py-12">
          <DocumentTextIcon className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">No system specifications</h3>
          <p className="mt-1 text-sm text-gray-500">
            Get started by creating a new system specification.
          </p>
          <div className="mt-6">
            <Link
              to="/system-specs/create"
              className="btn-primary"
            >
              <PlusIcon className="w-5 h-5 mr-2" />
              Create System Spec
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
