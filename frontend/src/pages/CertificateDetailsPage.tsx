import { useParams } from 'react-router-dom';

export default function CertificateDetailsPage() {
  const { id } = useParams<{ id: string }>();

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Certificate Details</h1>
        <p className="mt-2 text-gray-600">
          Detailed view of certificate {id}
        </p>
      </div>

      <div className="card">
        <div className="card-body">
          <p className="text-gray-500">Certificate details interface coming soon...</p>
        </div>
      </div>
    </div>
  );
}
