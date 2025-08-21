import { useState, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { toast } from 'react-hot-toast';
import { 
  PlusIcon, 
  TrashIcon, 
  EnvelopeIcon,
  ShieldCheckIcon,
  UserIcon,
  CalendarIcon
} from '@heroicons/react/24/outline';
import { useAuthStore } from '@/stores/auth.store';
import { api } from '@/services/api';

// Types
interface AuthorizedEmail {
  email: string;
  added_by: string;
  added_at: string;
}

// Validation Schema
const addEmailSchema = z.object({
  email: z.string().email('Invalid email address'),
});

type AddEmailForm = z.infer<typeof addEmailSchema>;

export default function AdminEmailsPage() {
  const { user } = useAuthStore();
  const [authorizedEmails, setAuthorizedEmails] = useState<AuthorizedEmail[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isAdding, setIsAdding] = useState(false);
  const [removingEmail, setRemovingEmail] = useState<string | null>(null);

  const {
    register,
    handleSubmit,
    reset,
    formState: { errors },
  } = useForm<AddEmailForm>({
    resolver: zodResolver(addEmailSchema),
  });

  // Check if user is admin
  const isAdmin = user?.role === 'admin';

  // Load authorized emails
  const loadAuthorizedEmails = async () => {
    try {
      console.log('🔄 Loading authorized emails...');
      setIsLoading(true);
      const response = await api.get('/admin/authorized-emails');
      console.log('📧 API response:', response.data);
      console.log('📧 Setting emails:', response.data.data?.length || 0);
      
      const emails = response.data.data || [];
      setAuthorizedEmails(emails);
      
      console.log('✅ State updated with', emails.length, 'emails');
    } catch (error: any) {
      console.error('Failed to load authorized emails:', error);
      toast.error(error.response?.data?.error || 'Failed to load authorized emails');
      setAuthorizedEmails([]); // Ensure state is set even on error
    } finally {
      setIsLoading(false);
    }
  };

  // Add email
  const onAddEmail = async (data: AddEmailForm) => {
    try {
      setIsAdding(true);
      await api.post('/admin/authorized-emails', { email: data.email });
      toast.success(`${data.email} added to authorized list`);
      reset();
      await loadAuthorizedEmails();
    } catch (error: any) {
      console.error('Failed to add email:', error);
      toast.error(error.response?.data?.error || 'Failed to add email');
    } finally {
      setIsAdding(false);
    }
  };

  // Remove email
  const removeEmail = async (email: string) => {
    try {
      setRemovingEmail(email);
      await api.delete('/admin/authorized-emails', { data: { email } });
      toast.success(`${email} removed from authorized list`);
      await loadAuthorizedEmails();
    } catch (error: any) {
      console.error('Failed to remove email:', error);
      toast.error(error.response?.data?.error || 'Failed to remove email');
    } finally {
      setRemovingEmail(null);
    }
  };

  // Load data on mount with debugging
  useEffect(() => {
    console.log('🔍 useEffect triggered, isAdmin:', isAdmin);
    if (isAdmin) {
      console.log('✅ Loading authorized emails on mount...');
      loadAuthorizedEmails();
    }
  }, [isAdmin]);

  // Force refresh function for debugging
  const forceRefresh = () => {
    console.log('🔄 FORCE REFRESH TRIGGERED');
    loadAuthorizedEmails();
  };

  // Redirect if not admin
  if (!isAdmin) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-8 max-w-md w-full text-center">
          <ShieldCheckIcon className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <h1 className="text-xl font-semibold text-gray-900 mb-2">Access Denied</h1>
          <p className="text-gray-600 mb-4">
            You need admin permissions to access this page.
          </p>
          <button
            onClick={() => window.history.back()}
            className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
          >
            Go Back
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Email Authorization Management
          </h1>
          <p className="text-gray-600">
            Manage which email addresses are authorized to create accounts.
          </p>
        </div>

        {/* Add Email Form */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-8">
          <h2 className="text-lg font-medium text-gray-900 mb-4 flex items-center gap-2">
            <PlusIcon className="w-5 h-5" />
            Add Authorized Email
          </h2>
          
          <form onSubmit={handleSubmit(onAddEmail)} className="flex gap-4">
            <div className="flex-1">
              <input
                {...register('email')}
                type="email"
                placeholder="Enter email address"
                className={`w-full px-4 py-3 rounded-lg border transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent ${
                  errors.email 
                    ? 'border-red-300 bg-red-50' 
                    : 'border-gray-200 bg-white hover:border-gray-300'
                }`}
              />
              {errors.email && (
                <p className="text-sm text-red-600 mt-1">{errors.email.message}</p>
              )}
            </div>
            
            <button
              type="submit"
              disabled={isAdding}
              className="px-6 py-3 bg-primary-600 text-white rounded-lg font-medium hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isAdding ? 'Adding...' : 'Add Email'}
            </button>
          </form>
        </div>

        {/* Authorized Emails List */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200">
          <div className="p-6 border-b border-gray-200">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-medium text-gray-900 flex items-center gap-2">
                <EnvelopeIcon className="w-5 h-5" />
                Authorized Emails ({authorizedEmails.length})
              </h2>
              <button
                onClick={forceRefresh}
                disabled={isLoading}
                className="px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded hover:bg-blue-200 disabled:opacity-50"
              >
                {isLoading ? 'Loading...' : 'Refresh List'}
              </button>
            </div>
          </div>

          {isLoading ? (
            <div className="p-8 text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto"></div>
              <p className="text-gray-600 mt-2">Loading authorized emails...</p>
            </div>
          ) : authorizedEmails.length === 0 ? (
            <div className="p-8 text-center">
              <EnvelopeIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600">No authorized emails found.</p>
            </div>
          ) : (
            <div className="divide-y divide-gray-200">
              {authorizedEmails.map((emailData) => (
                <div
                  key={emailData.email}
                  className="p-6 flex items-center justify-between hover:bg-gray-50"
                >
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <EnvelopeIcon className="w-5 h-5 text-gray-400" />
                      <span className="font-medium text-gray-900">{emailData.email}</span>
                    </div>
                    <div className="flex items-center gap-4 text-sm text-gray-600">
                      <div className="flex items-center gap-1">
                        <UserIcon className="w-4 h-4" />
                        Added by: {emailData.added_by}
                      </div>
                      <div className="flex items-center gap-1">
                        <CalendarIcon className="w-4 h-4" />
                        {new Date(emailData.added_at).toLocaleDateString()}
                      </div>
                    </div>
                  </div>
                  
                  <button
                    onClick={() => removeEmail(emailData.email)}
                    disabled={removingEmail === emailData.email}
                    className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    title="Remove from authorized list"
                  >
                    {removingEmail === emailData.email ? (
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-red-600"></div>
                    ) : (
                      <TrashIcon className="w-5 h-5" />
                    )}
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Info Box */}
        <div className="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-6">
          <div className="flex items-start gap-3">
            <ShieldCheckIcon className="w-6 h-6 text-blue-600 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="font-medium text-blue-900 mb-2">How Email Authorization Works</h3>
              <div className="text-sm text-blue-800 space-y-1">
                <p>• Only emails in this list can create new accounts</p>
                <p>• Unauthorized users will see a message directing them to contact Patrick</p>
                <p>• You can add or remove emails at any time</p>
                <p>• Changes take effect immediately</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
