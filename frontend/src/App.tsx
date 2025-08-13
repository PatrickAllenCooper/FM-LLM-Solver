import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';
import { useEffect } from 'react';

import { useAuthStore } from '@/stores/auth.store';
import Layout from '@/components/Layout';
import ProtectedRoute from '@/components/ProtectedRoute';

// Pages
import LoginPage from '@/pages/LoginPage';
import RegisterPage from '@/pages/RegisterPage';
import DashboardPage from '@/pages/DashboardPage';
import SystemSpecsPage from '@/pages/SystemSpecsPage';
import CreateSystemSpecPage from '@/pages/CreateSystemSpecPage';
import CertificatesPage from '@/pages/CertificatesPage';
import CertificateDetailsPage from '@/pages/CertificateDetailsPage';
import GenerateCertificatePage from '@/pages/GenerateCertificatePage';
import AboutPage from '@/pages/AboutPage';
import ExperimentsPage from '@/pages/ExperimentsPage';
import ProfilePage from '@/pages/ProfilePage';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

function App() {
  const { checkAuthStatus, loadUser, isAuthenticated } = useAuthStore();

  useEffect(() => {
    checkAuthStatus();
    if (isAuthenticated) {
      loadUser();
    }
  }, [checkAuthStatus, loadUser, isAuthenticated]);

  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="min-h-screen bg-gray-50">
          <Routes>
            {/* Public routes */}
            <Route path="/login" element={<LoginPage />} />
            <Route path="/register" element={<RegisterPage />} />
            
            {/* Protected routes */}
            <Route path="/" element={<ProtectedRoute><Layout /></ProtectedRoute>}>
              <Route index element={<Navigate to="/dashboard" replace />} />
              <Route path="dashboard" element={<DashboardPage />} />
              <Route path="system-specs" element={<SystemSpecsPage />} />
              <Route path="system-specs/create" element={<CreateSystemSpecPage />} />
              <Route path="certificates" element={<CertificatesPage />} />
              <Route path="certificates/generate" element={<GenerateCertificatePage />} />
              <Route path="certificates/:id" element={<CertificateDetailsPage />} />
              <Route path="about" element={<AboutPage />} />
              <Route path="experiments" element={<ExperimentsPage />} />
              <Route path="profile" element={<ProfilePage />} />
            </Route>
            
            {/* Catch-all route */}
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Routes>
          
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: '#363636',
                color: '#fff',
              },
              success: {
                duration: 3000,
                iconTheme: {
                  primary: '#10b981',
                  secondary: '#fff',
                },
              },
              error: {
                duration: 5000,
                iconTheme: {
                  primary: '#ef4444',
                  secondary: '#fff',
                },
              },
            }}
          />
        </div>
      </Router>
    </QueryClientProvider>
  );
}

export default App;
