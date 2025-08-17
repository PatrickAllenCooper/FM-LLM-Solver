import axios, { AxiosInstance, AxiosResponse } from 'axios';
import {
  ApiResponse,
  PaginatedResponse,
  User,
  AuthResponse,
  SystemSpec,
  Candidate,
  SystemSpecRequest,
  CertificateGenerationRequest,
  LoginForm,
  RegisterForm,
  ChangePasswordForm,
} from '@/types/api';

class ApiService {
  private api: AxiosInstance;

  constructor() {
    // FORCE REBUILD - Updated API URL for production
    this.api = axios.create({
      baseURL: 'https://fmgen-api-610214208348.us-central1.run.app/api',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add auth token to requests
    this.api.interceptors.request.use((config) => {
      const token = localStorage.getItem('auth_token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });

    // Handle auth errors
    this.api.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          localStorage.removeItem('auth_token');
          localStorage.removeItem('user');
          // Redirect to login if not already there
          if (window.location.pathname !== '/login') {
            window.location.href = '/login';
          }
        }
        return Promise.reject(error);
      }
    );
  }

  // Authentication
  async register(data: RegisterForm): Promise<AuthResponse> {
    const response: AxiosResponse<ApiResponse<AuthResponse>> = await this.api.post(
      '/auth/register',
      {
        email: data.email,
        password: data.password,
        role: data.role || 'researcher',
      }
    );
    return response.data.data!;
  }

  async login(data: LoginForm): Promise<AuthResponse> {
    const response: AxiosResponse<ApiResponse<AuthResponse>> = await this.api.post(
      '/auth/login',
      data
    );
    return response.data.data!;
  }

  async getCurrentUser(): Promise<User> {
    const response: AxiosResponse<ApiResponse<User>> = await this.api.get('/auth/me');
    return response.data.data!;
  }

  async changePassword(data: ChangePasswordForm): Promise<void> {
    await this.api.post('/auth/change-password', {
      currentPassword: data.currentPassword,
      newPassword: data.newPassword,
    });
  }

  // System Specifications
  async createSystemSpec(data: SystemSpecRequest): Promise<SystemSpec> {
    const response: AxiosResponse<ApiResponse<SystemSpec>> = await this.api.post(
      '/system-specs',
      data
    );
    return response.data.data!;
  }

  async getSystemSpecs(params?: {
    page?: number;
    limit?: number;
  }): Promise<PaginatedResponse<SystemSpec>> {
    const response: AxiosResponse<ApiResponse<PaginatedResponse<SystemSpec>>> = await this.api.get(
      '/system-specs',
      { params }
    );
    return response.data.data!;
  }

  async getSystemSpecById(id: string): Promise<SystemSpec> {
    const response: AxiosResponse<ApiResponse<SystemSpec>> = await this.api.get(
      `/system-specs/${id}`
    );
    return response.data.data!;
  }

  // Certificate Generation
  async generateCertificate(data: CertificateGenerationRequest): Promise<Candidate> {
    const response: AxiosResponse<ApiResponse<Candidate>> = await this.api.post(
      '/certificates/generate',
      data
    );
    return response.data.data!;
  }

  async getCandidates(params?: {
    page?: number;
    limit?: number;
    system_spec_id?: string;
    certificate_type?: string;
    verification_status?: string;
  }): Promise<PaginatedResponse<Candidate>> {
    const response: AxiosResponse<ApiResponse<PaginatedResponse<Candidate>>> = await this.api.get(
      '/certificates',
      { params }
    );
    return response.data.data!;
  }

  async getCandidateById(id: string): Promise<Candidate> {
    const response: AxiosResponse<ApiResponse<Candidate>> = await this.api.get(
      `/certificates/${id}`
    );
    return response.data.data!;
  }

  // Health
  async getHealth(): Promise<any> {
    const response = await this.api.get('/health');
    return response.data;
  }

  // Admin - Email Authorization
  async getAuthorizedEmails(): Promise<Array<{ email: string; added_by: string; added_at: string }>> {
    const response: AxiosResponse<ApiResponse<Array<{ email: string; added_by: string; added_at: string }>>> = 
      await this.api.get('/admin/authorized-emails');
    return response.data.data!;
  }

  async addAuthorizedEmail(email: string): Promise<void> {
    await this.api.post('/admin/authorized-emails', { email });
  }

  async removeAuthorizedEmail(email: string): Promise<void> {
    await this.api.delete('/admin/authorized-emails', { data: { email } });
  }

  async checkEmailAuthorization(email: string): Promise<{ email: string; isAuthorized: boolean }> {
    const response: AxiosResponse<ApiResponse<{ email: string; isAuthorized: boolean }>> = 
      await this.api.get(`/admin/check-email?email=${encodeURIComponent(email)}`);
    return response.data.data!;
  }

  // Generic methods for admin endpoints
  async get(path: string, params?: any): Promise<any> {
    const response = await this.api.get(path, { params });
    return response.data;
  }

  async post(path: string, data?: any): Promise<any> {
    const response = await this.api.post(path, data);
    return response.data;
  }

  async delete(path: string, config?: any): Promise<any> {
    const response = await this.api.delete(path, config);
    return response.data;
  }

  // Utils
  setAuthToken(token: string): void {
    localStorage.setItem('auth_token', token);
  }

  removeAuthToken(): void {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user');
  }

  getAuthToken(): string | null {
    return localStorage.getItem('auth_token');
  }
}

const api = new ApiService();
export { api };
export default api;
