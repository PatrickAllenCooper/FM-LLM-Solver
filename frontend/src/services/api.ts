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
    this.api = axios.create({
      baseURL: import.meta.env.VITE_API_URL || '/api',
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

export const apiService = new ApiService();
export default apiService;
