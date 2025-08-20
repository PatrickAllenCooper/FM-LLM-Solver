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
  Conversation,
  ConversationMessage,
  StartConversationRequest,
  SendMessageRequest,
  PublishCertificateFromConversationRequest,
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

  // Re-run acceptance check with custom parameters
  async rerunAcceptance(candidateId: string, params: {
    sample_count?: number;
    sampling_method?: 'uniform' | 'sobol' | 'lhs' | 'adaptive';
    tolerance?: number;
    enable_stage_b?: boolean;
  }): Promise<{
    candidate_id: string;
    acceptance_result: any;
    parameters_used: any;
    rerun_timestamp: string;
  }> {
    const response: AxiosResponse<ApiResponse<any>> = await this.api.post(
      `/certificates/${candidateId}/rerun-acceptance`,
      params
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

  // ====== CONVERSATIONAL MODE METHODS ======

  // Start a new conversation
  async startConversation(data: StartConversationRequest): Promise<Conversation> {
    const response: AxiosResponse<ApiResponse<Conversation>> = 
      await this.api.post('/conversations', data);
    return response.data.data!;
  }

  // Get conversation details
  async getConversation(conversationId: string): Promise<Conversation> {
    const response: AxiosResponse<ApiResponse<Conversation>> = 
      await this.api.get(`/conversations/${conversationId}`);
    return response.data.data!;
  }

  // Get user's conversations
  async getConversations(page: number = 1, limit: number = 20, status?: string): Promise<PaginatedResponse<Conversation>> {
    const params: any = { page, limit };
    if (status) params.status = status;
    
    const response: AxiosResponse<ApiResponse<PaginatedResponse<Conversation>>> = 
      await this.api.get('/conversations', { params });
    return response.data.data!;
  }

  // Send message in conversation
  async sendMessage(conversationId: string, data: SendMessageRequest): Promise<ConversationMessage> {
    const response: AxiosResponse<ApiResponse<{ message: ConversationMessage }>> = 
      await this.api.post(`/conversations/${conversationId}/messages`, data);
    return response.data.data!.message;
  }

  // Publish certificate from conversation
  async publishCertificateFromConversation(data: PublishCertificateFromConversationRequest): Promise<{ candidate_id: string; conversation_id: string }> {
    const response: AxiosResponse<ApiResponse<{ candidate_id: string; conversation_id: string }>> = 
      await this.api.post('/conversations/publish', data);
    return response.data.data!;
  }

  // Abandon conversation
  async abandonConversation(conversationId: string): Promise<void> {
    await this.api.delete(`/conversations/${conversationId}`);
  }

  // Admin: Re-validate all existing accepted certificates with corrected mathematical logic
  async revalidateAcceptedCertificates(): Promise<{
    total_certificates: number;
    revalidated_count: number;
    status_changed_count: number;
    results: Array<{
      candidate_id: string;
      expression: string;
      previous_status: string;
      new_status: string;
      status_changed: boolean;
      violations_found: number;
    }>;
    summary: {
      message: string;
      mathematical_rationale: string;
    };
  }> {
    const response: AxiosResponse<ApiResponse<any>> = 
      await this.api.post('/admin/revalidate-certificates');
    return response.data.data!;
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
