import { create } from 'zustand';
import { User, AuthResponse, LoginForm, RegisterForm, ChangePasswordForm } from '@/types/api';
import { api } from '@/services/api';
import toast from 'react-hot-toast';

interface AuthStore {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  
  // Actions
  login: (data: LoginForm) => Promise<void>;
  register: (data: RegisterForm) => Promise<void>;
  logout: () => void;
  loadUser: () => Promise<void>;
  changePassword: (data: ChangePasswordForm) => Promise<void>;
  checkAuthStatus: () => void;
}

export const useAuthStore = create<AuthStore>((set) => ({
  user: null,
  isLoading: false,
  isAuthenticated: false,

  login: async (data: LoginForm) => {
    try {
      set({ isLoading: true });
      
      const authResponse: AuthResponse = await api.login(data);
      
      // Store token and user
      api.setAuthToken(authResponse.token);
      localStorage.setItem('user', JSON.stringify(authResponse.user));
      
      set({
        user: authResponse.user as User,
        isAuthenticated: true,
        isLoading: false,
      });
      
      toast.success('Login successful!');
    } catch (error: any) {
      set({ isLoading: false });
      const message = error.response?.data?.error || 'Login failed';
      toast.error(message);
      throw error;
    }
  },

  register: async (data: RegisterForm) => {
    try {
      set({ isLoading: true });
      
      const authResponse: AuthResponse = await api.register(data);
      
      // Store token and user
      api.setAuthToken(authResponse.token);
      localStorage.setItem('user', JSON.stringify(authResponse.user));
      
      set({
        user: authResponse.user as User,
        isAuthenticated: true,
        isLoading: false,
      });
      
      toast.success('Registration successful!');
    } catch (error: any) {
      set({ isLoading: false });
      const message = error.response?.data?.error || 'Registration failed';
      toast.error(message);
      throw error;
    }
  },

  logout: () => {
    api.removeAuthToken();
    set({
      user: null,
      isAuthenticated: false,
    });
    toast.success('Logged out successfully');
  },

  loadUser: async () => {
    try {
      const token = api.getAuthToken();
      if (!token) {
        set({ isAuthenticated: false, user: null });
        return;
      }

      set({ isLoading: true });
      
      const user = await api.getCurrentUser();
      
      set({
        user,
        isAuthenticated: true,
        isLoading: false,
      });
    } catch (error) {
      // Token is invalid, clear it
      api.removeAuthToken();
      set({
        user: null,
        isAuthenticated: false,
        isLoading: false,
      });
    }
  },

  changePassword: async (data: ChangePasswordForm) => {
    try {
      set({ isLoading: true });
      
      await api.changePassword(data);
      
      set({ isLoading: false });
      toast.success('Password changed successfully!');
    } catch (error: any) {
      set({ isLoading: false });
      const message = error.response?.data?.error || 'Failed to change password';
      toast.error(message);
      throw error;
    }
  },

  checkAuthStatus: () => {
    const token = api.getAuthToken();
    const storedUser = localStorage.getItem('user');
    
    if (token && storedUser) {
      try {
        const user = JSON.parse(storedUser);
        set({
          user,
          isAuthenticated: true,
        });
      } catch (error) {
        // Invalid stored user, clear everything
        api.removeAuthToken();
        set({
          user: null,
          isAuthenticated: false,
        });
      }
    } else {
      set({
        user: null,
        isAuthenticated: false,
      });
    }
  },
}));
