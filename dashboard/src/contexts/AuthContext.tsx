import { createContext, useContext, useState, useEffect } from 'react';
import apiClient from '@/api/client';
import { useNavigate } from 'react-router-dom';

interface ClientUser {
  id: string;
  name: string;
  email: string;
  scope: string;
  status: string;
  default_user_id: string;  // Default user for memory operations
  created_at: string;
}

interface AuthContextType {
  user: ClientUser | null;
  token: string | null;
  login: (token: string, user: ClientUser) => void;
  logout: () => void;
  isAuthenticated: boolean;
  isLoading: boolean;
}

const AuthContext = createContext<AuthContextType | null>(null);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<ClientUser | null>(null);
  const [token, setToken] = useState<string | null>(localStorage.getItem('token'));
  const [isLoading, setIsLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const initAuth = async () => {
      const storedToken = localStorage.getItem('token');
      if (storedToken) {
        try {
          const response = await apiClient.get('/admin/auth/me');
          setUser(response.data);
          setToken(storedToken);
        } catch (error) {
          console.error('Failed to fetch user profile', error);
          logout();
        }
      }
      setIsLoading(false);
    };

    initAuth();
  }, []);

  const login = (newToken: string, newUser: ClientUser) => {
    localStorage.setItem('token', newToken);
    setToken(newToken);
    setUser(newUser);
    navigate('/dashboard');
  };

  const logout = () => {
    localStorage.removeItem('token');
    setToken(null);
    setUser(null);
    navigate('/login');
  };

  return (
    <AuthContext.Provider value={{ user, token, login, logout, isAuthenticated: !!user, isLoading }}>
      {children}
    </AuthContext.Provider>
  );
};
