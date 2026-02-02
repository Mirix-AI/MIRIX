import { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import apiClient from '@/api/client';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import logoImg from '@/assets/logo.png';

export const VerifyEmail: React.FC = () => {
  const [searchParams] = useSearchParams();
  const [status, setStatus] = useState<'verifying' | 'success' | 'error'>('verifying');
  const [errorMessage, setErrorMessage] = useState('');
  const { login } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    const verifyEmail = async () => {
      const token = searchParams.get('token');

      if (!token) {
        setStatus('error');
        setErrorMessage('Invalid verification link. Token is missing.');
        return;
      }

      try {
        const response = await apiClient.post('/admin/auth/verify-email', {
          token,
        });

        const { access_token, client } = response.data;
        
        // Set status to success
        setStatus('success');
        
        // Login and redirect after a short delay
        setTimeout(() => {
          login(access_token, client);
          navigate('/');
        }, 2000);
      } catch (err: any) {
        setStatus('error');
        const message = err.response?.data?.detail || 'Failed to verify email. The link may have expired.';
        setErrorMessage(typeof message === 'string' ? message : 'Failed to verify email');
      }
    };

    verifyEmail();
  }, [searchParams, login, navigate]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4 py-12 sm:px-6 lg:px-8">
      <Card className="w-full max-w-md">
        <CardHeader className="space-y-1 text-center">
          <div className="flex justify-center mb-4">
            <img src={logoImg} alt="Mirix" className="h-12 w-auto" />
          </div>
          <CardTitle className="text-2xl">
            {status === 'verifying' && 'Verifying your email...'}
            {status === 'success' && 'Email verified!'}
            {status === 'error' && 'Verification failed'}
          </CardTitle>
          <CardDescription>
            {status === 'verifying' && 'Please wait while we verify your email address'}
            {status === 'success' && 'Your email has been successfully verified'}
            {status === 'error' && 'We couldn\'t verify your email address'}
          </CardDescription>
        </CardHeader>
        <CardContent className="text-center">
          {status === 'verifying' && (
            <div className="flex justify-center py-8">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
            </div>
          )}

          {status === 'success' && (
            <div className="space-y-4 py-4">
              <div className="flex justify-center">
                <div className="rounded-full bg-green-100 dark:bg-green-900 p-3">
                  <svg
                    className="h-12 w-12 text-green-600 dark:text-green-400"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M5 13l4 4L19 7"
                    />
                  </svg>
                </div>
              </div>
              <p className="text-sm text-muted-foreground">
                Redirecting to your dashboard...
              </p>
            </div>
          )}

          {status === 'error' && (
            <div className="space-y-4 py-4">
              <div className="flex justify-center">
                <div className="rounded-full bg-red-100 dark:bg-red-900 p-3">
                  <svg
                    className="h-12 w-12 text-red-600 dark:text-red-400"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                </div>
              </div>
              <div className="rounded-lg bg-red-50 dark:bg-red-950 p-4 text-sm text-red-900 dark:text-red-100">
                <p>{errorMessage}</p>
              </div>
              <div className="text-sm text-muted-foreground">
                <a href="/register" className="text-primary hover:underline font-medium">
                  Try registering again
                </a>
                {' or '}
                <a href="/login" className="text-primary hover:underline font-medium">
                  sign in
                </a>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

