import { useEffect, useState } from 'react';
import { Link, useSearchParams, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import logoImg from '@/assets/logo.png';
import { useAuth } from '@/contexts/AuthContext';
import apiClient from '@/api/client';

export const Login: React.FC = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { login } = useAuth();
  
  const [step, setStep] = useState<'email' | 'password'>('email');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null);

  useEffect(() => {
    // Check if redirected from email verification
    const verified = searchParams.get('verified');
    const token = searchParams.get('token');
    const errorParam = searchParams.get('error');
    const errorMessage = searchParams.get('message');

    if (verified === 'true' && token) {
      // Auto-login with the token from email verification
      setMessage({ type: 'success', text: 'Email verified successfully! Logging you in...' });
      
      setTimeout(() => {
        localStorage.setItem('token', token);
        window.location.href = '/';
      }, 1500);
    } else if (errorParam) {
      setMessage({ 
        type: 'error', 
        text: errorMessage ? decodeURIComponent(errorMessage) : 'Verification failed. Please try again.' 
      });
    }
  }, [searchParams]);

  const handleGoogleLogin = () => {
    const redirectUrl = `${window.location.origin}/oauth/google`;
    const oauthUrl = `/api/admin/auth/google/login?redirect=${encodeURIComponent(redirectUrl)}`;
    window.location.href = oauthUrl;
  };

  const handleEmailContinue = (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim()) {
      setError('Please enter your email');
      return;
    }
    setError('');
    setStep('password');
  };

  const handlePasswordSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!password) {
      setError('Please enter your password');
      return;
    }

    setError('');
    setLoading(true);

    try {
      const response = await apiClient.post('/admin/auth/login', {
        email,
        password,
      });

      const { access_token, client } = response.data;
      login(access_token, client);
      navigate('/');
    } catch (err: any) {
      if (err.response?.status === 403 && err.response?.data?.detail?.includes('verify')) {
        setError('Please verify your email before signing in. Check your inbox for the verification link.');
      } else if (err.response?.status === 401) {
        setError('Invalid email or password');
      } else {
        setError(err.response?.data?.detail || 'Failed to sign in');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleGoBack = () => {
    setStep('email');
    setPassword('');
    setError('');
  };

  // Email step
  if (step === 'email') {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background px-4 py-12 sm:px-6 lg:px-8">
        <Card className="w-full max-w-md">
          <CardHeader className="space-y-1 text-center">
            <div className="flex justify-center mb-4">
              <img src={logoImg} alt="Mirix" className="h-12 w-auto" />
            </div>
            <CardTitle className="text-2xl">Sign in to your account</CardTitle>
            <CardDescription>
              Use your Google account or email to sign in
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {message && (
              <div className={`p-3 rounded-lg text-sm ${
                message.type === 'success' 
                  ? 'bg-green-50 dark:bg-green-950 text-green-900 dark:text-green-100' 
                  : 'bg-red-50 dark:bg-red-950 text-red-900 dark:text-red-100'
              }`}>
                {message.text}
              </div>
            )}
            
            <Button type="button" variant="outline" className="w-full" onClick={handleGoogleLogin}>
              Continue with Google
            </Button>

            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t" />
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-background px-2 text-muted-foreground">
                  Or continue with email
                </span>
              </div>
            </div>

            <form onSubmit={handleEmailContinue} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="Your email address"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>

              {error && (
                <div className="text-sm text-red-600 dark:text-red-400">
                  {error}
                </div>
              )}

              <Button type="submit" className="w-full">
                Continue
              </Button>
            </form>
          </CardContent>
          <CardFooter className="flex justify-center">
            <p className="text-sm text-muted-foreground">
              Don't have an account?{' '}
              <Link to="/register" className="font-medium text-primary hover:underline">
                Sign up
              </Link>
            </p>
          </CardFooter>
        </Card>
      </div>
    );
  }

  // Password step
  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4 py-12 sm:px-6 lg:px-8">
      <Card className="w-full max-w-md">
        <CardHeader className="space-y-1 text-center">
          <div className="flex justify-center mb-4">
            <img src={logoImg} alt="Mirix" className="h-12 w-auto" />
          </div>
          <CardTitle className="text-2xl">Email sign-in</CardTitle>
          <CardDescription>
            Enter your password to continue
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handlePasswordSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="email-display">Email</Label>
              <div className="text-sm font-medium">{email}</div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="password">Password</Label>
                <Link 
                  to="/forgot-password" 
                  className="text-sm text-primary hover:underline"
                >
                  Forgot your password?
                </Link>
              </div>
              <div className="relative">
                <Input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  placeholder="Your password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                >
                  {showPassword ? (
                    <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                    </svg>
                  ) : (
                    <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                  )}
                </button>
              </div>
            </div>

            {error && (
              <div className="text-sm text-red-600 dark:text-red-400">
                {error}
              </div>
            )}

            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? 'Signing in...' : 'Sign in'}
            </Button>
          </form>

          <div className="mt-4">
            <Button 
              type="button" 
              variant="ghost" 
              className="w-full" 
              onClick={handleGoBack}
            >
              ← Go back
            </Button>
          </div>
        </CardContent>
        <CardFooter className="flex justify-center">
          <p className="text-sm text-muted-foreground">
            Don't have an account?{' '}
            <Link to="/register" className="font-medium text-primary hover:underline">
              Sign up
            </Link>
          </p>
        </CardFooter>
      </Card>
    </div>
  );
};
