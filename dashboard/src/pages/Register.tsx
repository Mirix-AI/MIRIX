import { useState } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import apiClient from '@/api/client';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import logoImg from '@/assets/logo.png';

const parseErrorDetail = (detail: unknown, fallback: string) => {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item: any) => {
        const field = Array.isArray(item?.loc) ? item.loc[item.loc.length - 1] : undefined;
        const msg = item?.msg || item?.message;
        if (!msg) return null;
        return field ? `${field}: ${msg}` : msg;
      })
      .filter(Boolean);
    if (messages.length) return messages.join(', ');
  }
  return fallback;
};

export const Register: React.FC = () => {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [verificationPending, setVerificationPending] = useState(false);
  const [registeredEmail, setRegisteredEmail] = useState('');
  const { login } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const response = await apiClient.post('/admin/auth/register', {
        name,
        email,
        password,
      });

      // Check if verification is required (202 response)
      if (response.status === 202) {
        setVerificationPending(true);
        setRegisteredEmail(email);
        return;
      }

      // Otherwise, auto-login (should not happen anymore, but kept for compatibility)
      const { access_token, client } = response.data;
      if (access_token && client) {
        login(access_token, client);
      }
    } catch (err) {
      if (axios.isAxiosError(err)) {
        const message = parseErrorDetail(err.response?.data?.detail, 'Failed to register');
        setError(message);
      } else {
        setError('Failed to register');
      }
    } finally {
      setLoading(false);
    }
  };

  // Show verification pending message if verification email was sent
  if (verificationPending) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background px-4 py-12 sm:px-6 lg:px-8">
        <Card className="w-full max-w-md">
          <CardHeader className="space-y-1 text-center">
            <div className="flex justify-center mb-4">
              <img src={logoImg} alt="Mirix" className="h-12 w-auto" />
            </div>
            <CardTitle className="text-2xl">Check your email</CardTitle>
            <CardDescription>
              We've sent a verification link to {registeredEmail}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="rounded-lg bg-blue-50 dark:bg-blue-950 p-4 text-sm text-blue-900 dark:text-blue-100">
              <p className="font-medium mb-2">📧 Verification email sent!</p>
              <p className="text-blue-700 dark:text-blue-300">
                Please check your inbox and click the verification link to activate your account.
                The link will expire in 24 hours.
              </p>
            </div>
            <div className="text-center text-sm text-muted-foreground">
              <p>Didn't receive the email?</p>
              <button 
                className="text-primary hover:underline font-medium mt-1"
                onClick={() => setVerificationPending(false)}
              >
                Try again
              </button>
            </div>
          </CardContent>
          <CardFooter className="justify-center text-sm text-muted-foreground">
            <Link to="/login" className="font-medium text-primary hover:underline">
              Back to sign in
            </Link>
          </CardFooter>
        </Card>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4 py-12 sm:px-6 lg:px-8">
      <Card className="w-full max-w-md">
        <CardHeader className="space-y-1 text-center">
          <div className="flex justify-center mb-4">
            <img src={logoImg} alt="Mirix" className="h-12 w-auto" />
          </div>
          <CardTitle className="text-2xl">Create an account</CardTitle>
          <CardDescription>
            Enter your email below to create your account
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="name">Name</Label>
              <Input
                id="name"
                placeholder="My Application"
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                placeholder="m@example.com"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
            {error && <div className="text-sm text-red-500">{error}</div>}
            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? 'Creating account...' : 'Create account'}
            </Button>
          </form>
        </CardContent>
        <CardFooter className="justify-center text-sm text-muted-foreground">
          Already have an account?{' '}
          <Link to="/login" className="ml-1 font-medium text-primary hover:underline">
            Sign in
          </Link>
        </CardFooter>
      </Card>
    </div>
  );
};
