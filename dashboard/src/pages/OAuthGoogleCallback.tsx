import { useEffect, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import apiClient from '@/api/client';
import { useAuth } from '@/contexts/AuthContext';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';

export const OAuthGoogleCallback: React.FC = () => {
  const location = useLocation();
  const { login } = useAuth();
  const [message, setMessage] = useState('Signing you in with Google...');

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const token = params.get('token');
    const error = params.get('error');

    if (error) {
      setMessage('Google sign-in failed. Please try again.');
      return;
    }

    if (!token) {
      setMessage('Missing login token. Please try again.');
      return;
    }

    localStorage.setItem('token', token);

    apiClient.get('/admin/auth/me')
      .then((response) => {
        login(token, response.data);
      })
      .catch(() => {
        setMessage('Unable to complete Google sign-in. Please try again.');
      });
  }, [location.search, login]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4 py-12 sm:px-6 lg:px-8">
      <Card className="w-full max-w-md text-center">
        <CardHeader className="space-y-1">
          <CardTitle className="text-2xl">Google Sign-In</CardTitle>
          <CardDescription>Finalizing your login</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">{message}</p>
        </CardContent>
        <CardFooter className="justify-center">
          <Button asChild variant="outline">
            <Link to="/login">Back to login</Link>
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
};
