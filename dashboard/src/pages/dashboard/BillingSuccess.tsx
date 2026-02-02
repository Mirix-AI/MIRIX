import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { CheckCircle2 } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { useAuth } from '@/contexts/AuthContext';

export const BillingSuccess: React.FC = () => {
  const navigate = useNavigate();
  const { refreshUser } = useAuth();

  useEffect(() => {
    refreshUser();
  }, [refreshUser]);

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <Card>
        <CardHeader className="space-y-2">
          <div className="flex items-center gap-2 text-emerald-600">
            <CheckCircle2 className="h-6 w-6" />
            <CardTitle>Payment successful</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Thanks! Your credits are being applied. You can head back to usage to
            confirm the updated balance.
          </p>
          <Button onClick={() => navigate('/dashboard/usage')}>
            Go to Usage
          </Button>
        </CardContent>
      </Card>
    </div>
  );
};
