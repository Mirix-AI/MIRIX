import { useNavigate } from 'react-router-dom';
import { XCircle } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

export const BillingCancel: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <Card>
        <CardHeader className="space-y-2">
          <div className="flex items-center gap-2 text-red-500">
            <XCircle className="h-6 w-6" />
            <CardTitle>Payment canceled</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            No charges were made. If you want to try again, head back to usage.
          </p>
          <Button variant="outline" onClick={() => navigate('/dashboard/usage')}>
            Back to Usage
          </Button>
        </CardContent>
      </Card>
    </div>
  );
};
