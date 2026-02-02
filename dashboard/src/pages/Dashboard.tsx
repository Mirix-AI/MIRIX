import { Routes, Route, NavLink, Navigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { LayoutDashboard, Key, Brain, LogOut, CreditCard, Activity } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Overview } from './dashboard/Overview';
import { ApiKeys } from './dashboard/ApiKeys';
import { Memories } from './dashboard/Memories';
import { MemoryTraces } from './dashboard/MemoryTraces';
import { Usage } from './dashboard/Usage';
import { BillingSuccess } from './dashboard/BillingSuccess';
import { BillingCancel } from './dashboard/BillingCancel';
import { WorkspaceProvider } from '@/contexts/WorkspaceContext';
import { UserSwitcher } from '@/components/UserSwitcher';
import logoImg from '@/assets/logo.png';

export const Dashboard: React.FC = () => {
  const { user, logout } = useAuth();

  const navigation = [
    { name: 'Overview', href: '/dashboard', icon: LayoutDashboard, end: true },
    { name: 'API Keys', href: '/dashboard/api-keys', icon: Key },
    { name: 'Memories', href: '/dashboard/memories', icon: Brain },
    { name: 'Memory Traces', href: '/dashboard/memory-traces', icon: Activity },
    { name: 'Usage', href: '/dashboard/usage', icon: CreditCard },
  ];

  return (
    <WorkspaceProvider>
      <div className="flex h-screen overflow-hidden bg-background">
        {/* Sidebar */}
        <div className="hidden w-64 flex-col border-r bg-card md:flex">
          <div className="flex h-16 items-center px-6 border-b">
            <img src={logoImg} alt="Mirix" className="h-10 w-auto" />
          </div>
          
          <div className="px-4 py-4">
            <UserSwitcher />
          </div>

          <div className="flex-1 flex flex-col justify-between px-4 pb-4">
            <nav className="space-y-1">
              {navigation.map((item) => (
                <NavLink
                  key={item.name}
                  to={item.href}
                  end={item.end}
                  className={(navData) => `
                    flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors
                    ${navData.isActive 
                      ? 'bg-primary/10 text-primary' 
                      : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'}
                  `}
                >
                  <item.icon className="mr-3 h-4 w-4" />
                  {item.name}
                </NavLink>
              ))}
            </nav>
            
            <div className="space-y-4 mt-auto pt-4 border-t">
              <div className="px-1">
                <div className="text-xs font-semibold text-muted-foreground mb-2">ACCOUNT</div>
                <div className="flex items-center text-sm mb-4 px-2">
                  <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center text-primary font-bold mr-2">
                    {user?.name.charAt(0).toUpperCase()}
                  </div>
                  <div className="overflow-hidden">
                    <div className="font-medium truncate">{user?.name}</div>
                    <div className="text-xs text-muted-foreground truncate">{user?.email}</div>
                  </div>
                </div>
                <Button variant="outline" className="w-full justify-start" onClick={logout}>
                  <LogOut className="mr-2 h-4 w-4" />
                  Sign out
                </Button>
              </div>
            </div>
          </div>
        </div>

        {/* Main content */}
        <main className="flex-1 overflow-y-auto bg-background/50">
          <div className="container mx-auto p-8 max-w-7xl">
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/api-keys" element={<ApiKeys />} />
              <Route path="/memories" element={<Memories />} />
              <Route path="/memory-traces" element={<MemoryTraces />} />
              <Route path="/usage" element={<Usage />} />
              <Route path="/checkout/success" element={<BillingSuccess />} />
              <Route path="/checkout/cancel" element={<BillingCancel />} />
              <Route path="*" element={<Navigate to="/dashboard" replace />} />
            </Routes>
          </div>
        </main>
      </div>
    </WorkspaceProvider>
  );
};
