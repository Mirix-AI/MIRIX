import { useEffect, useState } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { useWorkspace } from '@/contexts/WorkspaceContext';
import apiClient from '@/api/client';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Plus, Copy, Check, Trash2 } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ApiKey {
  id: string;
  client_id: string;
  name: string;
  status: string;
  created_at: string;
  api_key?: string; // Only present when just created
}

type Permission = 'all' | 'restricted' | 'read_only';

export const ApiKeys: React.FC = () => {
  const { user } = useAuth();
  const { users, selectedUser } = useWorkspace();
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [loading, setLoading] = useState(true);
  
  // Create dialog state
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [newKeyName, setNewKeyName] = useState('');
  const [selectedUserId, setSelectedUserId] = useState<string>('');
  const [permission, setPermission] = useState<Permission>('all');
  const [creating, setCreating] = useState(false);
  
  // Success dialog state (after key created)
  const [successDialogOpen, setSuccessDialogOpen] = useState(false);
  const [createdKey, setCreatedKey] = useState<ApiKey | null>(null);
  const [copied, setCopied] = useState(false);

  const fetchKeys = async () => {
    if (!user?.id) return;
    try {
      const response = await apiClient.get(`/clients/${user.id}/api-keys`);
      setKeys(response.data);
    } catch (error) {
      console.error('Error fetching keys:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchKeys();
  }, [user?.id]);

  useEffect(() => {
    // Set default selected user when users load
    if (selectedUser && !selectedUserId) {
      setSelectedUserId(selectedUser.id);
    }
  }, [selectedUser, selectedUserId]);

  const handleOpenCreateDialog = () => {
    setNewKeyName('');
    setPermission('all');
    if (selectedUser) {
      setSelectedUserId(selectedUser.id);
    }
    setCreateDialogOpen(true);
  };

  const handleCreateKey = async () => {
    if (!user?.id) return;
    setCreating(true);
    try {
      const response = await apiClient.post(`/clients/${user.id}/api-keys`, {
        name: newKeyName || 'Secret key',
        user_id: selectedUserId,
        permission: permission,
      });
      setCreatedKey(response.data);
      setCreateDialogOpen(false);
      setSuccessDialogOpen(true);
      fetchKeys();
    } catch (error) {
      console.error('Error creating key:', error);
    } finally {
      setCreating(false);
    }
  };

  const handleDeleteKey = async (keyId: string) => {
    if (!user?.id) return;
    if (!confirm('Are you sure you want to delete this API key? This action cannot be undone and any applications using this key will stop working.')) return;
    try {
      await apiClient.delete(`/clients/${user.id}/api-keys/${keyId}`);
      // Remove from local state immediately
      setKeys(keys.filter(k => k.id !== keyId));
    } catch (error) {
      console.error('Error deleting key:', error);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleSuccessDialogClose = () => {
    setSuccessDialogOpen(false);
    setCreatedKey(null);
    setCopied(false);
  };

  const getPermissionLabel = (perm: Permission) => {
    switch (perm) {
      case 'all': return 'All';
      case 'restricted': return 'Restricted';
      case 'read_only': return 'Read only';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight">API keys</h2>
          <p className="text-sm text-muted-foreground mt-1">
            You have permission to view and manage all API keys in this project.
          </p>
        </div>
        <Button onClick={handleOpenCreateDialog}>
          <Plus className="mr-2 h-4 w-4" /> Create new secret key
        </Button>
      </div>

      {/* Info text */}
      <div className="text-sm text-muted-foreground space-y-2">
        <p>
          Do not share your API key with others or expose it in the browser or other client-side code. 
          To protect your account's security, Mirix may automatically disable any API key that has leaked publicly.
        </p>
      </div>

      {/* Table */}
      <div className="border rounded-lg overflow-hidden">
        <table className="w-full">
          <thead className="bg-muted/50">
            <tr className="text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
              <th className="px-6 py-3">Name</th>
              <th className="px-6 py-3">Secret Key</th>
              <th className="px-6 py-3">Created</th>
              <th className="px-6 py-3">Permissions</th>
              <th className="px-6 py-3 w-20"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {loading ? (
              <tr>
                <td colSpan={5} className="px-6 py-8 text-center text-muted-foreground">
                  Loading...
                </td>
              </tr>
            ) : keys.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-6 py-8 text-center text-muted-foreground">
                  No API keys found. Create one to get started.
                </td>
              </tr>
            ) : (
              keys.filter(k => k.status === 'active').map((key) => (
                <tr key={key.id} className="hover:bg-muted/30">
                  <td className="px-6 py-4 font-medium">{key.name || 'Secret key'}</td>
                  <td className="px-6 py-4 font-mono text-sm text-muted-foreground">
                    sk-...{key.id.slice(-4)}
                  </td>
                  <td className="px-6 py-4 text-sm text-muted-foreground">
                    {new Date(key.created_at).toLocaleDateString('en-US', { 
                      month: 'short', 
                      day: 'numeric', 
                      year: 'numeric' 
                    })}
                  </td>
                  <td className="px-6 py-4 text-sm">All</td>
                  <td className="px-6 py-4">
                    <Button 
                      variant="ghost" 
                      size="icon"
                      className="h-8 w-8 text-destructive hover:text-destructive hover:bg-destructive/10"
                      onClick={() => handleDeleteKey(key.id)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Create Key Dialog */}
      <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
        <DialogContent className="sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle>Create new secret key</DialogTitle>
          </DialogHeader>
          
          <div className="space-y-6 py-4">
            {/* User Selection */}
            <div className="space-y-2">
              <Label className="text-sm font-medium">User</Label>
              <select
                value={selectedUserId}
                onChange={(e) => setSelectedUserId(e.target.value)}
                className="w-full h-10 px-3 rounded-md border border-input bg-background text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                {users.map((u) => (
                  <option key={u.id} value={u.id}>
                    {u.name} {u.id === user?.default_user_id ? '(Default)' : ''}
                  </option>
                ))}
              </select>
              <p className="text-xs text-muted-foreground">
                This API key is tied to your user and can make requests against the selected project.
              </p>
            </div>

            {/* Name Input */}
            <div className="space-y-2">
              <Label className="text-sm font-medium">
                Name <span className="text-muted-foreground font-normal">Optional</span>
              </Label>
              <Input
                placeholder="My Test Key"
                value={newKeyName}
                onChange={(e) => setNewKeyName(e.target.value)}
              />
            </div>

            {/* Permissions */}
            <div className="space-y-2">
              <Label className="text-sm font-medium">Permissions</Label>
              <div className="flex gap-2">
                {(['all', 'restricted', 'read_only'] as Permission[]).map((perm) => (
                  <button
                    key={perm}
                    type="button"
                    onClick={() => setPermission(perm)}
                    className={cn(
                      "px-4 py-2 text-sm rounded-md border transition-colors",
                      permission === perm 
                        ? "bg-primary text-primary-foreground border-primary" 
                        : "bg-background border-input hover:bg-accent"
                    )}
                  >
                    {getPermissionLabel(perm)}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <DialogFooter className="gap-2">
            <Button variant="outline" onClick={() => setCreateDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleCreateKey} disabled={creating}>
              {creating ? 'Creating...' : 'Create secret key'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Success Dialog (Show created key) */}
      <Dialog open={successDialogOpen} onOpenChange={handleSuccessDialogClose}>
        <DialogContent className="sm:max-w-[560px]">
          <DialogHeader>
            <DialogTitle>Save your key</DialogTitle>
            <DialogDescription className="pt-2">
              Please save your secret key in a safe place since{' '}
              <span className="font-semibold text-foreground">you won't be able to view it again</span>.
              Keep it secure, as anyone with your API key can make requests on your behalf. 
              If you do lose it, you'll need to generate a new one.
            </DialogDescription>
          </DialogHeader>
          
          <div className="py-4">
            <div className="flex items-center gap-2 p-3 bg-muted rounded-md min-w-0">
              <code className="flex-1 min-w-0 text-sm font-mono truncate">
                {createdKey?.api_key}
              </code>
              <Button 
                variant="outline" 
                size="sm"
                onClick={() => copyToClipboard(createdKey?.api_key || '')}
                className="shrink-0"
              >
                {copied ? <Check className="h-4 w-4 mr-2" /> : <Copy className="h-4 w-4 mr-2" />}
                {copied ? 'Copied' : 'Copy'}
              </Button>
            </div>
            
            <div className="mt-4 space-y-1">
              <div className="text-sm font-medium">Permissions</div>
              <div className="text-sm text-muted-foreground">Read and write API resources</div>
            </div>
          </div>

          <DialogFooter>
            <Button onClick={handleSuccessDialogClose}>
              Done
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};
