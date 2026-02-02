import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useWorkspace } from '@/contexts/WorkspaceContext';
import apiClient from '@/api/client';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { RefreshCcw, Search, ChevronDown, ChevronUp, Trash2, MoreVertical } from 'lucide-react';
import { useSearchParams } from 'react-router-dom';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';

interface MemoryResult {
  memory_type: string;
  id: string;
  summary?: string;
  details?: string;
  content?: string;
  caption?: string;
  timestamp?: string;
  created_at?: string;
}

type MemoryTab =
  | 'episodic'
  | 'semantic'
  | 'procedural'
  | 'resource'
  | 'knowledge'
  | 'core';

interface MemoryCollection<T> {
  total_count: number;
  items: T[];
}

interface EpisodicMemory {
  id: string;
  occurred_at?: string;
  event_type?: string;
  actor?: string;
  summary?: string;
  details?: string;
  created_at?: string;
  updated_at?: string;
}

interface SemanticMemory {
  id: string;
  name?: string;
  summary?: string;
  details?: string;
  source?: string;
  created_at?: string;
  updated_at?: string;
}

interface ProceduralMemory {
  id: string;
  entry_type?: string;
  summary?: string;
  steps?: string[];
  created_at?: string;
  updated_at?: string;
}

interface ResourceMemory {
  id: string;
  resource_type?: string;
  title?: string;
  summary?: string;
  content?: string;
  created_at?: string;
  updated_at?: string;
}

interface KnowledgeMemory {
  id: string;
  entry_type?: string;
  source?: string;
  sensitivity?: string;
  secret_value?: string;
  caption?: string;
  created_at?: string;
  updated_at?: string;
}

interface CoreMemory {
  id: string;
  label?: string;
  value?: string;
}

type MemoryCollections = {
  episodic?: MemoryCollection<EpisodicMemory>;
  semantic?: MemoryCollection<SemanticMemory>;
  procedural?: MemoryCollection<ProceduralMemory>;
  resource?: MemoryCollection<ResourceMemory>;
  knowledge?: MemoryCollection<KnowledgeMemory>;
  core?: MemoryCollection<CoreMemory>;
};

type MemoryTypeFilter = 'all' | MemoryTab;

const DEFAULT_FIELDS: Record<MemoryTypeFilter, string[]> = {
  all: [],
  episodic: ['summary', 'details'],
  semantic: ['name', 'summary', 'details'],
  procedural: ['summary', 'steps'],
  resource: ['summary', 'content'],
  knowledge: ['caption', 'secret_value'],
  core: ['label', 'value'],
};

const MEMORY_TABS: { key: MemoryTab; label: string }[] = [
  { key: 'episodic', label: 'Episodic' },
  { key: 'semantic', label: 'Semantic' },
  { key: 'procedural', label: 'Procedural' },
  { key: 'resource', label: 'Resource' },
  { key: 'knowledge', label: 'Knowledge' },
  { key: 'core', label: 'Core' },
];

const formatDate = (value?: string) => (value ? new Date(value).toLocaleString() : 'N/A');
const normalizeUserToken = (value: string) => value.replace(/_/g, '-');

export const Memories: React.FC = () => {
  const { users, selectedUser, setSelectedUser, isLoading: usersLoading } = useWorkspace();
  const [searchParams, setSearchParams] = useSearchParams();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<MemoryResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [memoryTypeFilter, setMemoryTypeFilter] = useState<MemoryTypeFilter>('all');
  const [searchField, setSearchField] = useState<string>('null');
  const [searchMethod, setSearchMethod] = useState<'bm25' | 'embedding'>('bm25');
  const [viewMode, setViewMode] = useState<'search' | 'list'>('list');
  const [clearingMemory, setClearingMemory] = useState(false);
  const [deletingIds, setDeletingIds] = useState<Record<string, boolean>>({});

  const [memoryTab, setMemoryTab] = useState<MemoryTab>('episodic');
  const [memoryCollections, setMemoryCollections] = useState<MemoryCollections>({});
  const [memoryLoading, setMemoryLoading] = useState(false);
  const [memoryError, setMemoryError] = useState<string | null>(null);
  const [expandedEpisodic, setExpandedEpisodic] = useState<Record<string, boolean>>({});
  const [expandedSemantic, setExpandedSemantic] = useState<Record<string, boolean>>({});
  const [expandedProcedural, setExpandedProcedural] = useState<Record<string, boolean>>({});
  const [expandedResource, setExpandedResource] = useState<Record<string, boolean>>({});
  const [expandedKnowledge, setExpandedKnowledge] = useState<Record<string, boolean>>({});
  const [fieldsByType, setFieldsByType] = useState<Record<MemoryTypeFilter, string[]>>(DEFAULT_FIELDS);
  const prevUserIdRef = useRef<string | null>(null);
  const userParam = searchParams.get('user');
  const searchParamsString = searchParams.toString();
  const userParamAppliedRef = useRef(false);
  const lastUserParamRef = useRef<string | null>(null);
  const normalizedUserParam = userParam ? normalizeUserToken(userParam) : null;
  const userParamValid =
    !!userParam &&
    users.some(
      (user) =>
        user.id === userParam ||
        user.name === userParam ||
        normalizeUserToken(user.id) === normalizedUserParam ||
        normalizeUserToken(user.name) === normalizedUserParam
    );

  // Initialize state from URL params
  useEffect(() => {
    const typeParam = searchParams.get('type') as MemoryTypeFilter | null;
    if (typeParam && (['all', 'episodic', 'semantic', 'procedural', 'resource', 'knowledge', 'core'] as MemoryTypeFilter[]).includes(typeParam)) {
      setMemoryTypeFilter(typeParam);
    }

    const fieldParam = searchParams.get('field');
    if (fieldParam) {
      setSearchField(fieldParam);
    }

    const methodParam = searchParams.get('method');
    if (methodParam === 'bm25' || methodParam === 'embedding') {
      setSearchMethod(methodParam);
    }

    const tabParam = searchParams.get('tab') as MemoryTab | null;
    if (tabParam && (['episodic', 'semantic', 'procedural', 'resource', 'knowledge', 'core'] as MemoryTab[]).includes(tabParam)) {
      setMemoryTab(tabParam);
    }

    const qParam = searchParams.get('q');
    if (qParam !== null) {
      setQuery(qParam);
      setHasSearched(qParam.trim().length > 0);
    }

    const modeParam = searchParams.get('mode');
    if (modeParam === 'search' || modeParam === 'list') {
      setViewMode(modeParam);
    }

  }, []); // run once on mount

  const fetchMemoryComponents = useCallback(async () => {
    if (!selectedUser) return;
    setMemoryLoading(true);
    setMemoryError(null);
    try {
      const fieldsResponse = await apiClient.get('/memory/fields');
      setFieldsByType({ ...DEFAULT_FIELDS, ...(fieldsResponse.data?.fields || {}) });

      const response = await apiClient.get('/memory/components', {
        params: {
          user_id: selectedUser.id,
          memory_type: 'all',
          limit: 50,
        },
      });
      setMemoryCollections(response.data.memories || {});
    } catch (error) {
      console.error('Error loading memory components:', error);
      setMemoryError('Failed to load memory components. Please try again.');
    } finally {
      setMemoryLoading(false);
    }
  }, [selectedUser?.id]);

  const handleSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!selectedUser || !query.trim()) return;
    
    setLoading(true);
    setHasSearched(true);
    try {
      const response = await apiClient.get('/memory/search', {
        params: {
          user_id: selectedUser.id,
          query: query,
          memory_type: memoryTypeFilter,
          search_field: memoryTypeFilter === 'all' ? 'null' : searchField,
          search_method: searchMethod,
          limit: 20,
        },
      });
      setResults(response.data.results);
    } catch (error) {
      console.error('Error searching memories:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleClearMemory = async () => {
    if (!selectedUser || clearingMemory) return;
    const confirmed = confirm(
      `Clear all memories for ${selectedUser.name}? This will permanently delete episodic, semantic, procedural, resource, and knowledge memories, plus messages and blocks for this user.`
    );
    if (!confirmed) return;
    setClearingMemory(true);
    try {
      await apiClient.post('/memory/clear', { user_id: selectedUser.id });
      setResults([]);
      setHasSearched(false);
      setMemoryCollections({});
      setMemoryError(null);
      setExpandedEpisodic({});
      setExpandedSemantic({});
      setExpandedProcedural({});
      setExpandedResource({});
      setExpandedKnowledge({});
      await fetchMemoryComponents();
    } catch (error) {
      console.error('Error clearing memories:', error);
      setMemoryError('Failed to clear memories. Please try again.');
    } finally {
      setClearingMemory(false);
    }
  };

  const getDeleteEndpoint = (memoryType: string, memoryId: string) => {
    const normalized = memoryType.toLowerCase();
    switch (normalized) {
      case 'episodic':
      case 'semantic':
      case 'procedural':
      case 'resource':
      case 'knowledge':
        return `/memory/${normalized}/${memoryId}`;
      default:
        return null;
    }
  };

  const getMemoryLabel = (memoryType: string) => {
    const normalized = memoryType.toLowerCase();
    return `${normalized} memory`;
  };

  const isDeletableMemoryType = (memoryType: string) => {
    const normalized = memoryType.toLowerCase();
    return ['episodic', 'semantic', 'procedural', 'resource', 'knowledge'].includes(normalized);
  };

  const clearExpandedState = (memoryType: string, memoryId: string) => {
    const normalized = memoryType.toLowerCase();
    if (normalized === 'episodic') {
      setExpandedEpisodic(({ [memoryId]: _removed, ...rest }) => rest);
    }
    if (normalized === 'semantic') {
      setExpandedSemantic(({ [memoryId]: _removed, ...rest }) => rest);
    }
    if (normalized === 'procedural') {
      setExpandedProcedural(({ [memoryId]: _removed, ...rest }) => rest);
    }
    if (normalized === 'resource') {
      setExpandedResource(({ [memoryId]: _removed, ...rest }) => rest);
    }
    if (normalized === 'knowledge') {
      setExpandedKnowledge(({ [memoryId]: _removed, ...rest }) => rest);
    }
  };

  const handleDeleteMemory = async (memoryType: string, memoryId: string) => {
    if (!selectedUser || deletingIds[memoryId] || !isDeletableMemoryType(memoryType)) return;
    const endpoint = getDeleteEndpoint(memoryType, memoryId);
    if (!endpoint) {
      console.error('Unsupported memory type for deletion:', memoryType);
      return;
    }
    const confirmed = confirm(
      `Delete this ${getMemoryLabel(memoryType)} for ${selectedUser.name}? This action cannot be undone.`
    );
    if (!confirmed) return;
    setDeletingIds((prev) => ({ ...prev, [memoryId]: true }));
    try {
      await apiClient.delete(endpoint);
      setResults((prev) => prev.filter((item) => item.id !== memoryId));
      setMemoryCollections((prev) => {
        const normalized = memoryType.toLowerCase() as keyof MemoryCollections;
        const bucket = prev[normalized];
        if (!bucket) return prev;
        const nextItems = bucket.items.filter((item) => item.id !== memoryId);
        return {
          ...prev,
          [normalized]: {
            ...bucket,
            items: nextItems,
            total_count: Math.max(0, bucket.total_count - 1),
          },
        };
      });
      clearExpandedState(memoryType, memoryId);
    } catch (error) {
      console.error('Error deleting memory:', error);
      setMemoryError('Failed to delete memory. Please try again.');
    } finally {
      setDeletingIds((prev) => ({ ...prev, [memoryId]: false }));
    }
  };

  // Reset when switching to a different user (but preserve on initial load)
  useEffect(() => {
    const prev = prevUserIdRef.current;
    if (selectedUser?.id) {
      if (prev && prev !== selectedUser.id) {
        setResults([]);
        setHasSearched(false);
        setQuery('');
        setMemoryCollections({});
        setMemoryError(null);
        setMemoryTab('episodic');
        setExpandedEpisodic({});
        setExpandedSemantic({});
        setExpandedProcedural({});
        setExpandedResource({});
        setExpandedKnowledge({});
        setMemoryTypeFilter('all');
        setSearchField('null');
        setSearchMethod('bm25');
      }
      prevUserIdRef.current = selectedUser.id;
      fetchMemoryComponents();
    }
  }, [selectedUser?.id, fetchMemoryComponents]);

  useEffect(() => {
    if (usersLoading || !userParam || users.length === 0) return;
    const matchingUser =
      users.find((user) => user.id === userParam || user.name === userParam) ||
      (normalizedUserParam
        ? users.find(
            (user) =>
              normalizeUserToken(user.id) === normalizedUserParam ||
              normalizeUserToken(user.name) === normalizedUserParam
          )
        : null);
    const userParamChanged = userParam !== lastUserParamRef.current;
    if (matchingUser && (userParamChanged || !userParamAppliedRef.current)) {
      if (selectedUser?.id !== matchingUser.id) {
        setSelectedUser(matchingUser);
      }
      userParamAppliedRef.current = true;
    }
    lastUserParamRef.current = userParam;
  }, [usersLoading, userParam, users, selectedUser?.id, setSelectedUser]);

  const toggleEpisodic = (id: string) => {
    setExpandedEpisodic((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const toggleSemantic = (id: string) => {
    setExpandedSemantic((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const toggleProcedural = (id: string) => {
    setExpandedProcedural((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const toggleResource = (id: string) => {
    setExpandedResource((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const toggleKnowledge = (id: string) => {
    setExpandedKnowledge((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  // Keep search_field sensible when filters change
  useEffect(() => {
    if (memoryTypeFilter === 'all') {
      setSearchField('null');
      return;
    }
    const availableFields = fieldsByType[memoryTypeFilter] || [];
    if (!availableFields.includes(searchField)) {
      setSearchField(availableFields[0] || 'summary');
      return;
    }
    // Avoid invalid combinations: embedding cannot search resource.content or knowledge.secret_value
    if (searchMethod === 'embedding') {
      if (memoryTypeFilter === 'resource' && searchField === 'content') {
        setSearchField('summary');
      }
      if (memoryTypeFilter === 'knowledge' && searchField === 'secret_value') {
        setSearchField('caption');
      }
    }
  }, [memoryTypeFilter, searchMethod, searchField, fieldsByType]);

  // Sync state to URL so refresh preserves selection
  useEffect(() => {
    const params = new URLSearchParams();
    if (query) params.set('q', query);
    params.set('type', memoryTypeFilter);
    params.set('field', memoryTypeFilter === 'all' ? 'null' : searchField);
    params.set('method', searchMethod);
    params.set('tab', memoryTab);
    params.set('mode', viewMode);
    const deferUserParamUpdate = !!userParam && userParamValid && !userParamAppliedRef.current;
    const userIdParam = deferUserParamUpdate ? userParam : selectedUser?.id || userParam;
    if (userIdParam) {
      params.set('user', userIdParam);
    }
    if (params.toString() !== searchParamsString) {
      setSearchParams(params, { replace: true });
    }
  }, [
    query,
    memoryTypeFilter,
    searchField,
    searchMethod,
    memoryTab,
    viewMode,
    selectedUser?.id,
    userParam,
    searchParamsString,
    setSearchParams,
  ]);

  const renderTabContent = () => {
    if (!selectedUser) return null;

    const emptyState = (label: string) => (
      <div className="text-sm text-muted-foreground text-center py-8">
        No {label.toLowerCase()} memory entries found for {selectedUser.name}.
      </div>
    );

    switch (memoryTab) {
      case 'episodic': {
        const bucket = memoryCollections.episodic;
        if (!bucket || bucket.items.length === 0) return emptyState('Episodic');
        return (
          <div className="space-y-3">
            <div className="text-xs text-muted-foreground">
              Showing {bucket.items.length} of {bucket.total_count} episodic events
            </div>
            {bucket.items.map((item) => (
              <Card key={item.id} className="border border-primary/10">
                <CardContent className="space-y-3 pt-3 pb-3">
                  <div className="flex flex-wrap items-center justify-between text-xs text-muted-foreground gap-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-semibold uppercase tracking-wide">{item.event_type || 'Event'}</span>
                      {item.actor && (
                        <span className="px-2 py-0.5 rounded bg-muted text-[11px] uppercase tracking-wide">
                          {item.actor}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      {item.occurred_at && <span>{formatDate(item.occurred_at)}</span>}
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 text-destructive hover:text-destructive hover:bg-destructive/10"
                        onClick={() => handleDeleteMemory('episodic', item.id)}
                        disabled={!!deletingIds[item.id]}
                        title="Delete memory"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  </div>
                  <div className="flex items-start justify-between gap-2">
                    <div className="font-semibold text-base">{item.summary || 'No summary'}</div>
                    {item.details && (
                      <button
                        type="button"
                        onClick={() => toggleEpisodic(item.id)}
                        className="flex items-center text-xs text-muted-foreground hover:text-foreground transition-colors"
                      >
                        {expandedEpisodic[item.id] ? (
                          <>
                            Hide details <ChevronUp className="ml-1 h-4 w-4" />
                          </>
                        ) : (
                          <>
                            Show details <ChevronDown className="ml-1 h-4 w-4" />
                          </>
                        )}
                      </button>
                    )}
                  </div>
                  {item.details && expandedEpisodic[item.id] && (
                    <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                      {item.details}
                    </p>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        );
      }
      case 'semantic': {
        const bucket = memoryCollections.semantic;
        if (!bucket || bucket.items.length === 0) return emptyState('Semantic');
        return (
          <div className="space-y-3">
            <div className="text-xs text-muted-foreground">
              Showing {bucket.items.length} of {bucket.total_count} semantic entries
            </div>
            {bucket.items.map((item) => (
              <Card key={item.id} className="border border-primary/10">
                <CardContent className="space-y-2 pt-3 pb-3">
                  <div className="flex flex-wrap items-center justify-between gap-3 text-[12px] text-muted-foreground tracking-wide">
                    <div className="flex flex-wrap items-center gap-3">
                      <span className="font-semibold uppercase tracking-wide">Concept</span>
                      <span className="text-base font-semibold text-foreground leading-tight">{item.name || 'Untitled'}</span>
                      {item.source && (
                        <span className="text-xs text-muted-foreground">
                          <span className="font-semibold text-foreground mr-1">Source:</span>
                          {item.source}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-2 text-[12px]">
                      {item.created_at && <span>{formatDate(item.created_at)}</span>}
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 text-destructive hover:text-destructive hover:bg-destructive/10"
                        onClick={() => handleDeleteMemory('semantic', item.id)}
                        disabled={!!deletingIds[item.id]}
                        title="Delete memory"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  </div>
                  <div className="flex items-start justify-between gap-2">
                    <div className="text-sm text-muted-foreground leading-snug">
                      <span className="font-semibold mr-2 text-foreground">Summary:</span>
                      <span className="align-middle">{item.summary || 'No summary provided.'}</span>
                    </div>
                    {item.details && (
                      <button
                        type="button"
                        onClick={() => toggleSemantic(item.id)}
                        className="flex items-center text-[12px] text-muted-foreground hover:text-foreground transition-colors"
                      >
                        {expandedSemantic[item.id] ? (
                          <>
                            Hide details <ChevronUp className="ml-1 h-4 w-4" />
                          </>
                        ) : (
                          <>
                            Show details <ChevronDown className="ml-1 h-4 w-4" />
                          </>
                        )}
                      </button>
                    )}
                  </div>
                  {item.details && expandedSemantic[item.id] && (
                    <div className="text-sm text-muted-foreground whitespace-pre-wrap leading-snug">
                      <span className="font-semibold mr-2 text-foreground">Details:</span>
                      <span className="align-middle">{item.details}</span>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        );
      }
      case 'procedural': {
        const bucket = memoryCollections.procedural;
        if (!bucket || bucket.items.length === 0) return emptyState('Procedural');
        return (
          <div className="space-y-3">
            <div className="text-xs text-muted-foreground">
              Showing {bucket.items.length} of {bucket.total_count} procedures
            </div>
            {bucket.items.map((item) => (
              <Card key={item.id} className="border border-primary/10">
                <CardContent className="space-y-2 pt-3 pb-3">
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span className="font-semibold uppercase tracking-wide">
                      {item.entry_type || 'Procedure'}
                    </span>
                    <div className="flex items-center gap-2">
                      {item.created_at && <span>{formatDate(item.created_at)}</span>}
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 text-destructive hover:text-destructive hover:bg-destructive/10"
                        onClick={() => handleDeleteMemory('procedural', item.id)}
                        disabled={!!deletingIds[item.id]}
                        title="Delete memory"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  </div>
                  <div className="flex items-start justify-between gap-2">
                    <div className="font-semibold text-base">{item.summary || 'No summary'}</div>
                    {item.steps && item.steps.length > 0 && (
                      <button
                        type="button"
                        onClick={() => toggleProcedural(item.id)}
                        className="flex items-center text-xs text-muted-foreground hover:text-foreground transition-colors"
                      >
                        {expandedProcedural[item.id] ? (
                          <>
                            Hide details <ChevronUp className="ml-1 h-4 w-4" />
                          </>
                        ) : (
                          <>
                            Show details <ChevronDown className="ml-1 h-4 w-4" />
                          </>
                        )}
                      </button>
                    )}
                  </div>
                  {item.steps && item.steps.length > 0 ? (
                    expandedProcedural[item.id] ? (
                      <ul className="list-decimal space-y-1 pl-5 text-sm text-muted-foreground">
                        {item.steps.map((step, idx) => (
                          <li key={`${item.id}-step-${idx}`}>{step}</li>
                        ))}
                      </ul>
                    ) : null
                  ) : (
                    <p className="text-sm text-muted-foreground">No steps recorded.</p>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        );
      }
      case 'resource': {
        const bucket = memoryCollections.resource;
        if (!bucket || bucket.items.length === 0) return emptyState('Resource');
        return (
          <div className="space-y-3">
            <div className="text-xs text-muted-foreground">
              Showing {bucket.items.length} of {bucket.total_count} resources
            </div>
            {bucket.items.map((item) => (
              <Card key={item.id} className="border border-primary/10">
                <CardContent className="space-y-2 pt-3 pb-3">
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span className="font-semibold uppercase tracking-wide">
                      {item.resource_type || 'Resource'}
                    </span>
                    <div className="flex items-center gap-2">
                      {item.created_at && <span>{formatDate(item.created_at)}</span>}
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 text-destructive hover:text-destructive hover:bg-destructive/10"
                        onClick={() => handleDeleteMemory('resource', item.id)}
                        disabled={!!deletingIds[item.id]}
                        title="Delete memory"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  </div>
                  <div className="font-semibold text-base">{item.title || 'Untitled'}</div>
                  <div className="flex items-start justify-between gap-2">
                    <p className="text-sm text-muted-foreground">{item.summary || 'No summary provided.'}</p>
                    {item.content && (
                      <button
                        type="button"
                        onClick={() => toggleResource(item.id)}
                        className="flex items-center text-xs text-muted-foreground hover:text-foreground transition-colors"
                      >
                        {expandedResource[item.id] ? (
                          <>
                            Hide details <ChevronUp className="ml-1 h-4 w-4" />
                          </>
                        ) : (
                          <>
                            Show details <ChevronDown className="ml-1 h-4 w-4" />
                          </>
                        )}
                      </button>
                    )}
                  </div>
                  {item.content ? (
                    expandedResource[item.id] ? (
                      <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                        {item.content}
                      </p>
                    ) : null
                  ) : (
                    <p className="text-sm text-muted-foreground">No content stored.</p>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        );
      }
      case 'knowledge': {
        const bucket = memoryCollections.knowledge;
        if (!bucket || bucket.items.length === 0) return emptyState('Knowledge');
        return (
          <div className="space-y-3">
            <div className="text-xs text-muted-foreground">
              Showing {bucket.items.length} of {bucket.total_count} knowledge items
            </div>
            {bucket.items.map((item) => (
              <Card key={item.id} className="border border-primary/10">
                <CardContent className="space-y-2 pt-3 pb-3">
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span className="font-semibold uppercase tracking-wide">
                      {item.entry_type || 'Entry'}
                    </span>
                    <div className="flex items-center gap-2">
                      {item.created_at && <span>{formatDate(item.created_at)}</span>}
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 text-destructive hover:text-destructive hover:bg-destructive/10"
                        onClick={() => handleDeleteMemory('knowledge', item.id)}
                        disabled={!!deletingIds[item.id]}
                        title="Delete memory"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  </div>
                  <div className="flex items-start justify-between gap-2">
                    <div className="font-semibold text-base">{item.caption || 'No caption'}</div>
                    {item.secret_value && (
                      <button
                        type="button"
                        onClick={() => toggleKnowledge(item.id)}
                        className="flex items-center text-xs text-muted-foreground hover:text-foreground transition-colors"
                      >
                        {expandedKnowledge[item.id] ? (
                          <>
                            Hide details <ChevronUp className="ml-1 h-4 w-4" />
                          </>
                        ) : (
                          <>
                            Show details <ChevronDown className="ml-1 h-4 w-4" />
                          </>
                        )}
                      </button>
                    )}
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Source: {item.source || 'Unknown'} | Sensitivity: {item.sensitivity || 'Unspecified'}
                  </p>
                  {item.secret_value && expandedKnowledge[item.id] && (
                    <div className="rounded bg-muted px-3 py-2 text-sm font-mono break-all">
                      {item.secret_value}
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        );
      }
      case 'core': {
        const bucket = memoryCollections.core;
        if (!bucket || bucket.items.length === 0) return emptyState('Core');
        return (
          <div className="space-y-3">
            <div className="text-xs text-muted-foreground">
              Showing {bucket.items.length} of {bucket.total_count} core blocks
            </div>
            {bucket.items.map((item) => (
              <Card key={item.id} className="border border-primary/10">
                <CardContent className="space-y-2 pt-3 pb-3">
                  <div className="text-xs text-muted-foreground uppercase tracking-wide font-semibold">
                    {item.label || 'Core Block'}
                  </div>
                  <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                    {item.value || 'No value set.'}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>
        );
      }
      default:
        return null;
    }
  };

  if (!selectedUser) {
    return (
      <div className="text-center py-12">
        <h2 className="text-2xl font-bold text-muted-foreground">Please select a user to view memories</h2>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Memories: {selectedUser.name}</h2>
          <p className="text-muted-foreground">
            Browse all memory components and search across them for {selectedUser.name}.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant={viewMode === 'search' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('search')}
          >
            Search mode
          </Button>
          <Button
            variant={viewMode === 'list' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('list')}
          >
            List mode
          </Button>
          <Popover>
            <PopoverTrigger asChild>
              <Button variant="outline" size="icon" aria-label="More actions">
                <MoreVertical className="h-4 w-4" />
              </Button>
            </PopoverTrigger>
            <PopoverContent align="end" className="w-56 p-1">
              <Button
                variant="ghost"
                className="w-full justify-start text-destructive hover:text-destructive hover:bg-destructive/10"
                onClick={handleClearMemory}
                disabled={clearingMemory || memoryLoading || usersLoading}
              >
                <Trash2 className="mr-2 h-4 w-4" />
                {clearingMemory ? 'Clearing...' : 'Clear memory'}
              </Button>
              <div className="px-2 py-1 text-xs text-muted-foreground">
                Permanently deletes all memories for this user.
              </div>
            </PopoverContent>
          </Popover>
        </div>
      </div>

      {viewMode === 'search' && (
        <>
          <Card>
            <CardHeader>
              <CardTitle>Search Memories</CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSearch} className="space-y-3">
                <div className="flex flex-col gap-3 md:flex-row md:items-center md:flex-wrap">
                  <Input
                    placeholder="Query (e.g., 'meeting')"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    className="w-full md:w-80 lg:w-96"
                  />
                  <select
                    value={memoryTypeFilter}
                    onChange={(e) => setMemoryTypeFilter(e.target.value as any)}
                    className="h-10 rounded-md border bg-background px-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 w-full md:w-40"
                  >
                    <option value="all">All Types</option>
                    <option value="episodic">Episodic</option>
                    <option value="semantic">Semantic</option>
                    <option value="procedural">Procedural</option>
                    <option value="resource">Resource</option>
                    <option value="knowledge">Knowledge</option>
                    <option value="core">Core</option>
                  </select>
                  <select
                    value={searchMethod}
                    onChange={(e) => setSearchMethod(e.target.value as any)}
                    className="h-10 rounded-md border bg-background px-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 w-full md:w-32"
                  >
                    <option value="bm25">BM25</option>
                    <option value="embedding">Embedding</option>
                  </select>
                  <select
                    value={memoryTypeFilter === 'all' ? 'null' : searchField}
                    onChange={(e) => setSearchField(e.target.value)}
                    disabled={memoryTypeFilter === 'all'}
                    title={memoryTypeFilter === 'all' ? 'Select a Memory Type to choose Field' : undefined}
                    className="h-10 rounded-md border bg-background px-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 w-full md:w-44"
                  >
                    <option value="null">Auto Field</option>
                    {(fieldsByType[memoryTypeFilter] || []).map((field) => (
                      <option key={field} value={field}>
                        {field}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="flex justify-end">
                  <Button type="submit" disabled={loading || !selectedUser}>
                    <Search className="mr-2 h-4 w-4" /> Search
                  </Button>
                </div>
              </form>
            </CardContent>
          </Card>

          <div className="space-y-4">
            {loading ? (
              <div className="text-center py-8">Searching...</div>
            ) : hasSearched && results.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">No memories found matching your query.</div>
            ) : (
              hasSearched &&
              results.map((memory) => (
                <Card key={memory.id} className="overflow-hidden">
                  <div className="border-l-4 border-primary pl-4 py-4 pr-4">
                    <div className="flex items-center justify-between gap-2 mb-2">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground bg-muted px-2 py-0.5 rounded">
                          {memory.memory_type}
                        </span>
                        {memory.timestamp && (
                          <span className="text-xs text-muted-foreground">
                            {new Date(memory.timestamp).toLocaleString()}
                          </span>
                        )}
                      </div>
                      {isDeletableMemoryType(memory.memory_type) && (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7 text-destructive hover:text-destructive hover:bg-destructive/10"
                          onClick={() => handleDeleteMemory(memory.memory_type, memory.id)}
                          disabled={!!deletingIds[memory.id]}
                          title="Delete memory"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                      )}
                    </div>
                    <h4 className="font-medium text-lg mb-1">
                      {memory.summary || memory.caption || 'No summary'}
                    </h4>
                    <p className="text-sm text-muted-foreground line-clamp-3">
                      {memory.details || memory.content || 'No additional details.'}
                    </p>
                  </div>
                </Card>
              ))
            )}
          </div>
        </>
      )}

      {viewMode === 'list' && (
        <Card>
          <CardHeader className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div>
              <CardTitle>Memory Components</CardTitle>
              <p className="text-sm text-muted-foreground">
                Inspect each memory table stored for {selectedUser.name}. Use the tabs to switch between components.
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={fetchMemoryComponents}
              disabled={memoryLoading}
            >
              <RefreshCcw className="mr-2 h-4 w-4" />
              Refresh
            </Button>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap gap-2">
              {MEMORY_TABS.map((tab) => (
                <button
                  key={tab.key}
                  onClick={() => setMemoryTab(tab.key)}
                  disabled={memoryLoading}
                  className={`px-3 py-2 rounded-md text-sm border transition-colors ${
                    memoryTab === tab.key
                      ? 'bg-primary text-primary-foreground border-primary'
                      : 'bg-muted text-muted-foreground hover:text-foreground'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            <div className="rounded-lg border p-4">
              {memoryLoading ? (
                <div className="text-center py-8 text-muted-foreground">Loading memories...</div>
              ) : memoryError ? (
                <div className="text-sm text-red-500">{memoryError}</div>
              ) : (
                renderTabContent()
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

