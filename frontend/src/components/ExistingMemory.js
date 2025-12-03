import React, { useState, useEffect } from 'react';
import './ExistingMemory.css';
import MemoryTreeVisualization from './MemoryTreeVisualization';
import MemoryReferences from './MemoryReferences';
import UploadExportModal from './UploadExportModal';
import queuedFetch from '../utils/requestQueue';
import { useTranslation } from 'react-i18next';

const ExistingMemory = ({ settings }) => {
  const { t } = useTranslation();
  const [activeSubTab, setActiveSubTab] = useState('past-events');
  const [memoryData, setMemoryData] = useState({
    'past-events': [],
    'semantic': [],
    'procedural': [],
    'docs-files': [],
    'core-understanding': [],
    'credentials': [],
    'raw-memory': []
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedItems, setExpandedItems] = useState(new Set());
  const [showOnlyReferenced, setShowOnlyReferenced] = useState(false);
  const [expandedReferences, setExpandedReferences] = useState(new Set());
  const [expandedAppGroups, setExpandedAppGroups] = useState({});
  const [highlightedRawMemoryId, setHighlightedRawMemoryId] = useState(null);
  const [showOnlyReferencedRaw, setShowOnlyReferencedRaw] = useState(false);
  const [highlightedMemoryId, setHighlightedMemoryId] = useState(null);
  // State for memory view modes (list or tree) for each memory type
  const [viewModes, setViewModes] = useState({
    'past-events': 'list',
    'semantic': 'list',
    'procedural': 'list',
    'docs-files': 'list',
    'raw-memory': 'list'
  });
  // State for Upload & Export modal
  const [showUploadExportModal, setShowUploadExportModal] = useState(false);

  // State for Reflexion processing
  const [isReflexionProcessing, setIsReflexionProcessing] = useState(false);
  const [reflexionMessage, setReflexionMessage] = useState('');
  const [reflexionSuccess, setReflexionSuccess] = useState(null);

  // State for tracking edits to core memories
  const [editingCoreMemories, setEditingCoreMemories] = useState(new Set());
  const [editedCoreMemories, setEditedCoreMemories] = useState({});
  const [savingBlocks, setSavingBlocks] = useState(new Set());
  const [saveErrors, setSaveErrors] = useState({});
  const [saveSuccesses, setSaveSuccesses] = useState({});

  // Helper function to get view mode for current tab
  const getCurrentViewMode = () => viewModes[activeSubTab] || 'list';

  // Helper function to set view mode for current tab
  const setCurrentViewMode = (mode) => {
    setViewModes(prev => ({
      ...prev,
      [activeSubTab]: mode
    }));
  };

  // Function to highlight matching text
  const highlightText = (text, query) => {
    if (!text || !query.trim()) {
      return text;
    }

    const searchTerm = query.trim();
    const regex = new RegExp(`(${searchTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    const parts = text.split(regex);

    return parts.map((part, index) =>
      regex.test(part) ? (
        <span key={index} className="search-highlight">{part}</span>
      ) : part
    );
  };

  // Function to render text with preserved newlines
  const renderTextWithNewlines = (text, query) => {
    if (!text) return text;

    // Split by newlines first
    const lines = text.split('\n');

    return lines.map((line, index) => (
      <React.Fragment key={index}>
        {query && query.trim() ? highlightText(line, query) : line}
        {index < lines.length - 1 && <br />}
      </React.Fragment>
    ));
  };

  // Fetch memory data for each type
  const fetchMemoryData = async (memoryType) => {
    try {
      setLoading(true);
      setError(null);

      let endpoint = '';
      switch (memoryType) {
        case 'past-events':
          endpoint = '/memory/episodic';
          break;
        case 'semantic':
          endpoint = '/memory/semantic';
          break;
        case 'procedural':
          endpoint = '/memory/procedural';
          break;
        case 'docs-files':
          endpoint = '/memory/resources';
          break;
        case 'raw-memory':
          endpoint = '/memory/raw';
          break;
        case 'core-understanding':
          endpoint = '/memory/core';
          break;
        case 'credentials':
          endpoint = '/memory/credentials';
          break;
        default:
          return;
      }

      const response = await queuedFetch(`${settings.serverUrl}${endpoint}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch ${memoryType}: ${response.statusText}`);
      }

      const data = await response.json();
      setMemoryData(prev => ({
        ...prev,
        [memoryType]: data
      }));
    } catch (err) {
      console.error(`Error fetching ${memoryType}:`, err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Get all raw_memory ids that are referenced by semantic/episodic memories
  const getReferencedRawMemoryIds = () => {
    const referencedIds = new Set();

    // Helper to extract ID from ref (handles both string IDs and object refs)
    const extractId = (ref) => {
      if (typeof ref === 'string') {
        return ref;
      } else if (ref && ref.id) {
        return ref.id;
      }
      return null;
    };

    // Collect from all memory types
    const memoryTypes = ['semantic', 'past-events', 'skills-procedures', 'docs-files', 'core-understanding', 'credentials'];

    memoryTypes.forEach(memType => {
      (memoryData[memType] || []).forEach(item => {
        if (item.raw_memory_references && Array.isArray(item.raw_memory_references)) {
          item.raw_memory_references.forEach(ref => {
            const id = extractId(ref);
            if (id) referencedIds.add(id);
          });
        }
      });
    });

    return referencedIds;
  };

  // Filter memories based on search query and reference filter
  const filterMemories = (memories, query) => {
    let filtered = memories;

    // For raw-memory tab, apply special filter for referenced items
    if (activeSubTab === 'raw-memory' && showOnlyReferencedRaw) {
      const referencedIds = getReferencedRawMemoryIds();
      filtered = filtered.filter(item => referencedIds.has(item.id));
    }

    // For semantic/episodic tabs, filter by references if toggle is enabled
    if (['semantic', 'past-events'].includes(activeSubTab) && showOnlyReferenced) {
      filtered = filtered.filter(item => {
        return item.raw_memory_references && item.raw_memory_references.length > 0;
      });
    }

    // Filter by search query
    if (!query.trim()) {
      return filtered;
    }

    const searchTerm = query.toLowerCase();

    return filtered.filter(item => {
      // Search by id for all memory types (enables navigation from references)
      if (item.id && item.id.toLowerCase().includes(searchTerm)) {
        return true;
      }

      // Search in different fields depending on memory type
      const searchableText = [
        item.content,
        item.description,
        item.title,
        item.name,
        item.filename,
        item.service,
        item.aspect,
        item.category,
        item.understanding,
        item.context,
        item.summary,
        item.details,
        item.event_type,
        item.type,
        item.caption,
        item.entry_type,
        item.source,
        item.sensitivity,
        // Raw memory specific fields
        item.source_app,
        item.source_url,
        item.ocr_text,
        // Search in tags if they exist
        ...(item.tags || []),
        // Search in emotions if they exist
        ...(item.emotions || [])
      ]
        .filter(Boolean) // Remove null/undefined values
        .join(' ')
        .toLowerCase();

      return searchableText.includes(searchTerm);
    });
  };

  // Toggle expand/collapse for semantic memory details
  const toggleExpanded = (itemId) => {
    setExpandedItems(prev => {
      const newSet = new Set(prev);
      if (newSet.has(itemId)) {
        newSet.delete(itemId);
      } else {
        newSet.add(itemId);
      }
      return newSet;
    });
  };

  // Check if search query matches in details for auto-expansion
  const shouldAutoExpand = (item, query, memoryType) => {
    if (!query.trim()) return false;

    const searchTerm = query.toLowerCase();

    // For raw-memory, check OCR text
    if (memoryType === 'raw-memory' && item.ocr_text) {
      return item.ocr_text.toLowerCase().includes(searchTerm);
    }

    // For other types, check details
    if (item.details) {
      return item.details.toLowerCase().includes(searchTerm);
    }

    return false;
  };

  // Auto-expand items when search query matches their details
  useEffect(() => {
    if ((activeSubTab === 'semantic' || activeSubTab === 'past-events' || activeSubTab === 'raw-memory') && searchQuery.trim()) {
      const currentData = memoryData[activeSubTab] || [];
      const itemsToExpand = new Set();

      currentData.forEach((item, index) => {
        if (shouldAutoExpand(item, searchQuery, activeSubTab)) {
          if (activeSubTab === 'semantic') {
            itemsToExpand.add(`semantic-${index}`);
          } else if (activeSubTab === 'past-events') {
            itemsToExpand.add(`episodic-${index}`);
          } else if (activeSubTab === 'raw-memory') {
            itemsToExpand.add(`raw-${index}`);
          }
        }
      });

      setExpandedItems(itemsToExpand);
    } else if (!searchQuery.trim()) {
      // Clear expanded items when search is cleared
      setExpandedItems(new Set());
    }
  }, [searchQuery, memoryData, activeSubTab]);

  // Fetch data when component mounts or active tab changes
  useEffect(() => {
    fetchMemoryData(activeSubTab);
    // Clear expanded items when switching tabs
    setExpandedItems(new Set());
    // Clear edited core memories when switching away from core-understanding
    if (activeSubTab !== 'core-understanding') {
      setEditingCoreMemories(new Set());
      setEditedCoreMemories({});
      setSavingBlocks(new Set());
      setSaveErrors({});
      setSaveSuccesses({});
    }
  }, [activeSubTab, settings.serverUrl]);

  // Refresh data when backend reconnects
  useEffect(() => {
    if (settings.lastBackendRefresh && settings.serverUrl) {
      console.log('ExistingMemory: backend reconnected, refreshing data');
      fetchMemoryData(activeSubTab);
    }
  }, [settings.lastBackendRefresh, settings.serverUrl, activeSubTab]);

  const renderMemoryContent = () => {
    const currentViewMode = getCurrentViewMode();

    // Handle tree view for memory types that support it
    if (currentViewMode === 'tree') {
      const treeMemoryTypes = ['past-events', 'semantic', 'procedural', 'docs-files'];

      if (treeMemoryTypes.includes(activeSubTab)) {
        // Use generic tree visualization for all memory types
        const memoryTypeMap = {
          'past-events': 'episodic',
          'semantic': 'semantic',
          'procedural': 'procedural',
          'docs-files': 'resource'
        };

        const memoryType = memoryTypeMap[activeSubTab];

        return (
          <MemoryTreeVisualization
            memoryType={memoryType}
            serverUrl={settings.serverUrl}
            getItemTitle={(item) => {
              switch (memoryType) {
                case 'episodic':
                  return item.summary || 'Episodic Event';
                case 'semantic':
                  return item.title || item.name || item.summary || 'Semantic Item';
                case 'procedural':
                  return item.summary || item.title || 'Procedure';
                case 'resource':
                  return item.filename || item.name || 'Resource';
                default:
                  return item.title || item.name || 'Memory Item';
              }
            }}
            getItemDetails={(item) => {
              return {
                summary: item.summary,
                details: item.details || item.content
              };
            }}
          />
        );
      }
    }

    const currentData = memoryData[activeSubTab] || [];
    const filteredData = filterMemories(currentData, searchQuery);

    if (loading) {
      return (
        <div className="memory-loading">
          <div className="loading-spinner"></div>
          <p>{t('memory.states.loading')}</p>
        </div>
      );
    }

    if (error) {
      return (
        <div className="memory-error">
          <p>{t('memory.states.error', { error })}</p>
          <button onClick={() => fetchMemoryData(activeSubTab)} className="retry-button">
            {t('memory.actions.retry')}
          </button>
        </div>
      );
    }

    if (currentData.length === 0) {
      return (
        <div className="memory-empty">
          <p>{t('memory.states.empty', { type: getMemoryTypeLabel(activeSubTab).toLowerCase() })}</p>
        </div>
      );
    }

    if (filteredData.length === 0 && searchQuery.trim()) {
      return (
        <div className="memory-empty">
          <p>{t('memory.search.noResults', { type: getMemoryTypeLabel(activeSubTab).toLowerCase(), query: searchQuery })}</p>
          <p>{t('memory.search.tryDifferent')}</p>
        </div>
      );
    }

    return (
      <div className="memory-items">
        {filteredData.map((item, index) => (
          activeSubTab === 'core-understanding' ? (
            // For core memory, don't add extra wrapper to avoid double containers
            <div key={index} id={`memory-${item.id}`}>
              {renderMemoryItem(item, activeSubTab, index)}
            </div>
          ) : (
            // For other memory types, keep the wrapper
            <div
              key={index}
              id={`memory-${item.id}`}
              className={`memory-item ${highlightedMemoryId === item.id ? 'highlighted' : ''}`}
            >
              {renderMemoryItem(item, activeSubTab, index)}
            </div>
          )
        ))}
      </div>
    );
  };

  // Helper function to navigate to a specific memory type and highlight an item
  const navigateToMemory = (memoryType, memoryId) => {
    const tabMap = {
      'episodic': 'past-events',
      'semantic': 'semantic',
      'procedural': 'procedural',
      'resource': 'docs-files',
      'knowledge_vault': 'credentials'
    };

    const targetTab = tabMap[memoryType] || memoryType;
    setHighlightedMemoryId(memoryId);
    setActiveSubTab(targetTab);
    setSearchQuery(memoryId);

    setTimeout(() => {
      const element = document.getElementById(`memory-${memoryId}`);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }, 300);
  };

  // Helper function to render memory reference badges
  const renderMemoryReferences = (references, itemId) => {
    if (!references || references.length === 0) return null;

    const getDomain = (url) => {
      if (!url) return null;
      try {
        const urlObj = new URL(url.startsWith('http') ? url : `https://${url}`);
        return urlObj.hostname || urlObj.pathname.split('/')[0];
      } catch {
        return url;
      }
    };

    const getAppIcon = (app) => {
      switch (app) {
        case 'Chrome': return '🌐';
        case 'Safari': return '🧭';
        case 'Firefox': return '🦊';
        case 'Notion': return '📝';
        default: return '💻';
      }
    };

    // Group references by source_app
    const groupedRefs = references.reduce((acc, ref) => {
      const app = ref.source_app || 'Unknown';
      if (!acc[app]) {
        acc[app] = [];
      }
      acc[app].push(ref);
      return acc;
    }, {});

    // Get unique URLs (deduplicate by URL)
    const getUniqueRefs = (refs) => {
      const urlMap = new Map();
      refs.forEach(ref => {
        const key = ref.source_url || ref.id;
        if (!urlMap.has(key)) {
          urlMap.set(key, { ...ref, count: 1, timestamps: [ref.captured_at] });
        } else {
          const existing = urlMap.get(key);
          existing.count += 1;
          existing.timestamps.push(ref.captured_at);
        }
      });
      return Array.from(urlMap.values());
    };

    // Create summary for collapsed state
    const appSummary = Object.entries(groupedRefs)
      .map(([app, refs]) => `${getAppIcon(app)} ${app} (${refs.length})`)
      .join(' • ');

    const isExpanded = expandedReferences.has(itemId);

    const toggleExpanded = () => {
      setExpandedReferences(prev => {
        const newSet = new Set(prev);
        if (newSet.has(itemId)) {
          newSet.delete(itemId);
        } else {
          newSet.add(itemId);
        }
        return newSet;
      });
    };

    const handleBadgeClick = (refId) => {
      // Jump to Raw Memory tab and highlight the item
      setHighlightedRawMemoryId(refId);
      setActiveSubTab('raw-memory');
      // Set search query to the id for easy filtering
      setSearchQuery(refId);

      // Scroll to the item after a short delay to allow tab switch
      setTimeout(() => {
        const element = document.getElementById(`raw-memory-${refId}`);
        if (element) {
          element.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }, 300);
    };

    // Helper function to navigate to a specific memory type and highlight an item
    const navigateToMemory = (memoryType, memoryId) => {
      const tabMap = {
        'episodic': 'past-events',
        'semantic': 'semantic',
        'procedural': 'procedural',
        'resource': 'docs-files',
        'knowledge_vault': 'credentials'
      };

      const targetTab = tabMap[memoryType] || memoryType;
      setHighlightedMemoryId(memoryId);
      setActiveSubTab(targetTab);
      setSearchQuery(memoryId);

      setTimeout(() => {
        const element = document.getElementById(`memory-${memoryId}`);
        if (element) {
          element.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }, 300);
    };

    const toggleAppGroup = (app) => {
      setExpandedAppGroups(prev => ({
        ...prev,
        [`${itemId}-${app}`]: !prev[`${itemId}-${app}`]
      }));
    };

    return (
      <div className="memory-references-section">
        <div
          className="memory-references-header clickable"
          onClick={toggleExpanded}
          style={{ cursor: 'pointer' }}
        >
          <span className="memory-icon">📚</span>
          <span className="memory-title">{t('chat.memoryReferences', { defaultValue: 'Memory References' })}</span>
          <span className="memory-count">{references.length}</span>
          <span className="expand-icon">{isExpanded ? '▲' : '▼'}</span>
        </div>

        {!isExpanded && (
          <div className="memory-references-summary">
            {appSummary}
          </div>
        )}

        {isExpanded && (
          <div className="memory-badges-grouped">
            {Object.entries(groupedRefs).map(([app, refs]) => {
              const uniqueRefs = getUniqueRefs(refs);
              const showAll = expandedAppGroups[`${itemId}-${app}`] || false;
              const displayRefs = showAll ? uniqueRefs : uniqueRefs.slice(0, 3);

              return (
                <div key={app} className="memory-app-group">
                  <div className="memory-app-group-header">
                    <span className="app-icon">{getAppIcon(app)}</span>
                    <span className="app-name">{app}</span>
                    <span className="app-count">({refs.length} {t('memory.references.items', { defaultValue: 'references' })})</span>
                  </div>
                  <div className="memory-badges">
                    {displayRefs.map((ref, index) => {
                      const domain = getDomain(ref.source_url);
                      const capturedDate = ref.captured_at ? new Date(ref.captured_at).toLocaleDateString() : null;

                      return (
                        <div
                          key={ref.id || index}
                          className="memory-badge"
                          onClick={() => handleBadgeClick(ref.id)}
                        >
                          <div className="memory-badge-content">
                            {domain && <div className="memory-badge-url">{domain}</div>}
                            {!domain && ref.source_url && <div className="memory-badge-url">{ref.source_url}</div>}
                            {!domain && !ref.source_url && <div className="memory-badge-no-url">{t('memory.references.noUrl', { defaultValue: 'No URL captured' })}</div>}
                            <div className="memory-badge-meta">
                              {capturedDate && <span className="memory-badge-date">{capturedDate}</span>}
                              {ref.count > 1 && <span className="memory-badge-versions"> • {ref.count} {t('memory.references.versions', { defaultValue: 'versions' })}</span>}
                            </div>
                          </div>
                          {ref.ocr_text && (
                            <div className="memory-badge-preview" title={ref.ocr_text}>
                              {ref.ocr_text.substring(0, 80)}...
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                  {uniqueRefs.length > 3 && (
                    <button
                      className="show-all-refs-button"
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleAppGroup(app);
                      }}
                    >
                      {showAll
                        ? t('memory.references.showLess', { defaultValue: 'Show less' })
                        : t('memory.references.showAll', { defaultValue: `Show all ${uniqueRefs.length} references` }, { count: uniqueRefs.length })
                      }
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    );
  };

  const renderMemoryItem = (item, type, index) => {
    switch (type) {
      case 'past-events':
        const episodicItemId = `episodic-${index}`;
        const isEpisodicExpanded = expandedItems.has(episodicItemId);
        return (
          <div className="episodic-memory">
            <div className="memory-timestamp">
              {item.timestamp ? new Date(item.timestamp).toLocaleString() : t('memory.details.unknownTime')}
            </div>
            <div className="memory-content">{highlightText(item.summary, searchQuery)}</div>
            {item.details && (
              <div className="memory-details-section">
                <button
                  className="expand-toggle-button"
                  onClick={() => toggleExpanded(episodicItemId)}
                  title={isEpisodicExpanded ? t('memory.actions.expandDetails') : t('memory.actions.collapseDetails')}
                >
                  {isEpisodicExpanded ? `▼ ${t('memory.actions.hideDetails')}` : `▶ ${t('memory.actions.showDetails')}`}
                </button>
                {isEpisodicExpanded && (
                  <>
                    <div className="memory-details">{highlightText(item.details, searchQuery)}</div>
                    {renderMemoryReferences(item.raw_memory_references, episodicItemId)}
                  </>
                )}
              </div>
            )}
          </div>
        );

      case 'semantic':
        const itemId = `semantic-${index}`;
        const isExpanded = expandedItems.has(itemId);
        return (
          <div className="semantic-memory">
            <div className="memory-title">{highlightText(item.title || item.name, searchQuery)}</div>
            {item.summary && <div className="memory-summary">{highlightText(item.summary, searchQuery)}</div>}
            {item.details && (
              <div className="memory-details-section">
                <button
                  className="expand-toggle-button"
                  onClick={() => toggleExpanded(itemId)}
                  title={isExpanded ? t('memory.actions.expandDetails') : t('memory.actions.collapseDetails')}
                >
                  {isExpanded ? `▼ ${t('memory.actions.hideDetails')}` : `▶ ${t('memory.actions.showDetails')}`}
                </button>
                {isExpanded && (
                  <>
                    <div className="memory-details">{highlightText(item.details, searchQuery)}</div>
                    {renderMemoryReferences(item.raw_memory_references, itemId)}
                  </>
                )}
              </div>
            )}
            {item.last_updated && <div className="memory-updated">{t('memory.details.updated', { date: new Date(item.last_updated).toLocaleString() })}</div>}
            {item.tags && (
              <div className="memory-tags">
                {item.tags.map((tag, i) => (
                  <span key={i} className="memory-tag">{highlightText(tag, searchQuery)}</span>
                ))}
              </div>
            )}
          </div>
        );

      case 'procedural':
        return (
          <div className="procedural-memory">
            <div className="memory-title">{highlightText(item.summary, searchQuery)}</div>
            <div className="memory-content">
              {item.steps && item.steps.length > 0 ? (
                <div className="memory-steps">
                  <strong>🎯 {t('memory.details.stepByStepGuide')}</strong>
                  <ol>
                    {item.steps.map((step, i) => (
                      <li key={i}>{highlightText(step, searchQuery)}</li>
                    ))}
                  </ol>
                </div>
              ) : (
                <div>{highlightText(item.content || item.description || t('memory.details.noStepsAvailable'), searchQuery)}</div>
              )}
            </div>
            {item.proficiency && <div className="memory-proficiency">{t('memory.details.proficiency', { value: highlightText(item.proficiency, searchQuery) })}</div>}
            {item.difficulty && <div className="memory-difficulty">{t('memory.details.difficulty', { value: highlightText(item.difficulty, searchQuery) })}</div>}
            {item.success_rate && <div className="memory-success-rate">{t('memory.details.successRate', { value: highlightText(item.success_rate, searchQuery) })}</div>}
            {item.time_to_complete && <div className="memory-time">{t('memory.details.timeToComplete', { value: highlightText(item.time_to_complete, searchQuery) })}</div>}
            {item.last_practiced && <div className="memory-practiced">{t('memory.details.lastPracticed', { date: new Date(item.last_practiced).toLocaleString() })}</div>}
            {item.prerequisites && item.prerequisites.length > 0 && (
              <div className="memory-prerequisites">
                {t('memory.details.prerequisites', { list: item.prerequisites.map(prereq => highlightText(prereq, searchQuery)).join(', ') })}
              </div>
            )}
            {item.last_updated && <div className="memory-updated">{t('memory.details.updated', { date: new Date(item.last_updated).toLocaleString() })}</div>}
            {item.tags && (
              <div className="memory-tags">
                {item.tags.map((tag, i) => (
                  <span key={i} className="memory-tag">{highlightText(tag, searchQuery)}</span>
                ))}
              </div>
            )}
            {renderMemoryReferences(item.raw_memory_references, `procedural-${index}`)}
          </div>
        );

      case 'docs-files':
        return (
          <div className="resource-memory">
            <div className="memory-filename">{highlightText(item.filename || item.name, searchQuery)}</div>
            <div className="memory-file-type">{highlightText(item.type || t('memory.details.unknownType'), searchQuery)}</div>
            <div className="memory-summary">{highlightText(item.summary || item.content, searchQuery)}</div>
            {item.last_accessed && (
              <div className="memory-accessed">{t('memory.details.lastAccessed', { date: new Date(item.last_accessed).toLocaleString() })}</div>
            )}
            {item.size && <div className="memory-size">{t('memory.details.size', { size: item.size })}</div>}
            {renderMemoryReferences(item.raw_memory_references, `resource-${index}`)}
          </div>
        );

      case 'core-understanding':
        const isEditing = isCoreMemoryEditing(index);
        const currentContent = getCoreMemoryContent(item, index);
        const isSaving = savingBlocks.has(index);
        const saveError = saveErrors[index];
        const saveSuccess = saveSuccesses[index];

        return (
          <div className="core-memory">
            <div className="memory-aspect-header">
              <div className="memory-aspect">
                {highlightText(item.aspect || item.category, searchQuery)}
                {item.total_characters && item.max_characters && (
                  <span className="character-count-inline"> ({t('memory.details.characterCount', { current: currentContent.length, max: item.max_characters })})</span>
                )}
                {isEditing && <span className="edited-indicator"> • {t('memory.details.editing')}</span>}
              </div>
              {!isEditing && (
                <button
                  onClick={() => startEditingCoreMemory(index)}
                  className="edit-memory-button"
                >
                  ✏️ {t('memory.actions.edit')}
                </button>
              )}
            </div>

            {isEditing ? (
              <div className="memory-understanding-editable">
                <textarea
                  value={currentContent}
                  onChange={(e) => handleCoreMemoryEdit(index, e.target.value)}
                  className="core-memory-textarea"
                  rows={Math.max(3, Math.ceil(currentContent.length / 80))}
                  placeholder={t('memory.details.enterCoreUnderstanding')}
                />
                <div className="core-memory-actions">
                  <button
                    onClick={() => saveCoreMemoryBlock(index, item)}
                    className="save-memory-button"
                    disabled={isSaving}
                  >
                    {isSaving ? `💾 ${t('memory.actions.saving')}` : `💾 ${t('memory.actions.save')}`}
                  </button>
                  <button
                    onClick={() => cancelEditingCoreMemory(index)}
                    className="cancel-memory-button"
                    disabled={isSaving}
                  >
                    ❌ {t('memory.actions.cancel')}
                  </button>
                </div>
              </div>
            ) : (
              <div className="memory-understanding-display">
                <div className="memory-understanding">
                  {renderTextWithNewlines(item.understanding || item.content, searchQuery)}
                </div>
              </div>
            )}

            {/* Status messages for individual blocks */}
            {saveSuccess && (
              <div className="block-save-status success">
                ✅ {t('memory.reflexion.success')}
              </div>
            )}
            {saveError && (
              <div className="block-save-status error">
                ❌ {t('memory.states.error', { error: saveError })}
              </div>
            )}

            {item.last_updated && (
              <div className="memory-updated">{t('memory.details.updated', { date: new Date(item.last_updated).toLocaleString() })}</div>
            )}
          </div>
        );

      case 'credentials':
        return (
          <div className="credential-memory">
            <div className="memory-credential-name">{highlightText(item.caption, searchQuery)}</div>
            <div className="memory-credential-type">{highlightText(item.entry_type || t('memory.details.credentialType'), searchQuery)}</div>
            <div className="memory-credential-content">
              {item.content || t('memory.details.credentialMasked')}
            </div>
            {item.source && (
              <div className="memory-credential-source">{t('memory.details.source', { source: highlightText(item.source, searchQuery) })}</div>
            )}
            {item.sensitivity && (
              <div className="memory-credential-sensitivity">
                <span className={`sensitivity-badge sensitivity-${item.sensitivity}`}>
                  {t('memory.details.sensitivity', { level: item.sensitivity.charAt(0).toUpperCase() + item.sensitivity.slice(1) })}
                </span>
              </div>
            )}
            {renderMemoryReferences(item.raw_memory_references, `credential-${index}`)}
          </div>
        );

      case 'raw-memory':
        const rawItemId = `raw-${index}`;
        const isRawExpanded = expandedItems.has(rawItemId);
        const isHighlighted = highlightedRawMemoryId === item.id;

        // Helper to get app icon
        const getAppIcon = (app) => {
          switch (app) {
            case 'Chrome': return '🌐';
            case 'Safari': return '🧭';
            case 'Firefox': return '🦊';
            case 'Notion': return '📝';
            default: return '💻';
          }
        };

        return (
          <div
            id={`raw-memory-${item.id}`}
            className={`raw-memory ${isHighlighted ? 'highlighted' : ''}`}
          >
            <div className="memory-app-header">
              <span className="memory-app-icon">{getAppIcon(item.source_app)}</span>
              <span className="memory-app-name">{highlightText(item.source_app, searchQuery)}</span>
            </div>
            <div className="memory-id-display">
              🆔 {item.id}
            </div>
            {item.source_url && (
              <div className="memory-source-url">
                🔗 <a href={item.source_url} target="_blank" rel="noopener noreferrer">
                  {highlightText(item.source_url, searchQuery)}
                </a>
              </div>
            )}
            {item.captured_at && (
              <div className="memory-timestamp">
                📅 {new Date(item.captured_at).toLocaleString()}
              </div>
            )}

            {/* Screenshot Preview */}
            {item.screenshot_url && (
              <div className="memory-screenshot-preview">
                <img
                  src={`${settings.serverUrl}${item.screenshot_url}`}
                  alt={`Screenshot from ${item.source_app}`}
                  className="screenshot-thumbnail"
                  onError={(e) => {
                    e.target.style.display = 'none';
                    const fallback = e.target.nextSibling;
                    if (fallback) fallback.style.display = 'block';
                  }}
                />
                <div className="screenshot-fallback" style={{ display: 'none' }}>
                  📸 Screenshot unavailable
                </div>
              </div>
            )}

            {/* OCR Preview/Full Text */}
            {(item.ocr_preview || item.ocr_text) && (
              <div className="memory-details-section">
                <div className="memory-ocr-preview">
                  {highlightText(item.ocr_preview || item.ocr_text, searchQuery)}
                </div>
                {item.ocr_text && item.ocr_text.length > 200 && (
                  <>
                    <button
                      className="expand-toggle-button"
                      onClick={() => toggleExpanded(rawItemId)}
                      title={isRawExpanded ? t('memory.actions.collapseDetails') : t('memory.actions.expandDetails')}
                    >
                      {isRawExpanded ? `▼ ${t('memory.actions.hideFullText', { defaultValue: 'Hide Full Text' })}` : `▶ ${t('memory.actions.showFullText', { defaultValue: 'Show Full Text' })}`}
                    </button>
                    {isRawExpanded && (
                      <div className="memory-ocr-full-text">{highlightText(item.ocr_text, searchQuery)}</div>
                    )}
                  </>
                )}
              </div>
            )}

            {item.processed !== undefined && (
              <div className="memory-processed-status">
                {item.processed ? '✅ Processed' : '⏳ Pending'}
              </div>
            )}
            <MemoryReferences
              rawMemoryId={item.id}
              serverUrl={settings.serverUrl}
              queuedFetch={queuedFetch}
              navigateToMemory={navigateToMemory}
            />
          </div>
        );

      default:
        return <div className="memory-content">{JSON.stringify(item, null, 2)}</div>;
    }
  };

  const getMemoryTypeLabel = (type) => {
    switch (type) {
      case 'past-events': return t('memory.types.episodic');
      case 'semantic': return t('memory.types.semantic');
      case 'procedural': return t('memory.types.procedural');
      case 'docs-files': return t('memory.types.resource');
      case 'raw-memory': return t('memory.types.raw', { defaultValue: 'Raw Memory' });
      case 'core-understanding': return t('memory.types.core');
      case 'credentials': return t('memory.types.credentials');
      default: return 'Memory';
    }
  };

  const getMemoryTypeIcon = (type) => {
    switch (type) {
      case 'past-events': return '📅';
      case 'semantic': return '🧠';
      case 'procedural': return '🛠️';
      case 'docs-files': return '📁';
      case 'raw-memory': return '📸';
      case 'core-understanding': return '💡';
      case 'credentials': return '🔐';
      default: return '💭';
    }
  };

  // Helper functions for core memory editing
  const startEditingCoreMemory = (index) => {
    setEditingCoreMemories(prev => new Set([...prev, index]));
    // Clear any previous save states for this block
    setSaveErrors(prev => {
      const newErrors = { ...prev };
      delete newErrors[index];
      return newErrors;
    });
    setSaveSuccesses(prev => {
      const newSuccesses = { ...prev };
      delete newSuccesses[index];
      return newSuccesses;
    });
  };

  const cancelEditingCoreMemory = (index) => {
    setEditingCoreMemories(prev => {
      const newSet = new Set(prev);
      newSet.delete(index);
      return newSet;
    });
    // Clear any edits for this block
    setEditedCoreMemories(prev => {
      const newEdited = { ...prev };
      delete newEdited[index];
      return newEdited;
    });
  };

  const handleCoreMemoryEdit = (index, newContent) => {
    setEditedCoreMemories(prev => ({
      ...prev,
      [index]: newContent
    }));
  };

  // Check if core memory is being edited
  const isCoreMemoryEditing = (index) => {
    return editingCoreMemories.has(index);
  };

  // Get the current content for a core memory (edited or original)
  const getCoreMemoryContent = (item, index) => {
    if (editedCoreMemories.hasOwnProperty(index)) {
      return editedCoreMemories[index];
    }
    return item.understanding || item.content || '';
  };

  // Save individual core memory block
  const saveCoreMemoryBlock = async (index, item) => {
    try {
      setSavingBlocks(prev => new Set([...prev, index]));
      setSaveErrors(prev => {
        const newErrors = { ...prev };
        delete newErrors[index];
        return newErrors;
      });
      setSaveSuccesses(prev => {
        const newSuccesses = { ...prev };
        delete newSuccesses[index];
        return newSuccesses;
      });

      const newContent = getCoreMemoryContent(item, index);
      const originalContent = item.understanding || item.content || '';

      if (newContent === originalContent) {
        // No changes, just exit edit mode
        setEditingCoreMemories(prev => {
          const newSet = new Set(prev);
          newSet.delete(index);
          return newSet;
        });
        return;
      }

      // Send update to server
      const response = await queuedFetch(`${settings.serverUrl}/core_memory/update`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          label: item.aspect || item.category,
          text: newContent
        })
      });

      if (!response.ok) {
        throw new Error(`Failed to update core memory: ${response.statusText}`);
      }

      // Success - clear edit state and refresh data
      setEditingCoreMemories(prev => {
        const newSet = new Set(prev);
        newSet.delete(index);
        return newSet;
      });
      setEditedCoreMemories(prev => {
        const newEdited = { ...prev };
        delete newEdited[index];
        return newEdited;
      });

      setSaveSuccesses(prev => ({ ...prev, [index]: true }));
      setTimeout(() => {
        setSaveSuccesses(prev => {
          const newSuccesses = { ...prev };
          delete newSuccesses[index];
          return newSuccesses;
        });
      }, 3000);

      // Refresh the data to show updated content
      await fetchMemoryData('core-understanding');

    } catch (err) {
      console.error('Error saving core memory block:', err);
      setSaveErrors(prev => ({ ...prev, [index]: err.message }));
    } finally {
      setSavingBlocks(prev => {
        const newSet = new Set(prev);
        newSet.delete(index);
        return newSet;
      });
    }
  };

  // Handle reflexion request
  const handleReflexion = async () => {
    if (isReflexionProcessing) return; // Prevent multiple simultaneous requests

    try {
      setIsReflexionProcessing(true);
      setReflexionMessage('');
      setReflexionSuccess(null);

      console.log('Starting reflexion process...');

      const response = await queuedFetch(`${settings.serverUrl}/reflexion`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({})
      });

      if (!response.ok) {
        throw new Error(`Failed to trigger reflexion: ${response.statusText}`);
      }

      const result = await response.json();

      if (result.success) {
        setReflexionSuccess(true);
        setReflexionMessage(result.message);
        console.log('Reflexion completed successfully:', result.message);

        // Optionally refresh memory data after reflexion
        // You can uncomment this if you want to refresh the current tab's data
        // await fetchMemoryData(activeSubTab);
      } else {
        setReflexionSuccess(false);
        setReflexionMessage(result.message || 'Reflexion failed');
        console.error('Reflexion failed:', result.message);
      }

    } catch (err) {
      console.error('Error triggering reflexion:', err);
      setReflexionSuccess(false);
      setReflexionMessage(err.message || 'Failed to trigger reflexion');
    } finally {
      setIsReflexionProcessing(false);

      // Clear the message after 5 seconds
      setTimeout(() => {
        setReflexionMessage('');
        setReflexionSuccess(null);
      }, 5000);
    }
  };

  return (
    <div className="existing-memory">
      <div className="memory-header">
        <div className="memory-subtabs">
          <div className="memory-subtabs-left">
            {['past-events', 'semantic', 'procedural', 'docs-files', 'raw-memory', 'core-understanding', 'credentials'].map(subTab => (
              <button
                key={subTab}
                className={`memory-subtab ${activeSubTab === subTab ? 'active' : ''}`}
                onClick={() => setActiveSubTab(subTab)}
              >
                <span className="subtab-icon">{getMemoryTypeIcon(subTab)}</span>
                <span className="subtab-label">{getMemoryTypeLabel(subTab)}</span>
              </button>
            ))}
          </div>
          <div className="memory-subtabs-right">
            <button
              className="memory-subtab upload-export-btn"
              onClick={() => setShowUploadExportModal(true)}
              title={t('memory.tooltips.uploadExport')}
            >
              <span className="subtab-icon">📤</span>
              <span className="subtab-label">{t('memory.actions.uploadExport')}</span>
            </button>
            <button
              className="memory-subtab reflexion-btn"
              onClick={handleReflexion}
              disabled={isReflexionProcessing}
              title={t('memory.tooltips.reflexion')}
            >
              <span className="subtab-icon">
                {isReflexionProcessing ? '⏳' : '🧠'}
              </span>
              <span className="subtab-label">
                {isReflexionProcessing ? t('memory.actions.processing') : t('memory.actions.reflexion')}
              </span>
            </button>
          </div>
        </div>
      </div>

      <div className="memory-content">
        <div className="memory-search-and-actions">
          <div className="search-input-container">
            <span className="search-icon">🔍</span>
            <input
              type="text"
              placeholder={t('memory.search.placeholder', { type: getMemoryTypeLabel(activeSubTab).toLowerCase() })}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="search-input"
              disabled={getCurrentViewMode() === 'tree'}
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="clear-search-button"
                title={t('memory.actions.clearSearch')}
              >
                ✕
              </button>
            )}
          </div>

          {['past-events', 'semantic'].includes(activeSubTab) && (
            <div className="reference-filter-toggle">
              <button
                onClick={() => setShowOnlyReferenced(!showOnlyReferenced)}
                className={`filter-toggle-button ${showOnlyReferenced ? 'active' : ''}`}
                title={t('memory.tooltips.filterReferenced', { defaultValue: 'Show only memories with references' })}
              >
                📚 {showOnlyReferenced ? t('memory.filter.showAll', { defaultValue: 'Show All' }) : t('memory.filter.onlyReferenced', { defaultValue: 'Only Referenced' })}
              </button>
            </div>
          )}

          {activeSubTab === 'raw-memory' && (
            <div className="reference-filter-toggle">
              <button
                onClick={() => setShowOnlyReferencedRaw(!showOnlyReferencedRaw)}
                className={`filter-toggle-button ${showOnlyReferencedRaw ? 'active' : ''}`}
                title={t('memory.tooltips.filterReferencedRaw', { defaultValue: 'Show only referenced raw memories' })}
              >
                🔗 {showOnlyReferencedRaw ? t('memory.filter.showAll', { defaultValue: 'Show All' }) : t('memory.filter.onlyReferenced', { defaultValue: 'Only Referenced' })}
              </button>
            </div>
          )}

          {['past-events', 'semantic', 'procedural', 'docs-files'].includes(activeSubTab) && (
            <div className="view-mode-toggle">
              <button
                onClick={() => setCurrentViewMode('list')}
                className={`view-mode-button ${getCurrentViewMode() === 'list' ? 'active' : ''}`}
                title={t('memory.tooltips.listView')}
              >
                📋 {t('memory.view.listView')}
              </button>
              <button
                onClick={() => setCurrentViewMode('tree')}
                className={`view-mode-button ${getCurrentViewMode() === 'tree' ? 'active' : ''}`}
                title={t('memory.tooltips.treeView')}
              >
                🌳 {t('memory.view.treeView')}
              </button>
            </div>
          )}



          <button
            onClick={() => fetchMemoryData(activeSubTab)}
            className="refresh-button"
            disabled={loading}
          >
            🔄 {t('memory.actions.refresh')}
          </button>
        </div>

        {/* Reflexion Status Message */}
        {reflexionMessage && (
          <div className={`reflexion-status ${reflexionSuccess ? 'success' : 'error'}`}>
            <span className="status-icon">
              {reflexionSuccess ? '✅' : '❌'}
            </span>
            <span className="status-message">{reflexionMessage}</span>
          </div>
        )}


        {renderMemoryContent()}
      </div>

      {/* Upload & Export Modal */}
      <UploadExportModal
        isOpen={showUploadExportModal}
        onClose={() => setShowUploadExportModal(false)}
        settings={settings}
      />
    </div>
  );
};

export default ExistingMemory; 