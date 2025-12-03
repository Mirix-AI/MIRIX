import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';

const MemoryReferences = ({
    rawMemoryId,
    serverUrl,
    queuedFetch,
    navigateToMemory
}) => {
    const { t } = useTranslation();
    const [references, setReferences] = useState(null);
    const [isExpanded, setIsExpanded] = useState(false);
    const [isLoading, setIsLoading] = useState(false);

    const fetchReferences = async () => {
        if (references) return;

        setIsLoading(true);
        try {
            const response = await queuedFetch(`${serverUrl}/memory/raw/${rawMemoryId}/references`);
            if (!response.ok) {
                throw new Error(`Failed to fetch references: ${response.statusText}`);
            }
            const data = await response.json();
            setReferences(data);
        } catch (err) {
            console.error(`Error fetching reverse references for ${rawMemoryId}:`, err);
        } finally {
            setIsLoading(false);
        }
    };

    const toggleExpanded = () => {
        if (!isExpanded && !references) {
            fetchReferences();
        }
        setIsExpanded(!isExpanded);
    };

    const getMemoryTypeIcon = (type) => {
        switch (type) {
            case 'episodic': return '📅';
            case 'semantic': return '🧠';
            case 'procedural': return '🛠️';
            case 'resource': return '📁';
            case 'knowledge_vault': return '🔐';
            default: return '💭';
        }
    };

    const getMemoryTypeLabel = (type) => {
        switch (type) {
            case 'episodic': return t('memory.types.episodic', { defaultValue: 'Episodic' });
            case 'semantic': return t('memory.types.semantic', { defaultValue: 'Semantic' });
            case 'procedural': return t('memory.types.procedural', { defaultValue: 'Procedural' });
            case 'resource': return t('memory.types.resource', { defaultValue: 'Resource' });
            case 'knowledge_vault': return t('memory.types.credentials', { defaultValue: 'Credentials' });
            default: return 'Memory';
        }
    };

    if (!isExpanded) {
        return (
            <div className="memory-references-section">
                <div
                    className="memory-references-header clickable"
                    onClick={toggleExpanded}
                    style={{ cursor: 'pointer' }}
                >
                    <span className="memory-icon">🔗</span>
                    <span className="memory-title">{t('memory.reverseReferences', { defaultValue: 'Referenced By' })}</span>
                    {references && <span className="memory-count">{references.total_count}</span>}
                    <span className="expand-icon">▼</span>
                </div>
            </div>
        );
    }

    if (isLoading && !references) {
        return (
            <div className="memory-references-section">
                <div
                    className="memory-references-header clickable"
                    onClick={toggleExpanded}
                    style={{ cursor: 'pointer' }}
                >
                    <span className="memory-icon">🔗</span>
                    <span className="memory-title">{t('memory.reverseReferences', { defaultValue: 'Referenced By' })}</span>
                    <span className="expand-icon">▲</span>
                </div>
                <div className="memory-loading">Loading references...</div>
            </div>
        );
    }

    if (references && references.total_count === 0) {
        return (
            <div className="memory-references-section">
                <div
                    className="memory-references-header clickable"
                    onClick={toggleExpanded}
                    style={{ cursor: 'pointer' }}
                >
                    <span className="memory-icon">🔗</span>
                    <span className="memory-title">{t('memory.reverseReferences', { defaultValue: 'Referenced By' })}</span>
                    <span className="memory-count">0</span>
                    <span className="expand-icon">▲</span>
                </div>
                <div className="memory-no-references">No references found.</div>
            </div>
        );
    }

    return (
        <div className="memory-references-section">
            <div
                className="memory-references-header clickable"
                onClick={toggleExpanded}
                style={{ cursor: 'pointer' }}
            >
                <span className="memory-icon">🔗</span>
                <span className="memory-title">{t('memory.reverseReferences', { defaultValue: 'Referenced By' })}</span>
                <span className="memory-count">{references ? references.total_count : ''}</span>
                <span className="expand-icon">▲</span>
            </div>

            {references && (
                <div className="memory-badges-grouped">
                    {Object.entries(references.references).map(([memoryType, refs]) => {
                        if (refs.length === 0) return null;

                        return (
                            <div key={memoryType} className="memory-app-group">
                                <div className="memory-app-group-header">
                                    <span className="app-icon">{getMemoryTypeIcon(memoryType)}</span>
                                    <span className="app-name">{getMemoryTypeLabel(memoryType)}</span>
                                    <span className="app-count">({refs.length})</span>
                                </div>
                                <div className="memory-badges">
                                    {refs.map((ref, index) => (
                                        <div
                                            key={ref.id || index}
                                            className="memory-badge"
                                            onClick={() => navigateToMemory(memoryType, ref.id)}
                                        >
                                            <div className="memory-badge-content">
                                                <div className="memory-badge-url">{ref.title}</div>
                                                <div className="memory-badge-meta">
                                                    {ref.timestamp && (
                                                        <span className="memory-badge-date">
                                                            {new Date(ref.timestamp).toLocaleDateString()}
                                                        </span>
                                                    )}
                                                </div>
                                            </div>
                                            {ref.summary && (
                                                <div className="memory-badge-preview" title={ref.summary}>
                                                    {ref.summary.substring(0, 80)}...
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
};

export default MemoryReferences;
