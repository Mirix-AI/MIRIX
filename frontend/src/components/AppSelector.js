import React, { useState, useEffect } from 'react';
import './AppSelector.css';
import { useTranslation } from 'react-i18next';

const AppSelector = ({ onSourcesSelected, onClose }) => {
  const { t } = useTranslation();
  const [sources, setSources] = useState([]);
  const [selectedSources, setSelectedSources] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState('all'); // 'all', 'windows', 'screens'

  useEffect(() => {
    loadSources();
  }, []);

  const loadSources = async () => {
    if (!window.electronAPI || !window.electronAPI.getCaptureSources) {
      setError(t('appSelector.errors.desktopOnly'));
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      const result = await window.electronAPI.getCaptureSources();
      
      if (result.success) {
        setSources(result.sources);
      } else {
        setError(result.error || t('appSelector.errors.failedToLoad'));
      }
    } catch (err) {
      setError(t('appSelector.errors.loadError', { error: err.message }));
    } finally {
      setLoading(false);
    }
  };

  const toggleSource = (sourceId) => {
    setSelectedSources(prev => {
      if (prev.includes(sourceId)) {
        return prev.filter(id => id !== sourceId);
      } else {
        return [...prev, sourceId];
      }
    });
  };

  const handleConfirm = () => {
    const selected = sources.filter(source => selectedSources.includes(source.id));
    onSourcesSelected(selected);
    onClose();
  };

  // Group sources by application and filter out duplicate app windows
  const groupSourcesByApp = (sources) => {
    const appGroups = new Map();
    
    sources.forEach(source => {
      if (source.type === 'screen') {
        // Keep all screens as-is
        appGroups.set(source.id, source);
        return;
      }
      
      // Extract app name from window title
      let appName = source.name;
      
      // Microsoft Teams specific patterns
      if (source.name.includes('Microsoft Teams') || 
          source.name.includes('MSTeams') || 
          (source.name.includes('Chat |') && source.name.includes('| Microsoft Teams'))) {
        appName = 'Microsoft Teams';
      } 
      // WeChat specific patterns
      else if (source.name.includes('WeChat') || source.name.includes('微信')) {
        appName = 'WeChat';
      }
      // Slack specific patterns
      else if (source.name.includes('Slack')) {
        appName = 'Slack';
      }
      // Chrome specific patterns
      else if (source.name.includes('Google Chrome') || source.name.endsWith(' - Chrome')) {
        appName = 'Google Chrome';
      }
      // Safari specific patterns
      else if (source.name.includes('Safari') || source.name.endsWith(' — Safari')) {
        appName = 'Safari';
      }
      // Visual Studio Code
      else if (source.name.includes('Visual Studio Code') || source.name.endsWith(' - Code')) {
        appName = 'Visual Studio Code';
      }
      // Terminal/iTerm
      else if (source.name.includes('Terminal') || source.name.includes('iTerm')) {
        appName = source.name.includes('iTerm') ? 'iTerm' : 'Terminal';
      }
      // For other apps, try to extract from window title more carefully
      else if (source.name.includes(' — ')) {
        // For apps that use em dash separator (like many Mac apps)
        // Take the last part, but only if it looks like an app name (not too long)
        const lastPart = source.name.split(' — ').pop();
        if (lastPart && lastPart.length < 30) {
          appName = lastPart;
        }
      } else if (source.name.includes(' - ')) {
        // For apps that use regular dash separator
        // Be more careful - only take the last part if it's likely an app name
        const parts = source.name.split(' - ');
        const lastPart = parts[parts.length - 1];
        
        // Check if the last part looks like an app name (starts with capital, not too long, etc.)
        if (lastPart && 
            lastPart.length < 30 && 
            /^[A-Z]/.test(lastPart) &&
            !lastPart.includes('.') && // Not a filename
            !lastPart.includes('/') && // Not a path
            !lastPart.match(/^\d/)) {  // Doesn't start with a number
          appName = lastPart;
        }
      }
      
      // Final cleanup - if appName is still the full window title and it's very long,
      // just use the first part before any separator
      if (appName === source.name && appName.length > 50) {
        const firstPart = appName.split(/[\-—]/)[0].trim();
        if (firstPart && firstPart.length < 30) {
          appName = firstPart;
        }
      }
      
      // If we already have this app, prefer the main window over sub-windows
      const existingSource = appGroups.get(appName);
      
      if (!existingSource) {
        // First window for this app
        appGroups.set(appName, { ...source, appName });
      } else {
        // Special handling for Microsoft Teams windows
        if (appName === 'Microsoft Teams') {
          const isCurrentMSTeams = source.name.includes('MSTeams');
          const isExistingMSTeams = existingSource.name.includes('MSTeams');
          const isCurrentChat = source.name.includes('Chat |');
          const isExistingChat = existingSource.name.includes('Chat |');
          
          if (isCurrentMSTeams && !isExistingMSTeams) {
            // Current is MSTeams main window, prefer it over chat windows
            appGroups.set(appName, { ...source, appName });
          } else if (!isCurrentMSTeams && isExistingMSTeams) {
            // Existing is MSTeams main window, keep it
          } else if (isCurrentChat && !isExistingChat) {
            // Current is chat, existing is something else - prefer chat over generic
            appGroups.set(appName, { ...source, appName });
          } else {
            // Default: prefer shorter name
            if (source.name.length < existingSource.name.length) {
              appGroups.set(appName, { ...source, appName });
            }
          }
        } else {
          // Original logic for non-Teams apps
          const isCurrentMainWindow = source.name === appName || source.name.endsWith(appName);
          const isExistingMainWindow = existingSource.name === appName || existingSource.name.endsWith(appName);
          
          if (isCurrentMainWindow && !isExistingMainWindow) {
            // Current is main window, existing is sub-window - replace
            appGroups.set(appName, { ...source, appName });
          } else if (!isCurrentMainWindow && !isExistingMainWindow) {
            // Both are sub-windows - prefer shorter name (usually more general)
            if (source.name.length < existingSource.name.length) {
              appGroups.set(appName, { ...source, appName });
            }
          }
        }
      }
    });
    
    const result = Array.from(appGroups.values());
    return result;
  };

  const filteredSources = groupSourcesByApp(sources).filter(source => {
    if (filter === 'all') return true;
    if (filter === 'windows') return source.type === 'window';
    if (filter === 'screens') return source.type === 'screen';
    return true;
  });

  return (
    <div className="app-selector-overlay" onClick={onClose}>
      <div className="app-selector-modal" onClick={(e) => e.stopPropagation()}>
        <div className="app-selector-header">
          <h2>{t('appSelector.title')}</h2>
          <button className="close-button" onClick={onClose}>×</button>
        </div>

        {error && (
          <div className="app-selector-error">
            ⚠️ {error}
          </div>
        )}

        {loading ? (
          <div className="app-selector-loading">
            <div className="spinner"></div>
            <p>{t('appSelector.loading')}</p>
          </div>
        ) : (
          <>
            <div className="app-selector-filters">
              <button 
                className={`filter-button ${filter === 'all' ? 'active' : ''}`}
                onClick={() => setFilter('all')}
              >
                {t('appSelector.filters.all')}
              </button>
              <button 
                className={`filter-button ${filter === 'windows' ? 'active' : ''}`}
                onClick={() => setFilter('windows')}
              >
                {t('appSelector.filters.windows')}
              </button>
              <button 
                className={`filter-button ${filter === 'screens' ? 'active' : ''}`}
                onClick={() => setFilter('screens')}
              >
                {t('appSelector.filters.screens')}
              </button>
            </div>

            <div className="app-selector-grid">
              {filteredSources.map(source => (
                <div 
                  key={source.id}
                  className={`app-selector-item ${selectedSources.includes(source.id) ? 'selected' : ''}`}
                  onClick={() => toggleSource(source.id)}
                >
                  <div className="app-thumbnail">
                    <img src={source.thumbnail} alt={source.name} />
                    {selectedSources.includes(source.id) && (
                      <div className="selected-overlay">
                        <div className="checkmark">✓</div>
                      </div>
                    )}
                  </div>
                  <div className="app-info">
                    {source.appIcon && (
                      <img className="app-icon" src={source.appIcon} alt="" />
                    )}
                    <span className="app-name" title={source.name}>{source.name}</span>
                    <div className="app-badges">
                      <span className="app-type">{t(`appSelector.types.${source.type}`)}</span>
                      {source.isVirtual && !source.isVisible && (
                        <span className="app-status" title={t('appSelector.status.hiddenTooltip')}>
                          {t('appSelector.status.hidden')}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="app-selector-footer">
              <div className="selection-info">
                {t('appSelector.footer.sourcesSelected', { count: selectedSources.length })}
              </div>
              <div className="action-buttons">
                <button className="cancel-button" onClick={onClose}>
                  {t('appSelector.footer.cancel')}
                </button>
                <button 
                  className="confirm-button" 
                  onClick={handleConfirm}
                  disabled={selectedSources.length === 0}
                >
                  {t('appSelector.footer.startMonitoring')}
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default AppSelector;