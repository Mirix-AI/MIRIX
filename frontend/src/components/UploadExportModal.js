import React, { useState } from 'react';
import queuedFetch from '../utils/requestQueue';
import './UploadExportModal.css';
import { useTranslation } from 'react-i18next';

function UploadExportModal({ isOpen, onClose, settings }) {
  const { t } = useTranslation();
  const [selectedMemoryTypes, setSelectedMemoryTypes] = useState({
    episodic: true,
    semantic: true,
    procedural: true,
    resource: true
  });
  const [exportPath, setExportPath] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [exportStatus, setExportStatus] = useState(null);

  const memoryTypes = [
    { key: 'episodic', label: t('uploadExport.memoryTypes.episodic'), icon: '📚', description: t('uploadExport.memoryTypeDescriptions.episodic') },
    { key: 'semantic', label: t('uploadExport.memoryTypes.semantic'), icon: '🧠', description: t('uploadExport.memoryTypeDescriptions.semantic') },
    { key: 'procedural', label: t('uploadExport.memoryTypes.procedural'), icon: '🔧', description: t('uploadExport.memoryTypeDescriptions.procedural') },
    { key: 'resource', label: t('uploadExport.memoryTypes.resource'), icon: '📁', description: t('uploadExport.memoryTypeDescriptions.resource') }
  ];

  const handleMemoryTypeToggle = (type) => {
    setSelectedMemoryTypes(prev => ({
      ...prev,
      [type]: !prev[type]
    }));
  };

  const handleBrowse = async () => {
    if (window.electronAPI && window.electronAPI.selectSavePath) {
      try {
        const result = await window.electronAPI.selectSavePath({
          title: t('uploadExport.descriptions.saveDialogTitle'),
          defaultName: t('uploadExport.descriptions.defaultFileName')
        });
        
        if (!result.canceled && result.filePath) {
          setExportPath(result.filePath);
        }
      } catch (error) {
        console.error('Error opening file dialog:', error);
        alert(t('uploadExport.alerts.browserFailed'));
      }
    } else {
      alert(t('uploadExport.alerts.browserUnavailable'));
    }
  };

  const handleUpload = () => {
    alert(t('uploadExport.alerts.uploadNotImplemented'));
  };

  const handleExport = async () => {
    if (!exportPath.trim()) {
      alert(t('uploadExport.alerts.pathRequired'));
      return;
    }

    const selectedTypes = Object.keys(selectedMemoryTypes).filter(
      type => selectedMemoryTypes[type]
    );

    if (selectedTypes.length === 0) {
      alert(t('uploadExport.alerts.selectTypes'));
      return;
    }

    setIsLoading(true);
    setExportStatus(null);

    try {
      const response = await queuedFetch(`${settings.serverUrl}/export/memories`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          file_path: exportPath,
          memory_types: selectedTypes,
          include_embeddings: false
        }),
      });

      if (response.ok) {
        const result = await response.json();
        setExportStatus({
          success: true,
          message: result.message,
          counts: result.exported_counts,
          total: result.total_exported
        });
      } else {
        const errorData = await response.json();
        const detail = String(errorData?.detail || '');
        const localized =
          detail.includes('At least one sheet must be visible') ? t('uploadExport.errors.atLeastOneSheetVisible') :
          detail.includes('No data') ? t('uploadExport.errors.noData') :
          detail.toLowerCase().includes('permission') ? t('uploadExport.errors.permissionDenied') :
          t('uploadExport.errors.unknown');

        setExportStatus({ success: false, message: localized });
      }
    } catch (error) {
      console.error('Export error:', error);
      setExportStatus({
        success: false,
        message: `${t('uploadExport.status.failed')}: ${error.message}`
      });
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="upload-export-modal-overlay" onClick={onClose}>
      <div className="upload-export-modal" onClick={(e) => e.stopPropagation()}>
        <div className="upload-export-modal-header">
          <h2>📤 {t('uploadExport.title')}</h2>
          <button 
            className="upload-export-modal-close"
            onClick={onClose}
            title={t('uploadExport.form.close')}
          >
            ✕
          </button>
        </div>
        
        <div className="upload-export-modal-content">
          <div className="upload-export-modal-description">
            <p>{t('uploadExport.descriptions.modalDescription')}</p>
          </div>

          <div className="memory-types-section">
            <h3>{t('uploadExport.form.selectTypes')}</h3>
            <div className="memory-types-grid">
              {memoryTypes.map(type => (
                <div 
                  key={type.key}
                  className={`memory-type-card ${selectedMemoryTypes[type.key] ? 'selected' : ''}`}
                  onClick={() => handleMemoryTypeToggle(type.key)}
                >
                  <div className="memory-type-icon">{type.icon}</div>
                  <div className="memory-type-info">
                    <div className="memory-type-label">{type.label}</div>
                    <div className="memory-type-description">{type.description}</div>
                  </div>
                  <div className="memory-type-checkbox">
                    <input 
                      type="checkbox" 
                      checked={selectedMemoryTypes[type.key]}
                      onChange={() => {}}
                      readOnly
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="actions-section">
            <div className="upload-section">
              <h3>{t('uploadExport.sections.upload')}</h3>
              <p>{t('uploadExport.descriptions.uploadSection')}</p>
              <button 
                className="upload-btn"
                onClick={handleUpload}
              >
                📤 {t('uploadExport.form.upload')}
              </button>
            </div>

            <div className="export-section">
              <h3>{t('uploadExport.sections.export')}</h3>
              <p>{t('uploadExport.descriptions.exportSection')}</p>
              
              <div className="export-path-input">
                <label htmlFor="exportPath">{t('uploadExport.form.exportPath')}</label>
                <div className="path-input-group">
                  <input
                    id="exportPath"
                    type="text"
                    value={exportPath}
                    onChange={(e) => setExportPath(e.target.value)}
                    placeholder={t('uploadExport.form.pathPlaceholder')}
                    className="path-input"
                  />
                  <button 
                    type="button"
                    className="browse-btn"
                    onClick={handleBrowse}
                    title={t('uploadExport.form.browse')}
                  >
                    📁 {t('uploadExport.form.browse')}
                  </button>
                </div>
              </div>

              <button 
                className="export-btn"
                onClick={handleExport}
                disabled={isLoading}
              >
                {isLoading ? `⏳ ${t('uploadExport.form.exporting')}` : `📥 ${t('uploadExport.form.export')}`}
              </button>

              {exportStatus && (
                <div className={`export-status ${exportStatus.success ? 'success' : 'error'}`}>
                  <div className="status-message">{exportStatus.message}</div>
                  {exportStatus.success && exportStatus.counts && (
                    <div className="export-details">
                      <div className="total-exported">{t('uploadExport.status.exported', { total: exportStatus.total })}</div>
                      <div className="counts-breakdown">
                        {Object.entries(exportStatus.counts).map(([type, count]) => (
                          <span key={type} className="count-item">
                            {type === 'episodic' ? t('uploadExport.memoryTypes.episodic')
                              : type === 'semantic' ? t('uploadExport.memoryTypes.semantic')
                              : type === 'procedural' ? t('uploadExport.memoryTypes.procedural')
                              : type === 'resource' ? t('uploadExport.memoryTypes.resource')
                              : type}: {count}
                          </span>
                        ))}
                      </div>  
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default UploadExportModal; 