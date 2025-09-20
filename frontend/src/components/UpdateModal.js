import React from 'react';
import './UpdateModal.css';

const UpdateModal = ({ 
  isOpen, 
  onClose, 
  updateInfo, 
  currentVersion, 
  onUpdate
}) => {
  if (!isOpen) return null;

  const formatDate = (dateString) => {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric' 
    });
  };

  const formatReleaseNotes = (notes) => {
    if (!notes) return 'No release notes available';
    
    const lines = notes.split('\n').slice(0, 10);
    if (notes.split('\n').length > 10) {
      lines.push('...');
    }
    return lines.join('\n');
  };

  return (
    <div className="update-modal-overlay" onClick={onClose}>
      <div className="update-modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="update-modal-header">
          <h3>🎉 Update Available!</h3>
          <button className="update-modal-close" onClick={onClose}>
            ×
          </button>
        </div>
        
        <div className="update-modal-body">
          <div className="version-info">
            <div className="version-row">
              <span className="version-label">Current version:</span>
              <span className="version-value current">{currentVersion}</span>
            </div>
            <div className="version-row">
              <span className="version-label">New version:</span>
              <span className="version-value new">{updateInfo?.version}</span>
            </div>
            {updateInfo?.releaseDate && (
              <div className="version-row">
                <span className="version-label">Released:</span>
                <span className="version-value">{formatDate(updateInfo.releaseDate)}</span>
              </div>
            )}
          </div>

          <div className="release-notes">
            <h4>What's New:</h4>
            <pre>{formatReleaseNotes(updateInfo?.releaseNotes)}</pre>
          </div>

          <div className="update-actions">
            <button 
              className="update-btn-primary"
              onClick={onUpdate}
            >
              Download Update
            </button>
            <button 
              className="update-btn-secondary"
              onClick={onClose}
            >
              Later
            </button>
          </div>

          <div className="update-note">
            <span className="info-icon">ℹ️</span>
            You will be redirected to the download page. Please install the update manually.
          </div>
        </div>
      </div>
    </div>
  );
};

export default UpdateModal;