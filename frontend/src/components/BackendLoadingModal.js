import React, { useState, useEffect } from 'react';
import './BackendLoadingModal.css';

const BackendLoadingModal = ({ isVisible, onRetry, isReconnection = false }) => {
  const [dots, setDots] = useState('');
  const [retryCount, setRetryCount] = useState(0);

  // Animate loading dots
  useEffect(() => {
    if (isVisible) {
      const interval = setInterval(() => {
        setDots(prev => {
          if (prev === '...') return '';
          return prev + '.';
        });
      }, 500);
      return () => clearInterval(interval);
    }
  }, [isVisible]);

  // Reset state when modal becomes visible
  useEffect(() => {
    if (isVisible) {
      setRetryCount(0);
      setDots('');
    }
  }, [isVisible]);

  const handleRetry = () => {
    setRetryCount(prev => prev + 1);
    setDots('');
    if (onRetry) {
      onRetry();
    }
  };

  if (!isVisible) return null;

  return (
    <div className="backend-loading-modal-overlay">
      <div className="backend-loading-modal">
        <div className="loading-content">
          <h3>Loading Memory Systems{dots}</h3>
          
          {retryCount > 0 && (
            <div className="retry-section">
              <p className="retry-text">
                Still connecting... This may take up to 60 seconds.
              </p>
              <button 
                className="retry-button"
                onClick={handleRetry}
                title="Force retry connection"
              >
                🔄 Retry
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default BackendLoadingModal; 