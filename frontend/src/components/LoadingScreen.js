import React, { useState, useEffect } from 'react';
import './LoadingScreen.css';
import Logo from './Logo';

const LoadingScreen = ({ message, status = 'loading', error = null, onRetry = null, phase = 'starting' }) => {
  const [dots, setDots] = useState('');
  const [progress, setProgress] = useState(0);

  // Phase-based progress mapping
  const phaseProgress = {
    starting: 10,
    server: 20,
    database: 35,
    cloud: 50,
    cloud_sync: 65,
    agent: 80,
    agent_ready: 90,
    health_check: 95,
    ready: 100
  };

  // Update progress based on phase
  useEffect(() => {
    if (status === 'loading' && phase in phaseProgress) {
      setProgress(phaseProgress[phase]);
    }
  }, [phase, status]);

  // Animate loading dots
  useEffect(() => {
    if (status === 'loading') {
      const interval = setInterval(() => {
        setDots(prev => {
          if (prev === '...') return '';
          return prev + '.';
        });
      }, 500);
      return () => clearInterval(interval);
    }
  }, [status]);

  const getIcon = () => {
    switch (status) {
      case 'loading':
        return '🔄';
      case 'error':
        return '❌';
      case 'success':
        return '✅';
      default:
        return '🔥';
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'loading':
        return '#3498db';
      case 'error':
        return '#e74c3c';
      case 'success':
        return '#27ae60';
      default:
        return '#3498db';
    }
  };

  const getStepStatus = (stepPhase) => {
    const stepOrder = ['server', 'database', 'cloud', 'agent'];
    const currentIndex = stepOrder.indexOf(phase);
    const stepIndex = stepOrder.indexOf(stepPhase);
    
    if (stepIndex < currentIndex) return 'completed';
    if (stepIndex === currentIndex) return 'active';
    return 'pending';
  };

  return (
    <div className="loading-screen">
      <div className="loading-container">
        <div className="loading-logo">
          <Logo 
            size="large" 
            showText={true} 
            textColor="#2c3e50"
          />
          <span className="loading-subtitle">AI Assistant</span>
        </div>

        <div className="loading-content">
          <div className="loading-icon" style={{ color: getStatusColor() }}>
            {getIcon()}
          </div>
          
          <div className="loading-message">
            {status === 'loading' && (
              <>
                <h3>{message}{dots}</h3>
                <div className="loading-progress">
                  <div className="progress-bar">
                    <div 
                      className="progress-fill" 
                      style={{ 
                        width: `${progress}%`,
                        backgroundColor: getStatusColor()
                      }}
                    />
                  </div>
                  <span className="progress-text">{Math.round(progress)}%</span>
                </div>
                <div className="loading-steps">
                  <div className={`step ${getStepStatus('server')}`}>
                    🔌 Connecting to backend...
                    {getStepStatus('server') === 'completed' && <span className="step-check">✅</span>}
                    {getStepStatus('server') === 'active' && <span className="step-spinner">🔄</span>}
                  </div>
                  <div className={`step ${getStepStatus('database')}`}>
                    🗄️ Initializing database...
                    {getStepStatus('database') === 'completed' && <span className="step-check">✅</span>}
                    {getStepStatus('database') === 'active' && <span className="step-spinner">🔄</span>}
                  </div>
                  <div className={`step ${getStepStatus('cloud')}`}>
                    ☁️ Connecting to cloud services...
                    {getStepStatus('cloud') === 'completed' && <span className="step-check">✅</span>}
                    {getStepStatus('cloud') === 'active' && <span className="step-spinner">🔄</span>}
                  </div>
                  <div className={`step ${getStepStatus('agent')}`}>
                    🤖 Loading AI agents...
                    {getStepStatus('agent') === 'completed' && <span className="step-check">✅</span>}
                    {getStepStatus('agent') === 'active' && <span className="step-spinner">🔄</span>}
                  </div>
                </div>
              </>
            )}
            
            {status === 'error' && (
              <>
                <h3>Backend Connection Failed</h3>
                <p className="error-message">{message}</p>
                {error && (
                  <details className="error-details">
                    <summary>Error Details</summary>
                    <pre>{error}</pre>
                  </details>
                )}
                {onRetry && (
                  <button className="retry-button" onClick={onRetry}>
                    🔄 Retry Connection
                  </button>
                )}
                <div className="error-help">
                  <p>The backend may take 30-60 seconds to fully initialize.</p>
                  <p>This includes database connection, cloud services, and AI agent loading.</p>
                </div>
              </>
            )}
            
            {status === 'success' && (
              <>
                <h3>Ready!</h3>
                <p>{message}</p>
              </>
            )}
          </div>
        </div>

        <div className="loading-footer">
          <p>Please wait while MIRIX initializes...</p>
          <p className="loading-tip">
            💡 <strong>Tip:</strong> First startup may take longer as we connect to database and cloud services
          </p>
        </div>
      </div>
    </div>
  );
};

export default LoadingScreen; 