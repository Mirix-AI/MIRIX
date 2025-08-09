import React from 'react';
import './ClearChatModal.css';
import { useTranslation } from 'react-i18next';

const ClearChatModal = ({ isOpen, onClose, onClearLocal, onClearPermanent, isClearing }) => {
  const { t } = useTranslation();
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>{t('clearChat.title')}</h3>
          <button className="modal-close" onClick={onClose} disabled={isClearing}>
            ×
          </button>
        </div>
        
        <div className="modal-body">
          <p>{t('clearChat.choose')}</p>
          
          <div className="clear-options">
            <div className="clear-option">
              <div className="clear-option-header">
                <h4>{t('clearChat.local.title')}</h4>
                <span className="option-type">{t('clearChat.local.type')}</span>
              </div>
              <p>
                {t('clearChat.local.desc')}
              </p>
              <button 
                className="clear-local-btn"
                onClick={onClearLocal}
                disabled={isClearing}
              >
                {t('clearChat.local.button')}
              </button>
            </div>
            
            <div className="clear-option permanent">
              <div className="clear-option-header">
                <h4>{t('clearChat.permanent.title')}</h4>
                <span className="option-type permanent">{t('clearChat.permanent.type')}</span>
              </div>
              <p>
                {t('clearChat.permanent.desc')}
              </p>
              <div className="warning-note">
                <span className="warning-icon">⚠️</span>
                {t('clearChat.permanent.note')}
              </div>
              <button 
                className="clear-permanent-btn"
                onClick={onClearPermanent}
                disabled={isClearing}
              >
                {isClearing ? t('clearChat.permanent.clearing') : t('clearChat.permanent.button')}
              </button>
            </div>
          </div>
        </div>
        
        <div className="modal-footer">
          <button className="cancel-btn" onClick={onClose} disabled={isClearing}>
            {t('clearChat.cancel')}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ClearChatModal; 