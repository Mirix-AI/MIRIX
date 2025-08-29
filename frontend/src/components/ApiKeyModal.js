import React, { useState, useEffect } from 'react';
import './ApiKeyModal.css';
import queuedFetch from '../utils/requestQueue';
import { useTranslation } from 'react-i18next';

const ApiKeyModal = ({ isOpen, onClose, missingKeys, modelType, onSubmit, serverUrl }) => {
  const { t } = useTranslation();
  const [selectedService, setSelectedService] = useState('');
  const [apiKeyValue, setApiKeyValue] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');

  // Define all available API services using translations
  const getApiServices = () => [
    { value: 'OPENAI_API_KEY', label: t('settings.modals.apiKey.services.OPENAI_API_KEY.label'), description: t('settings.modals.apiKey.services.OPENAI_API_KEY.description') },
    { value: 'ANTHROPIC_API_KEY', label: t('settings.modals.apiKey.services.ANTHROPIC_API_KEY.label'), description: t('settings.modals.apiKey.services.ANTHROPIC_API_KEY.description') },
    { value: 'GEMINI_API_KEY', label: t('settings.modals.apiKey.services.GEMINI_API_KEY.label'), description: t('settings.modals.apiKey.services.GEMINI_API_KEY.description') },
    { value: 'GROQ_API_KEY', label: t('settings.modals.apiKey.services.GROQ_API_KEY.label'), description: t('settings.modals.apiKey.services.GROQ_API_KEY.description') },
    { value: 'TOGETHER_API_KEY', label: t('settings.modals.apiKey.services.TOGETHER_API_KEY.label'), description: t('settings.modals.apiKey.services.TOGETHER_API_KEY.description') },
    { value: 'AZURE_API_KEY', label: t('settings.modals.apiKey.services.AZURE_API_KEY.label'), description: t('settings.modals.apiKey.services.AZURE_API_KEY.description') },
    { value: 'AZURE_BASE_URL', label: t('settings.modals.apiKey.services.AZURE_BASE_URL.label'), description: t('settings.modals.apiKey.services.AZURE_BASE_URL.description') },
    { value: 'AZURE_API_VERSION', label: t('settings.modals.apiKey.services.AZURE_API_VERSION.label'), description: t('settings.modals.apiKey.services.AZURE_API_VERSION.description') },
    { value: 'AWS_ACCESS_KEY_ID', label: t('settings.modals.apiKey.services.AWS_ACCESS_KEY_ID.label'), description: t('settings.modals.apiKey.services.AWS_ACCESS_KEY_ID.description') },
    { value: 'AWS_SECRET_ACCESS_KEY', label: t('settings.modals.apiKey.services.AWS_SECRET_ACCESS_KEY.label'), description: t('settings.modals.apiKey.services.AWS_SECRET_ACCESS_KEY.description') },
    { value: 'AWS_REGION', label: t('settings.modals.apiKey.services.AWS_REGION.label'), description: t('settings.modals.apiKey.services.AWS_REGION.description') },
  ];

  const apiServices = getApiServices();

  const getKeyPlaceholder = (keyName) => {
    const placeholders = {
      'OPENAI_API_KEY': 'sk-...',
      'ANTHROPIC_API_KEY': 'sk-ant-...',
      'GEMINI_API_KEY': 'AI...',
      'AZURE_BASE_URL': 'https://your-resource.openai.azure.com',
      'AZURE_API_VERSION': '2024-09-01-preview',
      'AWS_REGION': 'us-east-1',
    };
    return placeholders[keyName] || t('settings.modals.apiKey.enterKeyPlaceholder');
  };

  const isMissingKeysMode = missingKeys && missingKeys.length > 0;

  // Clear state when modal opens
  useEffect(() => {
    if (isOpen) {
      setError('');
      setIsSubmitting(false);
      setSelectedService('');
      setApiKeyValue('');
    }
  }, [isOpen]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(`⏳ ${t('settings.modals.apiKey.savingKeys')}`);

    try {
      if (isMissingKeysMode) {
        // Handle missing keys mode - submit all missing keys
        const apiKeys = {};
        missingKeys.forEach(keyName => {
          const input = document.getElementById(keyName);
          if (input && input.value) {
            apiKeys[keyName] = input.value;
          }
        });

        for (const keyName of missingKeys) {
          if (apiKeys[keyName]) {
            const response = await queuedFetch(`${serverUrl}/api_keys/update`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                key_name: keyName,
                key_value: apiKeys[keyName]
              }),
            });

            if (!response.ok) {
              const errorData = await response.json();
              throw new Error(errorData.detail || `Failed to update ${keyName}`);
            }
          }
        }
      } else {
        // Handle manual update mode - submit selected service
        if (!selectedService || !apiKeyValue) {
          setError(t('settings.modals.apiKey.pleaseSelectService'));
          setIsSubmitting(false);
          return;
        }

        const response = await queuedFetch(`${serverUrl}/api_keys/update`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            key_name: selectedService,
            key_value: apiKeyValue
          }),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || `Failed to update ${selectedService}`);
        }
      }

      // Show success message
      setError(`✅ ${t('settings.modals.apiKey.keysSuccessfullySaved')}`);
      
      // Small delay to show the message before closing
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Call the onSubmit callback to refresh the parent component
      onSubmit();
      onClose();
    } catch (err) {
      setError(err.message || t('settings.modals.apiKey.failedToUpdate'));
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleServiceChange = (e) => {
    setSelectedService(e.target.value);
    setApiKeyValue(''); // Clear the input when service changes
    setError(''); // Clear any errors
  };

  const getSelectedServiceInfo = () => {
    return apiServices.find(service => service.value === selectedService);
  };

  if (!isOpen) return null;

  return (
    <div className="api-key-modal-overlay">
      <div className="api-key-modal">
        <div className="api-key-modal-header">
          <h2>🔑 {isMissingKeysMode ? t('settings.modals.apiKey.titleRequired') : t('settings.modals.apiKey.title')}</h2>
          {isMissingKeysMode ? (
            <p>
              {t('settings.modals.apiKey.requiredDescription', { modelType })}
            </p>
          ) : (
            <p>
              {t('settings.modals.apiKey.manualDescription')}
            </p>
          )}
        </div>

        <form onSubmit={handleSubmit} className="api-key-form">
          {isMissingKeysMode ? (
            // Missing keys mode - show all missing keys
            missingKeys.map((keyName) => {
              const serviceInfo = apiServices.find(s => s.value === keyName);
              return (
                <div key={keyName} className="api-key-field">
                  <label htmlFor={keyName}>
                    <strong>{serviceInfo ? serviceInfo.label : keyName}</strong>
                    <span className="key-description">{serviceInfo ? serviceInfo.description : `Your ${keyName}`}</span>
                  </label>
                  <input
                    type={keyName.includes('SECRET') || keyName.includes('KEY') ? 'password' : 'text'}
                    id={keyName}
                    placeholder={getKeyPlaceholder(keyName)}
                    required
                    className="api-key-input"
                  />
                </div>
              );
            })
          ) : (
            // Manual update mode - show dropdown and single input
            <>
              <div className="api-key-field">
                <label htmlFor="service-select">
                  <strong>{t('settings.modals.apiKey.selectService')}</strong>
                  <span className="key-description">{t('settings.modals.apiKey.selectServiceDescription')}</span>
                </label>
                <select
                  id="service-select"
                  value={selectedService}
                  onChange={handleServiceChange}
                  required
                  className="api-key-select"
                >
                  <option value="">{t('settings.modals.apiKey.selectServicePlaceholder')}</option>
                  {apiServices.map((service) => (
                    <option key={service.value} value={service.value}>
                      {service.label}
                    </option>
                  ))}
                </select>
              </div>

              {selectedService && (
                <div className="api-key-field">
                  <label htmlFor="api-key-input">
                    <strong>{getSelectedServiceInfo()?.label}</strong>
                    <span className="key-description">{getSelectedServiceInfo()?.description}</span>
                  </label>
                  <input
                    type={selectedService.includes('SECRET') || selectedService.includes('KEY') ? 'password' : 'text'}
                    id="api-key-input"
                    value={apiKeyValue}
                    onChange={(e) => setApiKeyValue(e.target.value)}
                    placeholder={getKeyPlaceholder(selectedService)}
                    required
                    className="api-key-input"
                  />
                </div>
              )}
            </>
          )}

          {error && (
            <div className={`api-key-error ${error.includes(t('settings.modals.apiKey.keysSuccessfullySaved')) || error.includes(t('settings.modals.apiKey.savingKeys')) ? 'api-key-info' : ''}`}>
              {error.includes(t('settings.modals.apiKey.keysSuccessfullySaved')) || error.includes(t('settings.modals.apiKey.savingKeys')) ? error : `❌ ${error}`}
            </div>
          )}

          <div className="api-key-modal-actions">
            <button
              type="button"
              onClick={onClose}
              className="api-key-cancel-btn"
              disabled={isSubmitting}
            >
              {t('settings.modals.apiKey.cancel')}
            </button>
            <button
              type="submit"
              className="api-key-submit-btn"
              disabled={isSubmitting || (isMissingKeysMode ? false : (!selectedService || !apiKeyValue))}
            >
              {isSubmitting ? (error.includes(t('settings.modals.apiKey.keysSuccessfullySaved')) ? `✅ ${t('settings.modals.apiKey.saved')}` : `⏳ ${t('settings.modals.apiKey.saving')}`) : `✅ ${t('settings.modals.apiKey.save')}`}
            </button>
          </div>
        </form>

        <div className="api-key-note">
          <p>
            <strong>{t('settings.modals.apiKey.noteLabel')}</strong> {t('settings.modals.apiKey.note')}
          </p>
        </div>
      </div>
    </div>
  );
};

export default ApiKeyModal; 