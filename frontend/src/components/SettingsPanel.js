import React, { useState, useEffect, useCallback } from 'react';
import './SettingsPanel.css';
import queuedFetch from '../utils/requestQueue';
import LocalModelModal from './LocalModelModal';
import { useTranslation } from 'react-i18next';

const SettingsPanel = ({ settings, onSettingsChange, onApiKeyCheck, onApiKeyRequired, isVisible }) => {
  const { t, i18n } = useTranslation();
  const [personaDetails, setPersonaDetails] = useState({});
  const [selectedPersonaText, setSelectedPersonaText] = useState('');
  const [isUpdatingPersona, setIsUpdatingPersona] = useState(false);
  const [isApplyingTemplate, setIsApplyingTemplate] = useState(false);
  const [updateMessage, setUpdateMessage] = useState('');
  const [isChangingModel, setIsChangingModel] = useState(false);
  const [modelUpdateMessage, setModelUpdateMessage] = useState('');
  const [isChangingMemoryModel, setIsChangingMemoryModel] = useState(false);
  const [memoryModelUpdateMessage, setMemoryModelUpdateMessage] = useState('');
  const [isChangingTimezone, setIsChangingTimezone] = useState(false);
  const [timezoneUpdateMessage, setTimezoneUpdateMessage] = useState('');
  const [isCheckingApiKeys, setIsCheckingApiKeys] = useState(false);
  const [apiKeyMessage, setApiKeyMessage] = useState('');
  const [isEditingPersona, setIsEditingPersona] = useState(false);
  const [selectedTemplateInEdit, setSelectedTemplateInEdit] = useState('');
  const [showLocalModelModal, setShowLocalModelModal] = useState(false);
  const [customModels, setCustomModels] = useState([]);

  // Debug logging for settings
  useEffect(() => {
    console.log('SettingsPanel: settings changed:', {
      serverUrl: settings.serverUrl,
      model: settings.model,
      persona: settings.persona,
      timezone: settings.timezone
    });
  }, [settings]);

  const handleInputChange = (key, value) => {
    onSettingsChange({ [key]: value });
  };

  const fetchPersonaDetails = useCallback(async () => {
    if (!settings.serverUrl) {
      console.log('fetchPersonaDetails: serverUrl not available yet');
      return;
    }
    try {
      const response = await queuedFetch(`${settings.serverUrl}/personas`);
      if (response.ok) {
        const data = await response.json();
        setPersonaDetails(data.personas);
      } else {
        console.error('Failed to fetch persona details');
      }
    } catch (error) {
      console.error('Error fetching persona details:', error);
    }
  }, [settings.serverUrl]);

  const fetchCoreMemoryPersona = useCallback(async () => {
    if (!settings.serverUrl) {
      console.log('fetchCoreMemoryPersona: serverUrl not available yet');
      return;
    }
    try {
      const response = await queuedFetch(`${settings.serverUrl}/personas/core_memory`);
      if (response.ok) {
        const data = await response.json();
        setSelectedPersonaText(data.text);
      } else {
        console.error('Failed to fetch core memory persona');
      }
    } catch (error) {
      console.error('Error fetching core memory persona:', error);
    }
  }, [settings.serverUrl]);

  const fetchCurrentModel = useCallback(async () => {
    if (!settings.serverUrl) {
      console.log('fetchCurrentModel: serverUrl not available yet');
      return;
    }
    try {
      const response = await queuedFetch(`${settings.serverUrl}/models/current`);
      if (response.ok) {
        const data = await response.json();
        // Only update if the model is different from current settings
        if (data.current_model !== settings.model) {
          onSettingsChange({ model: data.current_model });
        }
      } else {
        console.error('Failed to fetch current model');
      }
    } catch (error) {
      console.error('Error fetching current model:', error);
    }
  }, [settings.serverUrl, settings.model, onSettingsChange]);

  const fetchCurrentMemoryModel = useCallback(async () => {
    if (!settings.serverUrl) {
      console.log('fetchCurrentMemoryModel: serverUrl not available yet');
      return;
    }
    try {
      const response = await queuedFetch(`${settings.serverUrl}/models/memory/current`);
      if (response.ok) {
        const data = await response.json();
        // Only update if the memory model is different from current settings
        if (data.current_model !== settings.memoryModel) {
          onSettingsChange({ memoryModel: data.current_model });
        }
      } else {
        console.error('Failed to fetch current memory model');
      }
    } catch (error) {
      console.error('Error fetching current memory model:', error);
    }
  }, [settings.serverUrl, settings.memoryModel, onSettingsChange]);

  const fetchCurrentTimezone = useCallback(async () => {
    if (!settings.serverUrl) {
      console.log('fetchCurrentTimezone: serverUrl not available yet');
      return;
    }
    try {
      const response = await queuedFetch(`${settings.serverUrl}/timezone/current`);
      if (response.ok) {
        const data = await response.json();
        // Only update if the timezone is different from current settings
        if (data.timezone !== settings.timezone) {
          onSettingsChange({ timezone: data.timezone });
        }
      } else {
        console.error('Failed to fetch current timezone');
      }
    } catch (error) {
      console.error('Error fetching current timezone:', error);
    }
  }, [settings.serverUrl, settings.timezone, onSettingsChange]);

  const fetchCustomModels = useCallback(async () => {
    if (!settings.serverUrl) {
      console.log('fetchCustomModels: serverUrl not available yet');
      return;
    }
    try {
      const response = await queuedFetch(`${settings.serverUrl}/models/custom/list`);
      if (response.ok) {
        const data = await response.json();
        setCustomModels(data.models || []);
      } else {
        console.error('Failed to fetch custom models');
      }
    } catch (error) {
      console.error('Error fetching custom models:', error);
    }
  }, [settings.serverUrl]);

  const applyPersonaTemplate = useCallback(async (personaName) => {
    if (!settings.serverUrl) {
      console.log('applyPersonaTemplate: serverUrl not available yet');
      return;
    }
    try {
      const response = await queuedFetch(`${settings.serverUrl}/personas/apply_template`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          persona_name: personaName,
        }),
      });

      if (response.ok) {
        // Refresh the core memory persona text
        fetchCoreMemoryPersona();
        console.log(`Applied persona template: ${personaName}`);
      } else {
        console.error('Failed to apply persona template');
      }
    } catch (error) {
      console.error('Error applying persona template:', error);
    }
  }, [settings.serverUrl, fetchCoreMemoryPersona]);

  // Fetch initial data only when serverUrl is available
  useEffect(() => {
    if (settings.serverUrl) {
      console.log('SettingsPanel: serverUrl is available, fetching initial data');
      fetchPersonaDetails();
      fetchCoreMemoryPersona();
      fetchCurrentModel();
      fetchCurrentMemoryModel();
      fetchCurrentTimezone();
      fetchCustomModels();
    }
  }, [settings.serverUrl, fetchPersonaDetails, fetchCoreMemoryPersona, fetchCurrentModel, fetchCurrentMemoryModel, fetchCurrentTimezone, fetchCustomModels]);

  // Fetch current models and timezone whenever settings panel becomes visible
  useEffect(() => {
    if (isVisible && settings.serverUrl) {
      console.log('SettingsPanel: became visible, refreshing current models and timezone');
      fetchCurrentModel();
      fetchCurrentMemoryModel();
      fetchCurrentTimezone();
      fetchCustomModels();
    }
  }, [isVisible, settings.serverUrl, fetchCurrentModel, fetchCurrentMemoryModel, fetchCurrentTimezone, fetchCustomModels]);

  // Refresh all backend data when backend reconnects
  useEffect(() => {
    if (settings.lastBackendRefresh && settings.serverUrl) {
      console.log('SettingsPanel: backend reconnected, refreshing all data');
      fetchPersonaDetails();
      fetchCoreMemoryPersona();
      fetchCurrentModel();
      fetchCurrentMemoryModel();
      fetchCurrentTimezone();
      fetchCustomModels();
    }
  }, [settings.lastBackendRefresh, settings.serverUrl, fetchPersonaDetails, fetchCoreMemoryPersona, fetchCurrentModel, fetchCurrentMemoryModel, fetchCurrentTimezone, fetchCustomModels]);

  const handlePersonaChange = async (newPersona) => {
    console.log('handlePersonaChange called with:', newPersona);
    console.log('personaDetails:', personaDetails);
    console.log('settings.serverUrl:', settings.serverUrl);
    
    if (!settings.serverUrl) {
      console.error('Cannot change persona: serverUrl not available');
      setUpdateMessage('❌ Server not available');
      return;
    }
    
    // Only update the settings, don't apply template to backend yet
    handleInputChange('persona', newPersona);
    
    // Apply the template regardless of whether personaDetails is loaded
    // The backend will handle checking if the persona exists
    setIsApplyingTemplate(true);
    setUpdateMessage('Applying persona template...');
    
    try {
      console.log(`Applying persona template: ${newPersona}`);
      
      const response = await queuedFetch(`${settings.serverUrl}/personas/apply_template`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          persona_name: newPersona,
        }),
      });

      console.log('Response status:', response.status);
      
      if (response.ok) {
        // Immediately refresh the core memory persona text
        await fetchCoreMemoryPersona();
        setUpdateMessage('✅ Persona template applied successfully!');
        console.log(`Successfully applied and updated persona: ${newPersona}`);
      } else {
        const errorData = await response.text();
        console.error('Failed to apply persona template:', errorData);
        setUpdateMessage('❌ Failed to apply persona template');
      }
    } catch (error) {
      console.error('Error applying persona template:', error);
      setUpdateMessage('❌ Error applying persona template');
    } finally {
      setIsApplyingTemplate(false);
      // Clear message after 2 seconds
      setTimeout(() => setUpdateMessage(''), 2000);
    }
  };

    const handlePersonaTemplateChange = async (newPersona) => {
    console.log('handlePersonaTemplateChange called with:', newPersona);
    
    // Only update the edit-mode template selection (don't update main settings)
    setSelectedTemplateInEdit(newPersona);
    
    // Load the template text without updating backend
    if (personaDetails[newPersona]) {
      // Use cached persona details
      setSelectedPersonaText(personaDetails[newPersona]);
      setUpdateMessage('📝 Template loaded - click Save to apply');
      setTimeout(() => setUpdateMessage(''), 3000);
    } else {
      // Fallback: refresh persona details if not found
      setIsApplyingTemplate(true);
      setUpdateMessage('Loading template...');
      
      try {
        if (!settings.serverUrl) {
          setUpdateMessage('❌ Server not available');
          return;
        }
        
        const response = await queuedFetch(`${settings.serverUrl}/personas`);
        if (response.ok) {
          const data = await response.json();
          setPersonaDetails(data.personas);
          
          if (data.personas[newPersona]) {
            setSelectedPersonaText(data.personas[newPersona]);
            setUpdateMessage('📝 Template loaded - click Save to apply');
          } else {
            setUpdateMessage('❌ Template not found');
          }
        } else {
          setUpdateMessage('❌ Failed to load templates');
        }
      } catch (error) {
        console.error('Error loading persona template:', error);
        setUpdateMessage('❌ Error loading template');
      } finally {
        setIsApplyingTemplate(false);
        setTimeout(() => setUpdateMessage(''), 3000);
      }
    }
  };

  const handleModelChange = async (newModel) => {
    console.log('handleModelChange called with:', newModel);
    
    if (!settings.serverUrl) {
      console.error('Cannot change model: serverUrl not available');
      setModelUpdateMessage('❌ Server not available');
      return;
    }
    
    handleInputChange('model', newModel);
    
    setIsChangingModel(true);
    setModelUpdateMessage('Changing chat agent model...');
    
    try {
      console.log(`Setting chat agent model: ${newModel}`);
      
      const response = await queuedFetch(`${settings.serverUrl}/models/set`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: newModel,
        }),
      });

      console.log('Response status:', response.status);
      
      if (response.ok) {
        const data = await response.json();
        
        if (data.success) {
          setModelUpdateMessage('✅ Chat agent model set successfully!');
          console.log(`Successfully set chat agent model: ${newModel}`);
          
          // Show initialization message when model is successfully set
          setModelUpdateMessage('🔄 Initializing chat agent with new model...');
          
          // Automatically check for API keys after model change
          if (onApiKeyCheck) {
            console.log('Checking API keys for new model...');
            setTimeout(() => {
              onApiKeyCheck();
            }, 500); // Small delay to allow backend to update
          }
        } else {
          // Handle case where backend returned success: false
          if (data.missing_keys && data.missing_keys.length > 0) {
            // If there are missing API keys, immediately open the API key modal
            console.log(`Missing API keys for chat model '${newModel}': ${data.missing_keys.join(', ')}`);
            setModelUpdateMessage('🔑 Opening API key configuration...');
            
            if (onApiKeyRequired) {
              // Create retry function for this model change
              const retryFunction = () => handleModelChange(newModel);
              
              // Small delay to show the message before opening modal
              setTimeout(() => {
                onApiKeyRequired(data.missing_keys, newModel, newModel, 'chat', retryFunction);
              }, 500);
            }
          } else {
            // Show error message for other types of failures
            let errorMessage = data.message || 'Failed to set chat agent model';
            setModelUpdateMessage(`❌ ${errorMessage}`);
            console.error('Chat model set failed:', data);
          }
        }
      } else {
        const errorData = await response.text();
        console.error('Failed to set chat agent model:', errorData);
        setModelUpdateMessage('❌ Failed to set chat agent model');
      }
    } catch (error) {
      console.error('Error setting chat agent model:', error);
      setModelUpdateMessage('❌ Error setting chat agent model');
    } finally {
      setIsChangingModel(false);
      // Clear message after 2 seconds
      setTimeout(() => setModelUpdateMessage(''), 2000);
    }
  };

  const handleMemoryModelChange = async (newModel) => {
    console.log('handleMemoryModelChange called with:', newModel);
    
    if (!settings.serverUrl) {
      console.error('Cannot change memory model: serverUrl not available');
      setMemoryModelUpdateMessage('❌ Server not available');
      return;
    }
    
    handleInputChange('memoryModel', newModel);
    
    setIsChangingMemoryModel(true);
    setMemoryModelUpdateMessage('Changing memory manager model...');
    
    try {
      console.log(`Setting memory manager model: ${newModel}`);
      
      const response = await queuedFetch(`${settings.serverUrl}/models/memory/set`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: newModel,
        }),
      });

      console.log('Response status:', response.status);
      
      if (response.ok) {
        const data = await response.json();
        
        if (data.success) {
          setMemoryModelUpdateMessage('✅ Memory manager model set successfully!');
          console.log(`Successfully set memory manager model: ${newModel}`);
          
          // Show initialization message when model is successfully set
          setMemoryModelUpdateMessage('🔄 Initializing memory manager with new model...');
          
          // Automatically check for API keys after memory model change
          if (onApiKeyCheck) {
            console.log('Checking API keys for new memory model...');
            setTimeout(() => {
              onApiKeyCheck();
            }, 500); // Small delay to allow backend to update
          }
        } else {
          // Handle case where backend returned success: false
          if (data.missing_keys && data.missing_keys.length > 0) {
            // If there are missing API keys, immediately open the API key modal
            console.log(`Missing API keys for memory model '${newModel}': ${data.missing_keys.join(', ')}`);
            setMemoryModelUpdateMessage('🔑 Opening API key configuration...');
            
            if (onApiKeyRequired) {
              // Create retry function for this model change
              const retryFunction = () => handleMemoryModelChange(newModel);
              
              // Small delay to show the message before opening modal
              setTimeout(() => {
                onApiKeyRequired(data.missing_keys, newModel, newModel, 'memory', retryFunction);
              }, 500);
            }
          } else {
            // Show error message for other types of failures
            let errorMessage = data.message || 'Failed to set memory manager model';
            setMemoryModelUpdateMessage(`❌ ${errorMessage}`);
            console.error('Memory model set failed:', data);
          }
        }
      } else {
        const errorData = await response.text();
        console.error('Failed to set memory manager model:', errorData);
        setMemoryModelUpdateMessage('❌ Failed to set memory manager model');
      }
    } catch (error) {
      console.error('Error setting memory manager model:', error);
      setMemoryModelUpdateMessage('❌ Error setting memory manager model');
    } finally {
      setIsChangingMemoryModel(false);
      // Clear message after 2 seconds
      setTimeout(() => setMemoryModelUpdateMessage(''), 2000);
    }
  };

  const handleTimezoneChange = async (newTimezone) => {
    console.log('handleTimezoneChange called with:', newTimezone);
    
    if (!settings.serverUrl) {
      console.error('Cannot change timezone: serverUrl not available');
      setTimezoneUpdateMessage('❌ Server not available');
      return;
    }
    
    handleInputChange('timezone', newTimezone);
    
    setIsChangingTimezone(true);
    setTimezoneUpdateMessage('Changing timezone...');
    
    try {
      console.log(`Setting timezone: ${newTimezone}`);
      
      const response = await queuedFetch(`${settings.serverUrl}/timezone/set`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          timezone: newTimezone,
        }),
      });

      console.log('Response status:', response.status);
      
      if (response.ok) {
        const data = await response.json();
        setTimezoneUpdateMessage('✅ Timezone set successfully!');
        console.log(`Successfully set timezone: ${newTimezone}`);
      } else {
        const errorData = await response.text();
        console.error('Failed to set timezone:', errorData);
        setTimezoneUpdateMessage('❌ Failed to set timezone');
      }
    } catch (error) {
      console.error('Error setting timezone:', error);
      setTimezoneUpdateMessage('❌ Error setting timezone');
    } finally {
      setIsChangingTimezone(false);
      // Clear message after 2 seconds
      setTimeout(() => setTimezoneUpdateMessage(''), 2000);
    }
  };

  const handlePersonaTextChange = (event) => {
    setSelectedPersonaText(event.target.value);
  };

  const updatePersonaText = async () => {
    if (!settings.serverUrl) {
      console.error('Cannot update persona text: serverUrl not available');
      setUpdateMessage('❌ Server not available');
      return;
    }
    
    setIsUpdatingPersona(true);
    setUpdateMessage('Updating core memory persona...');
    
    try {
      const response = await queuedFetch(`${settings.serverUrl}/personas/update`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: selectedPersonaText,
        }),
      });

      if (response.ok) {
        setUpdateMessage('✅ Core memory persona updated successfully!');
        // Update the main settings with the selected template
        if (selectedTemplateInEdit) {
          handleInputChange('persona', selectedTemplateInEdit);
        }
        // Stay in edit mode - don't close automatically
      } else {
        const errorData = await response.text();
        console.error('Failed to update core memory persona:', errorData);
        setUpdateMessage('❌ Failed to update core memory persona');
      }
    } catch (error) {
      console.error('Error updating core memory persona:', error);
      setUpdateMessage('❌ Error updating core memory persona');
    } finally {
      setIsUpdatingPersona(false);
      // Clear message after 2 seconds
      setTimeout(() => setUpdateMessage(''), 2000);
    }
  };

  const baseModels = [
    'gpt-4o-mini',
    'gpt-4o',
    'gpt-4.1-mini',
    'gpt-4.1',
    'claude-3-5-sonnet-20241022',
    'gemini-2.0-flash',
    'gemini-2.5-flash',
    'gemini-2.5-flash-lite',
    'gemini-1.5-pro',
    'gemini-2.0-flash-lite'
  ];

  // Combine base models with custom models
  const models = [...baseModels, ...customModels];

  // Memory models support both Gemini and OpenAI models, plus custom models
  const baseMemoryModels = [
    'gemini-2.0-flash',
    'gemini-2.5-flash',
    'gemini-2.5-flash-lite',
    'gpt-4o-mini',
    'gpt-4o',
    'gpt-4.1-mini',
    'gpt-4.1',
  ];

  // Combine base memory models with custom models
  const memoryModels = [...baseMemoryModels, ...customModels];

  // Convert personaDetails object to array format for dropdown
  const personas = Object.keys(personaDetails).map(key => ({
    value: key,
    label: key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')
  }));

  const timezones = [
    'America/New_York (UTC-5:00)',
    'America/Los_Angeles (UTC-8:00)',
    'America/Chicago (UTC-6:00)',
    'America/Denver (UTC-7:00)',
    'America/Toronto (UTC-5:00)',
    'America/Vancouver (UTC-8:00)',
    'Europe/London (UTC+0:00)',
    'Europe/Paris (UTC+1:00)',
    'Europe/Berlin (UTC+1:00)',
    'Europe/Rome (UTC+1:00)',
    'Europe/Madrid (UTC+1:00)',
    'Europe/Amsterdam (UTC+1:00)',
    'Europe/Stockholm (UTC+1:00)',
    'Europe/Moscow (UTC+3:00)',
    'Asia/Tokyo (UTC+9:00)',
    'Asia/Shanghai (UTC+8:00)',
    'Asia/Seoul (UTC+9:00)',
    'Asia/Hong_Kong (UTC+8:00)',
    'Asia/Singapore (UTC+8:00)',
    'Asia/Bangkok (UTC+7:00)',
    'Asia/Jakarta (UTC+7:00)',
    'Asia/Manila (UTC+8:00)',
    'Asia/Kolkata (UTC+5:30)',
    'Asia/Dubai (UTC+4:00)',
    'Asia/Tehran (UTC+3:30)',
    'Australia/Sydney (UTC+10:00)',
    'Australia/Melbourne (UTC+10:00)',
    'Australia/Perth (UTC+8:00)',
    'Pacific/Auckland (UTC+12:00)',
    'Pacific/Honolulu (UTC-10:00)',
    'Africa/Cairo (UTC+2:00)',
    'Africa/Lagos (UTC+1:00)',
    'Africa/Johannesburg (UTC+2:00)',
    'America/Sao_Paulo (UTC-3:00)',
    'America/Buenos_Aires (UTC-3:00)',
    'America/Mexico_City (UTC-6:00)'
  ];

  return (
    <div className="settings-panel">
      <div className="settings-header">
        <h2>{t('settings.title')}</h2>
        <p>{t('settings.subtitle')}</p>
      </div>

      <div className="settings-content">
        <div className="settings-section">
          <h3>{t('settings.sections.model')}</h3>
          
          <div className="setting-item">
            <label htmlFor="model-select">{t('settings.chatModel')}</label>
            <div className="model-select-container">
              <select
                id="model-select"
                value={settings.model}
                onChange={(e) => handleModelChange(e.target.value)}
                className="setting-select"
                disabled={isChangingModel}
              >
                {models.map(model => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
              <button
                className="add-model-button"
                onClick={() => setShowLocalModelModal(true)}
                title={t('settings.descriptions.addModelTooltip')}
                disabled={isChangingModel}
              >
                {t('settings.add')}
              </button>
            </div>
            <span className="setting-description">
              {isChangingModel ? t('settings.descriptions.changingChatModel') : t('settings.descriptions.chatModel')}
            </span>
            {modelUpdateMessage && (
              <span className={`update-message ${modelUpdateMessage.includes('✅') ? 'success' : modelUpdateMessage.includes('Changing') ? 'info' : 'error'}`}>
                {modelUpdateMessage}
              </span>
            )}
          </div>

          <div className="setting-item">
            <label htmlFor="memory-model-select">{t('settings.memoryModel')}</label>
            <div className="model-select-container">
              <select
                id="memory-model-select"
                value={settings.memoryModel || settings.model}
                onChange={(e) => handleMemoryModelChange(e.target.value)}
                className="setting-select"
                disabled={isChangingMemoryModel}
              >
                {memoryModels.map(model => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
              <button
                className="add-model-button"
                onClick={() => setShowLocalModelModal(true)}
                title={t('settings.descriptions.addModelTooltip')}
                disabled={isChangingMemoryModel}
              >
                {t('settings.add')}
              </button>
            </div>
            <span className="setting-description">
              {isChangingMemoryModel ? t('settings.descriptions.changingMemoryModel') : t('settings.descriptions.memoryModel')}
            </span>
            {memoryModelUpdateMessage && (
              <span className={`update-message ${memoryModelUpdateMessage.includes('✅') ? 'success' : memoryModelUpdateMessage.includes('Changing') ? 'info' : 'error'}`}>
                {memoryModelUpdateMessage}
              </span>
            )}
          </div>

          {/* Persona Display/Editor */}
          <div className="setting-item persona-container">
            <label>{t('settings.persona')}</label>
            
            {!isEditingPersona ? (
              /* Display Mode */
              <div className="persona-display-mode">
                <div className="persona-display-text">
                  {selectedPersonaText || t('settings.descriptions.loadingPersona')}
                </div>
                <button
                  onClick={() => {
                    setIsEditingPersona(true);
                    setSelectedTemplateInEdit(settings.persona); // Initialize with current persona
                  }}
                  className="edit-persona-btn"
                  disabled={isApplyingTemplate}
                >
                  ✏️ {t('settings.personaEdit')}
                </button>
              </div>
            ) : (
              /* Edit Mode */
              <div className="persona-edit-mode">
                <div className="persona-template-selector">
                  <label htmlFor="persona-select">{t('settings.applyTemplate')}</label>
                  <select
                    id="persona-select"
                    value={selectedTemplateInEdit}
                    onChange={(e) => handlePersonaTemplateChange(e.target.value)}
                    className="setting-select"
                    disabled={isApplyingTemplate}
                  >
                    {personas.map(persona => (
                      <option key={persona.value} value={persona.value}>
                        {persona.label}
                      </option>
                    ))}
                  </select>
                  <span className="setting-description">
                    {isApplyingTemplate ? t('settings.descriptions.loadingTemplate') : t('settings.descriptions.templateSelector')}
                  </span>
                </div>
                
                <div className="persona-text-editor">
                  <label htmlFor="persona-text">{t('settings.editPersonaText')}</label>
                  <textarea
                    id="persona-text"
                    value={selectedPersonaText}
                    onChange={handlePersonaTextChange}
                    className="persona-textarea"
                    placeholder={t('settings.descriptions.personaPlaceholder')}
                    rows={6}
                    disabled={isApplyingTemplate}
                  />
                </div>
                
                <div className="persona-edit-actions">
                  <button
                    onClick={updatePersonaText}
                    disabled={isUpdatingPersona || isApplyingTemplate}
                    className="save-persona-btn"
                  >
                    {isUpdatingPersona ? t('settings.states.saving') : `💾 ${t('settings.buttons.save')}`}
                  </button>
                  <button
                    onClick={() => {
                      setIsEditingPersona(false);
                      setSelectedTemplateInEdit('');
                      setUpdateMessage('');
                    }}
                    className="cancel-persona-btn"
                    disabled={isUpdatingPersona || isApplyingTemplate}
                  >
                    {t('settings.buttons.cancel')}
                  </button>
                  {updateMessage && (
                    <span className={`update-message ${updateMessage.includes('✅') ? 'success' : updateMessage.includes('Applying') || updateMessage.includes('Updating') ? 'info' : 'error'}`}>
                      {updateMessage}
                    </span>
                  )}
                </div>
              </div>
            )}
            
            <span className="setting-description">
              {isEditingPersona 
                ? t('settings.descriptions.personaEdit')
                : t('settings.descriptions.personaDisplay')
              }
            </span>
          </div>
        </div>

        <div className="settings-section">
          <h3>{t('settings.sections.preferences')}</h3>
          
          {/* Language */}
          <div className="setting-item">
            <label htmlFor="language-select">{t('settings.language')}</label>
            <select
              id="language-select"
              value={i18n.language?.startsWith('zh') ? 'zh' : 'en'}
              onChange={(e) => i18n.changeLanguage(e.target.value)}
              className="setting-select"
            >
              <option value="en">English</option>
              <option value="zh">中文</option>
            </select>
            <span className="setting-description">
              {t('settings.languageDescription')}
            </span>
          </div>
          
          <div className="setting-item">
            <label htmlFor="timezone-select">{t('settings.timezone')}</label>
            <select
              id="timezone-select"
              value={settings.timezone}
              onChange={(e) => handleTimezoneChange(e.target.value)}
              className="setting-select"
              disabled={isChangingTimezone}
            >
              {timezones.map(tz => (
                <option key={tz} value={tz}>
                  {tz.replace(/_/g, ' ')}
                </option>
              ))}
            </select>
            <span className="setting-description">
              {isChangingTimezone ? t('settings.descriptions.changingTimezone') : t('settings.descriptions.timezone')}
            </span>
            {timezoneUpdateMessage && (
              <span className={`update-message ${timezoneUpdateMessage.includes('✅') ? 'success' : timezoneUpdateMessage.includes('Changing') ? 'info' : 'error'}`}>
                {timezoneUpdateMessage}
              </span>
            )}
          </div>
        </div>

        <div className="settings-section">
          <h3>{t('settings.sections.apiKeys')}</h3>
          
          <div className="setting-item">
            <label>{t('settings.apiKeyManagement')}</label>
            <div className="api-key-actions">
              <button
                onClick={() => {
                  // Force API key modal to open for manual updates
                  if (onApiKeyCheck) {
                    onApiKeyCheck(true); // Pass true to force modal open regardless of missing keys
                  }
                }}
                className="api-key-update-btn"
              >
                {`🔧 ${t('settings.updateApiKeys')}`}
              </button>
              {apiKeyMessage && (
                <span className={`update-message ${apiKeyMessage.includes('✅') ? 'success' : apiKeyMessage.includes('Checking') ? 'info' : 'error'}`}>
                  {apiKeyMessage}
                </span>
              )}
            </div>
            <span className="setting-description">
              {t('settings.descriptions.apiKeyManagement')}
            </span>
          </div>
        </div>

        <div className="settings-section">
          <h3>{t('settings.sections.about')}</h3>
          <div className="about-info">
            <p><strong>{t('settings.about.name')}</strong></p>
            <p>{t('settings.about.version')} 0.1.2</p>
            <p>{t('settings.about.description')}</p>
            <div className="about-links">
              <button 
                className="link-button"
                onClick={() => window.open('https://docs.mirix.io', '_blank')}
              >
                📖 {t('settings.about.docs')}
              </button>
              <button 
                className="link-button"
                onClick={() => window.open('https://github.com/Mirix-AI/MIRIX/issues', '_blank')}
              >
                🐛 {t('settings.about.reportIssue')}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Local Model Modal */}
      <LocalModelModal
        isOpen={showLocalModelModal}
        onClose={() => setShowLocalModelModal(false)}
        serverUrl={settings.serverUrl}
        onSuccess={(modelName) => {
          // Refresh custom models list and optionally switch to the new model
          fetchCustomModels();
          console.log(`Custom model '${modelName}' added successfully`);
        }}
      />
    </div>
  );
};

export default SettingsPanel; 