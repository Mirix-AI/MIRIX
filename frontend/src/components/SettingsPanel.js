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
  const [mcpMarketplace, setMcpMarketplace] = useState({ servers: [], categories: [] });
  const [mcpSearchQuery, setMcpSearchQuery] = useState('');
  const [mcpSearchResults, setMcpSearchResults] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [isLoadingMcp, setIsLoadingMcp] = useState(false);
  const [mcpMessage, setMcpMessage] = useState('');
  const [showGmailModal, setShowGmailModal] = useState(false);
  const [gmailCredentials, setGmailCredentials] = useState({ clientId: '', clientSecret: '' });
  const [users, setUsers] = useState([]);
  const [isLoadingUsers, setIsLoadingUsers] = useState(false);
  const [currentUser, setCurrentUser] = useState(null);
  const [isUserDropdownOpen, setIsUserDropdownOpen] = useState(false);
  const [showAddUserModal, setShowAddUserModal] = useState(false);
  const [newUserName, setNewUserName] = useState('');

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
        setPersonaDetails(prevDetails => {
          const hasChanged = JSON.stringify(prevDetails) !== JSON.stringify(data.personas);
          return hasChanged ? data.personas : prevDetails;
        });
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
        setSelectedPersonaText(prevText => {
          return prevText !== data.text ? data.text : prevText;
        });
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
        setCustomModels(prevModels => {
          const newModels = data.models || [];
          const hasChanged = JSON.stringify(prevModels) !== JSON.stringify(newModels);
          return hasChanged ? newModels : prevModels;
        });
      } else {
        console.error('Failed to fetch custom models');
      }
    } catch (error) {
      console.error('Error fetching custom models:', error);
    }
  }, [settings.serverUrl]);

  const fetchMcpMarketplace = useCallback(async () => {
    if (!settings.serverUrl) {
      console.log('fetchMcpMarketplace: serverUrl not available yet');
      return;
    }
    try {
      setIsLoadingMcp(true);
      console.log('Fetching MCP marketplace data...');
      const response = await queuedFetch(`${settings.serverUrl}/mcp/marketplace`);
      if (response.ok) {
        const data = await response.json();
        console.log('MCP marketplace data:', data);
        
        const newMarketplace = { 
          servers: data.servers || [], 
          categories: ['All', ...(data.categories || [])] 
        };
        
        // Only update state if data has actually changed
        setMcpMarketplace(prevMarketplace => {
          const hasChanged = JSON.stringify(prevMarketplace) !== JSON.stringify(newMarketplace);
          return hasChanged ? newMarketplace : prevMarketplace;
        });
        
        setMcpSearchResults(prevResults => {
          const newResults = data.servers || [];
          const hasChanged = JSON.stringify(prevResults) !== JSON.stringify(newResults);
          return hasChanged ? newResults : prevResults;
        });
        
        // Log connection status
        const connectedServers = data.servers?.filter(s => s.is_connected) || [];
        console.log(`Found ${connectedServers.length} connected MCP servers:`, connectedServers.map(s => s.id));
      } else {
        console.error('Failed to fetch MCP marketplace');
        setMcpMessage('❌ Failed to load MCP marketplace');
      }
    } catch (error) {
      console.error('Error fetching MCP marketplace:', error);
      setMcpMessage('❌ Error loading MCP marketplace');
    } finally {
      setIsLoadingMcp(false);
    }
  }, [settings.serverUrl]);

  const fetchUsers = useCallback(async () => {
    if (!settings.serverUrl) {
      console.log('fetchUsers: serverUrl not available yet');
      return;
    }
    try {
      setIsLoadingUsers(true);
      console.log('Fetching users data...');
      const response = await queuedFetch(`${settings.serverUrl}/users`);
      if (response.ok) {
        const data = await response.json();
        console.log('Users data:', data);
        const usersList = data.users || [];
        setUsers(usersList);
        
        // Set current user (find user with active status)
        const activeUser = usersList.find(user => user.status === 'active');
        if (activeUser) {
          setCurrentUser(activeUser);
        } else if (!currentUser && usersList.length > 0) {
          // Fallback: if no active user found, use first user
          setCurrentUser(usersList[0]);
        }
      } else {
        console.error('Failed to fetch users');
      }
    } catch (error) {
      console.error('Error fetching users:', error);
    } finally {
      setIsLoadingUsers(false);
    }
  }, [settings.serverUrl]);

  const searchMcpMarketplace = useCallback(async (query = '') => {
    if (!settings.serverUrl) {
      return;
    }
    try {
      setIsLoadingMcp(true);
      let url = `${settings.serverUrl}/mcp/marketplace`;
      if (query.trim()) {
        url += `/search?query=${encodeURIComponent(query)}`;
      }
      
      const response = await queuedFetch(url);
      if (response.ok) {
        const data = await response.json();
        const results = query.trim() ? (data.results || []) : (data.servers || []);
        
        // Filter by category if not 'All'
        const filteredResults = selectedCategory === 'All' 
          ? results 
          : results.filter(server => server.category === selectedCategory);
          
        setMcpSearchResults(filteredResults);
      }
    } catch (error) {
      console.error('Error searching MCP marketplace:', error);
    } finally {
      setIsLoadingMcp(false);
    }
  }, [settings.serverUrl, selectedCategory]);

  const refreshMcpStatus = useCallback(async () => {
    if (!settings.serverUrl) {
      return;
    }
    try {
      console.log('Refreshing MCP connection status...');
      const response = await queuedFetch(`${settings.serverUrl}/mcp/status`);
      if (response.ok) {
        const data = await response.json();
        console.log('MCP status:', data);
        
        // Check if we have any connected servers
        if (data.connected_servers && data.connected_servers.length > 0) {
          console.log(`Found ${data.connected_servers.length} connected MCP servers:`, data.connected_servers);
        }
        
        // Refresh the marketplace to get updated connection status
        await fetchMcpMarketplace();
        await searchMcpMarketplace(mcpSearchQuery);
      }
    } catch (error) {
      console.error('Error refreshing MCP status:', error);
    }
  }, [settings.serverUrl, fetchMcpMarketplace, searchMcpMarketplace, mcpSearchQuery]);

  const connectMcpServer = useCallback(async (serverId, envVars = {}) => {
    if (!settings.serverUrl) {
      return;
    }
    
    // Special handling for Gmail - show modal for credentials
    if (serverId === 'gmail-native') {
      setShowGmailModal(true);
      return; // Exit here, actual connection happens in handleGmailConnect
    }
    
    try {
      setIsLoadingMcp(true);
      setMcpMessage(`Connecting to ${serverId}...`);
      
      const response = await queuedFetch(`${settings.serverUrl}/mcp/marketplace/connect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ server_id: serverId, env_vars: envVars })
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          setMcpMessage(`✅ Connected to ${data.server_name}! Found ${data.tools_count} tools.`);
          // Refresh the marketplace to update connection status
          await fetchMcpMarketplace();
          await searchMcpMarketplace(mcpSearchQuery);
        } else {
          setMcpMessage(`❌ ${data.error}`);
        }
      } else {
        setMcpMessage('❌ Failed to connect to server');
      }
    } catch (error) {
      console.error('Error connecting MCP server:', error);
      setMcpMessage('❌ Connection error');
    } finally {
      setIsLoadingMcp(false);
      setTimeout(() => setMcpMessage(''), 5000);
    }
  }, [settings.serverUrl, fetchMcpMarketplace, searchMcpMarketplace, mcpSearchQuery]);

  const handleGmailConnect = useCallback(async () => {
    const { clientId, clientSecret } = gmailCredentials;
    
    if (!clientId.trim()) {
      setMcpMessage('❌ Gmail Client ID is required');
      return;
    }
    
    if (!clientSecret.trim()) {
      setMcpMessage('❌ Gmail Client Secret is required');
      return;
    }
    
    setShowGmailModal(false);
    setMcpMessage('🔐 Connecting to Gmail... This will open a browser window for OAuth authorization.');
    
    const envVars = { client_id: clientId.trim(), client_secret: clientSecret.trim() };
    
    try {
      setIsLoadingMcp(true);
      setMcpMessage('Connecting to Gmail...');
      
      const response = await queuedFetch(`${settings.serverUrl}/mcp/marketplace/connect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ server_id: 'gmail-native', env_vars: envVars })
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          setMcpMessage(`✅ Connected to ${data.server_name}! Found ${data.tools_count} tools.`);
          // Clear credentials for security
          setGmailCredentials({ clientId: '', clientSecret: '' });
          // Refresh the marketplace to update connection status
          await fetchMcpMarketplace();
          await searchMcpMarketplace(mcpSearchQuery);
        } else {
          setMcpMessage(`❌ ${data.error}`);
        }
      } else {
        setMcpMessage('❌ Failed to connect to Gmail server');
      }
    } catch (error) {
      console.error('Error connecting Gmail server:', error);
      setMcpMessage('❌ Gmail connection error');
    } finally {
      setIsLoadingMcp(false);
      setTimeout(() => setMcpMessage(''), 5000);
    }
  }, [gmailCredentials, settings.serverUrl, fetchMcpMarketplace, searchMcpMarketplace, mcpSearchQuery]);

  const handleGmailModalClose = useCallback(() => {
    setShowGmailModal(false);
    setGmailCredentials({ clientId: '', clientSecret: '' });
  }, []);

  const handleUserSelection = useCallback(async (selectedUser) => {
    if (!settings.serverUrl) {
      return;
    }
    
    try {
      console.log('Switching to user:', selectedUser);
      
      const response = await queuedFetch(`${settings.serverUrl}/users/switch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: selectedUser.id })
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          setCurrentUser(selectedUser);
          console.log('Successfully switched to user:', data.user);
          // Refresh users list to update status
          await fetchUsers();
        } else {
          console.error('Failed to switch user:', data.message);
        }
      } else {
        console.error('Failed to switch user - server error');
      }
    } catch (error) {
      console.error('Error switching user:', error);
    } finally {
      setIsUserDropdownOpen(false);
    }
  }, [settings.serverUrl, fetchUsers]);

  const handleCreateUser = useCallback(async () => {
    if (!settings.serverUrl || !newUserName.trim()) {
      return;
    }
    
    try {
      const response = await queuedFetch(`${settings.serverUrl}/users/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: newUserName.trim() })
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('User created successfully:', data);
        
        // Refresh users list
        await fetchUsers();
        
        // Close modal and reset form
        setShowAddUserModal(false);
        setNewUserName('');
        
        // Set the new user as current user
        if (data.user) {
          setCurrentUser(data.user);
        }
      } else {
        console.error('Failed to create user');
      }
    } catch (error) {
      console.error('Error creating user:', error);
    }
  }, [settings.serverUrl, newUserName, fetchUsers]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (isUserDropdownOpen && !event.target.closest('.user-selector-dropdown')) {
        setIsUserDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isUserDropdownOpen]);

  const disconnectMcpServer = useCallback(async (serverId) => {
    if (!settings.serverUrl) {
      return;
    }
    try {
      setIsLoadingMcp(true);
      setMcpMessage(`Disconnecting from ${serverId}...`);
      
      const response = await queuedFetch(`${settings.serverUrl}/mcp/marketplace/disconnect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ server_id: serverId })
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          setMcpMessage(`✅ Disconnected from ${serverId}`);
          // Refresh the marketplace to update connection status
          await fetchMcpMarketplace();
          await searchMcpMarketplace(mcpSearchQuery);
        } else {
          setMcpMessage(`❌ ${data.error}`);
        }
      } else {
        setMcpMessage('❌ Failed to disconnect from server');
      }
    } catch (error) {
      console.error('Error disconnecting MCP server:', error);
      setMcpMessage('❌ Disconnection error');
    } finally {
      setIsLoadingMcp(false);
      setTimeout(() => setMcpMessage(''), 5000);
    }
  }, [settings.serverUrl, fetchMcpMarketplace, searchMcpMarketplace, mcpSearchQuery]);

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
      fetchMcpMarketplace();
      fetchUsers();
    }
  }, [settings.serverUrl, fetchPersonaDetails, fetchCoreMemoryPersona, fetchCurrentModel, fetchCurrentMemoryModel, fetchCurrentTimezone, fetchCustomModels, fetchMcpMarketplace, fetchUsers]);

  // Fetch current models and timezone whenever settings panel becomes visible
  useEffect(() => {
    if (isVisible && settings.serverUrl) {
      console.log('SettingsPanel: became visible, refreshing current models and timezone');
      fetchCurrentModel();
      fetchCurrentMemoryModel();
      fetchCurrentTimezone();
      fetchCustomModels();
      fetchMcpMarketplace();
      fetchUsers();
      // Also refresh MCP status to ensure connections are shown correctly
      refreshMcpStatus();
    }
  }, [isVisible, settings.serverUrl, fetchCurrentModel, fetchCurrentMemoryModel, fetchCurrentTimezone, fetchCustomModels, fetchMcpMarketplace, fetchUsers, refreshMcpStatus]);

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
      fetchMcpMarketplace();
      fetchUsers();
    }
  }, [settings.lastBackendRefresh, settings.serverUrl, fetchPersonaDetails, fetchCoreMemoryPersona, fetchCurrentModel, fetchCurrentMemoryModel, fetchCurrentTimezone, fetchCustomModels, fetchMcpMarketplace, fetchUsers]);

  // Handle MCP search and filtering
  useEffect(() => {
    if (settings.serverUrl && mcpMarketplace.servers.length > 0) {
      searchMcpMarketplace(mcpSearchQuery);
    }
  }, [mcpSearchQuery, selectedCategory, searchMcpMarketplace, settings.serverUrl, mcpMarketplace.servers.length]);

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
          <div className="section-header-with-action">
            <div>
              <h3>🔧 MCP Tools Marketplace</h3>
              <p>Discover and connect to Model Context Protocol (MCP) tools to extend your agent's capabilities.</p>
            </div>
            <button
              onClick={refreshMcpStatus}
              disabled={isLoadingMcp}
              className="refresh-mcp-btn"
              title="Refresh MCP connection status"
            >
              🔄 Refresh
            </button>
          </div>
          
          <div className="mcp-marketplace">
            {/* Search and Filter Controls */}
            <div className="mcp-controls">
              <div className="mcp-search">
                <input
                  type="text"
                  placeholder="Search MCP tools..."
                  value={mcpSearchQuery}
                  onChange={(e) => setMcpSearchQuery(e.target.value)}
                  className="mcp-search-input"
                />
              </div>
              <div className="mcp-filter">
                <select
                  value={selectedCategory}
                  onChange={(e) => setSelectedCategory(e.target.value)}
                  className="mcp-category-select"
                >
                  {mcpMarketplace.categories.map(category => (
                    <option key={category} value={category}>{category}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* Status Message */}
            {mcpMessage && (
              <div className={`mcp-message ${mcpMessage.includes('✅') ? 'success' : mcpMessage.includes('❌') ? 'error' : 'info'}`}>
                {mcpMessage}
              </div>
            )}

            {/* Loading State */}
            {isLoadingMcp && (
              <div className="mcp-loading">🔄 Loading MCP tools...</div>
            )}

            {/* MCP Tools List */}
            <div className="mcp-servers-list">
              {mcpSearchResults.map(server => (
                <div key={server.id} className="mcp-server-item">
                  <div className="mcp-server-header">
                    <div className="mcp-server-info">
                      <h4 className="mcp-server-name">{server.name}</h4>
                      <span className="mcp-server-category">{server.category}</span>
                      {server.is_connected && <span className="mcp-connected-badge">✅ Connected</span>}
                    </div>
                    <div className="mcp-server-actions">
                      {server.is_connected ? (
                        <button
                          onClick={() => disconnectMcpServer(server.id)}
                          disabled={isLoadingMcp}
                          className="mcp-disconnect-btn"
                        >
                          Disconnect
                        </button>
                      ) : (
                        <button
                          onClick={() => connectMcpServer(server.id)}
                          disabled={isLoadingMcp}
                          className="mcp-connect-btn"
                        >
                          Connect
                        </button>
                      )}
                    </div>
                  </div>
                  <p className="mcp-server-description">{server.description}</p>
                  <div className="mcp-server-details">
                    {server.author && <span className="mcp-author">By {server.author}</span>}
                    {server.tags && server.tags.length > 0 && (
                      <div className="mcp-tags">
                        {server.tags.map(tag => (
                          <span key={tag} className="mcp-tag">{tag}</span>
                        ))}
                      </div>
                    )}
                    {server.requirements && server.requirements.length > 0 && (
                      <div className="mcp-requirements">
                        <strong>Requirements:</strong> {server.requirements.join(', ')}
                      </div>
                    )}
                    {server.documentation_url && (
                      <a 
                        href={server.documentation_url} 
                        target="_blank" 
                        rel="noopener noreferrer" 
                        className="mcp-github-link"
                      >
                        📖 Documentation
                      </a>
                    )}
                  </div>
                </div>
              ))}
              
              {!isLoadingMcp && mcpSearchResults.length === 0 && (
                <div className="mcp-no-results">
                  {mcpSearchQuery ? 'No MCP tools found matching your search.' : 'No MCP tools available.'}
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="settings-section">
          <h3>👥 User Selection</h3>
          
          <div className="setting-item">
            <label htmlFor="user-selector">Current User</label>
            <div className="user-selector-container">
              <div className="user-selector-with-add">
                <div 
                  className={`user-selector-dropdown ${isUserDropdownOpen ? 'open' : ''}`}
                  onClick={() => setIsUserDropdownOpen(!isUserDropdownOpen)}
                >
                  <div className="user-selector-current">
                    {isLoadingUsers ? (
                      <span className="loading-text">🔄 Loading users...</span>
                    ) : currentUser ? (
                      <span className="current-user-display">
                        {currentUser.name}
                      </span>
                    ) : (
                      <span className="no-user-selected">No user selected</span>
                    )}
                    <span className={`dropdown-arrow ${isUserDropdownOpen ? 'up' : 'down'}`}>
                      {isUserDropdownOpen ? '▲' : '▼'}
                    </span>
                  </div>
                  
                  {isUserDropdownOpen && !isLoadingUsers && (
                    <div className="user-dropdown-list">
                      {users.length > 0 ? (
                        users.map(user => (
                          <div 
                            key={user.id} 
                            className={`user-dropdown-item ${currentUser?.id === user.id ? 'selected' : ''}`}
                            onClick={(e) => {
                              e.stopPropagation();
                              handleUserSelection(user);
                            }}
                          >
                            <div className="user-dropdown-info">
                              <span className="user-dropdown-name">{user.name}</span>
                            </div>
                            {currentUser?.id === user.id && (
                              <span className="selected-indicator">✓</span>
                            )}
                          </div>
                        ))
                      ) : (
                        <div className="no-users-dropdown">No users available</div>
                      )}
                    </div>
                  )}
                </div>
                <button
                  className="add-user-button"
                  onClick={(e) => {
                    e.stopPropagation();
                    setShowAddUserModal(true);
                  }}
                  title="Add New User"
                >
                  + Add User
                </button>
              </div>
            </div>
            <span className="setting-description">
              Select the current user context for the application.
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

      {/* Gmail Credentials Modal */}
      {showGmailModal && (
        <div className="modal-overlay">
          <div className="modal-content gmail-modal">
            <div className="modal-header">
              <h3>📧 Gmail OAuth2 Setup</h3>
              <button className="modal-close-btn" onClick={handleGmailModalClose}>×</button>
            </div>
            <div className="modal-body">
              <p className="gmail-modal-description">
                To connect Gmail, you need OAuth2 credentials from Google Cloud Console.
              </p>
              <div className="gmail-setup-steps">
                <ol>
                  <li>Go to <a href="https://console.cloud.google.com/apis/credentials" target="_blank" rel="noopener noreferrer">Google Cloud Console</a></li>
                  <li>Create OAuth2 credentials if you don't have them</li>
                  <li>Add <code>http://localhost:8080</code>, <code>http://localhost:8081</code>, and <code>http://localhost:8082</code> as redirect URIs</li>
                  <li>Enter your credentials below</li>
                </ol>
              </div>
              
              <div className="gmail-form">
                <div className="form-group">
                  <label htmlFor="gmail-client-id">Client ID:</label>
                  <input
                    id="gmail-client-id"
                    type="text"
                    value={gmailCredentials.clientId}
                    onChange={(e) => setGmailCredentials(prev => ({...prev, clientId: e.target.value}))}
                    placeholder="Enter your Gmail OAuth2 Client ID"
                    className="gmail-input"
                  />
                </div>
                
                <div className="form-group">
                  <label htmlFor="gmail-client-secret">Client Secret:</label>
                  <input
                    id="gmail-client-secret"
                    type="password"
                    value={gmailCredentials.clientSecret}
                    onChange={(e) => setGmailCredentials(prev => ({...prev, clientSecret: e.target.value}))}
                    placeholder="Enter your Gmail OAuth2 Client Secret"
                    className="gmail-input"
                  />
                </div>
              </div>
            </div>
            
            <div className="modal-footer">
              <button className="modal-btn secondary" onClick={handleGmailModalClose}>
                Cancel
              </button>
              <button 
                className="modal-btn primary" 
                onClick={handleGmailConnect}
                disabled={!gmailCredentials.clientId.trim() || !gmailCredentials.clientSecret.trim()}
              >
                Connect to Gmail
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Add User Modal */}
      {showAddUserModal && (
        <div className="modal-overlay">
          <div className="modal-content add-user-modal">
            <div className="modal-header">
              <h3>👤 Add New User</h3>
              <button className="modal-close-btn" onClick={() => setShowAddUserModal(false)}>×</button>
            </div>
            <div className="modal-body">
              <p className="add-user-description">
                Create a new user account in the system.
              </p>
              
              <div className="add-user-form">
                <div className="form-group">
                  <label htmlFor="new-user-name">User Name:</label>
                  <input
                    id="new-user-name"
                    type="text"
                    value={newUserName}
                    onChange={(e) => setNewUserName(e.target.value)}
                    placeholder="Enter user name"
                    className="add-user-input"
                    autoFocus
                  />
                </div>
              </div>
            </div>
            
            <div className="modal-footer">
              <button className="modal-btn secondary" onClick={() => setShowAddUserModal(false)}>
                Cancel
              </button>
              <button 
                className="modal-btn primary" 
                onClick={handleCreateUser}
                disabled={!newUserName.trim()}
              >
                Create User
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SettingsPanel; 