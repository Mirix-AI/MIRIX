import React, { useState, useRef, useCallback } from 'react';
// import VoiceRecorder from './VoiceRecorder';
import './ScreenshotMonitor.css';
import queuedFetch from '../utils/requestQueue';
import AppSelector from './AppSelector';
import { useTranslation } from 'react-i18next';

const ScreenshotMonitor = ({ settings, onMonitoringStatusChange }) => {
  const { t } = useTranslation();
  
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [screenshotCount, setScreenshotCount] = useState(0);
  const [lastProcessedTime, setLastProcessedTime] = useState(null);
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState(null);
  const [skipSimilarityCheck, setSkipSimilarityCheck] = useState(false);
  const [isRequestInProgress, setIsRequestInProgress] = useState(false);
  const [isProcessingScreenshot, setIsProcessingScreenshot] = useState(false);
  const [hasScreenPermission, setHasScreenPermission] = useState(null);
  const [isCheckingPermission, setIsCheckingPermission] = useState(false);
  
  // App selection state
  const [showAppSelector, setShowAppSelector] = useState(false);
  const [selectedSources, setSelectedSources] = useState([]);
  const [monitorMode, setMonitorMode] = useState('fullscreen'); // 'fullscreen' or 'selected'
  const [currentAppName, setCurrentAppName] = useState('');
  
  // Voice recording state - COMMENTED OUT
  // const [voiceData, setVoiceData] = useState([]);
  // const voiceRecorderRef = useRef(null);
  
  const intervalRef = useRef(null);
  const lastImageDataRef = useRef(null);
  const abortControllerRef = useRef(null);
  const sourceLastImageDataRef = useRef({}); // Store last image data per source
  const lastCaptureTimeRef = useRef(0); // Track last capture time
  const lastVisibilityCheckRef = useRef(0); // Track last visibility check time
  const cachedVisibleSourcesRef = useRef(null); // Cache visible sources

  // Configuration (matches main.py defaults)
  const BASE_INTERVAL = 3000; // 1.5 seconds base interval
  const MULTI_APP_INTERVAL = 3000; // 3 seconds for multiple apps
  const SIMILARITY_THRESHOLD = 0.99;
  const VISIBILITY_CHECK_INTERVAL = 2000; // Check visibility every 2 seconds max (was 500ms - too frequent!)

  // Check screenshot permissions
  const checkScreenPermissions = useCallback(async () => {
    
    if (!window.electronAPI || !window.electronAPI.takeScreenshot) {
      setHasScreenPermission(false);
      setError(t('screenshot.errors.desktopOnly'));
      return false;
    }

    setIsCheckingPermission(true);
    setError(null);

    try {
      console.log('[ScreenshotMonitor] Checking permissions...');
      // Skip the unnecessary permission check screenshot
      // We'll find out about permissions when we actually need to take a screenshot
      const result = { success: true };
      
      if (result.success) {
        console.log('[ScreenshotMonitor] Permission check passed');
        setHasScreenPermission(true);
        // Clean up the test screenshot
        if (result.filepath) {
          try {
            await window.electronAPI.deleteScreenshot(result.filepath);
          } catch (cleanupError) {
            // Silent cleanup error
          }
        }
        return true;
      } else {
        console.error('[ScreenshotMonitor] Permission check failed:', result);
        setHasScreenPermission(false);
        if (result.error && result.error.includes('permission')) {
          setError(t('screenshot.errors.permissionDenied'));
        } else {
          setError(result.error || 'Failed to access screenshot functionality');
        }
        return false;
      }
    } catch (err) {
      console.error('[ScreenshotMonitor] Permission check exception:', err);
      setHasScreenPermission(false);
      if (err.message && err.message.includes('permission')) {
        setError(t('screenshot.errors.permissionDenied'));
      } else {
        setError(t('screenshot.errors.permissionCheckFailed', { error: err.message }));
      }
      return false;
    } finally {
      setIsCheckingPermission(false);
    }
  }, []);



  // Open System Preferences to Screen Recording section
  const openSystemPreferences = useCallback(async () => {
    if (!window.electronAPI || !window.electronAPI.openScreenRecordingPrefs) {
      setError(t('screenshot.errors.systemPrefsOnly'));
      return;
    }

    try {
      const result = await window.electronAPI.openScreenRecordingPrefs();
      if (result.success) {
        setError(null);
        // Check permissions again after a short delay to see if they were granted
        setTimeout(() => {
          checkScreenPermissions();
        }, 2000);
      } else {
        setError(result.message || t('screenshot.errors.systemPrefsFailed'));
      }
    } catch (err) {
      setError(t('screenshot.errors.systemPrefsError', { error: err.message }));
    }
  }, [checkScreenPermissions]);

  // Handle voice data from the recorder - COMMENTED OUT
  // const handleVoiceData = useCallback((data) => {
  //   setVoiceData(prev => [...prev, data]);
  // }, []);

  // Calculate image similarity using a simple pixel difference approach
  // Note: This is a simplified version compared to SSIM in main.py
  const calculateImageSimilarity = useCallback((imageData1, imageData2) => {
    if (!imageData1 || !imageData2) return 0;
    if (imageData1.length !== imageData2.length) return 0;

    let totalDiff = 0;
    const pixelCount = imageData1.length / 4; // RGBA channels

    for (let i = 0; i < imageData1.length; i += 4) {
      // Calculate grayscale values for comparison
      const gray1 = 0.299 * imageData1[i] + 0.587 * imageData1[i + 1] + 0.114 * imageData1[i + 2];
      const gray2 = 0.299 * imageData2[i] + 0.587 * imageData2[i + 1] + 0.114 * imageData2[i + 2];
      
      totalDiff += Math.abs(gray1 - gray2);
    }

    const averageDiff = totalDiff / pixelCount;
    const similarity = 1 - (averageDiff / 255); // Normalize to 0-1 range
    return Math.max(0, Math.min(1, similarity));
  }, []);

  // Convert canvas to image data for similarity comparison
  const getImageDataFromCanvas = useCallback((canvas) => {
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return imageData.data;
  }, []);

  // Delete screenshot that is too similar - only call this after backend processing is complete
  const deleteSimilarScreenshot = useCallback(async (filepath) => {
    if (!window.electronAPI || !window.electronAPI.deleteScreenshot) {
      return;
    }

    try {
      await window.electronAPI.deleteScreenshot(filepath);
    } catch (error) {
      // Silent error handling
    }
  }, []);

  // Send multiple screenshots to backend with sources information
  const sendScreenshotsToBackend = useCallback(async (imagePaths, sources) => {
    console.log('[ScreenshotMonitor] sendScreenshotsToBackend called with:', { 
      imageCount: imagePaths?.length, 
      sources,
      isRequestInProgress 
    });
    
    if (!imagePaths || imagePaths.length === 0 || isRequestInProgress) {
      console.log('[ScreenshotMonitor] Skipping send - invalid params or request in progress');
      return;
    }

    let currentAbortController = null;
    let cleanup = null;

    try {
      setIsRequestInProgress(true);
      setStatus('sending');
      
      const requestData = {
        image_uris: imagePaths,
        sources: sources, // New sources parameter
        memorizing: true,
        is_screen_monitoring: true
      };

      // Use a fresh abort controller for this request
      currentAbortController = new AbortController();
      abortControllerRef.current = currentAbortController;

      const result = await queuedFetch(`${settings.serverUrl}/send_streaming_message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
        signal: currentAbortController.signal,
        isStreaming: true
      });

      const response = result.response;
      cleanup = result.cleanup;

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Increment count immediately after successful response
      setScreenshotCount(prev => prev + imagePaths.length);
      setLastProcessedTime(new Date().toISOString());
      setStatus('monitoring');
      setError(null);

      // Consume the streaming response to complete the request
      try {
        if (response.body) {
          const reader = response.body.getReader();
          while (true) {
            const { done } = await reader.read();
            if (done) break;
          }
        }
      } catch (streamError) {
        console.warn('Error consuming streaming response:', streamError);
      }

      return { success: true, shouldDelete: false };

    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(t('screenshot.errors.screenshotsFailed', { error: err.message }));
      }
      return { success: false, shouldDelete: false };
    } finally {
      setIsRequestInProgress(false);
      if (abortControllerRef.current?.signal === currentAbortController?.signal) {
        abortControllerRef.current = null;
      }
      
      if (cleanup) {
        cleanup();
      }
    }
  }, [settings.serverUrl, isRequestInProgress]);

  // Send single screenshot to backend (legacy method for fullscreen)
  const sendScreenshotToBackend = useCallback(async (screenshotFile) => {
    if (!screenshotFile || isRequestInProgress) {
      return;
    }

    let currentAbortController = null;
    let cleanup = null;

    try {
      setIsRequestInProgress(true);
      setStatus('sending');
      
      // Get accumulated audio from voice recorder - COMMENTED OUT
      // let accumulatedAudio = [];
      // let voiceFiles = [];
      
      // if (voiceRecorderRef.current && typeof voiceRecorderRef.current.getAccumulatedAudio === 'function') {
      //   accumulatedAudio = voiceRecorderRef.current.getAccumulatedAudio();
      // }

      // Convert audio blobs to base64 for sending with screenshot - COMMENTED OUT
      // if (accumulatedAudio.length > 0) {
      //   try {
      //     for (const audioData of accumulatedAudio) {
      //       const arrayBuffer = await audioData.blob.arrayBuffer();
      //       const base64Data = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
      //       voiceFiles.push(base64Data);
      //     }
      //   } catch (audioError) {
      //     // Silent error handling
      //   }
      // }

      // Prepare the message with voice context info - COMMENTED OUT
      // let message = null;
      // if (voiceFiles.length > 0) {
      //   const totalDuration = accumulatedAudio.reduce((sum, audio) => sum + audio.duration, 0);
      //   message = `[Screenshot with voice recording: ${voiceFiles.length} audio chunks, ${(totalDuration/1000).toFixed(1)}s total]`;
      // }

      const requestData = {
        // message: message,
        image_uris: [screenshotFile.path],
        // voice_files: voiceFiles.length > 0 ? voiceFiles : null, // COMMENTED OUT
        memorizing: true, // This is the key difference from chat
        is_screen_monitoring: true // Indicate this request is from screen monitoring
      };

      // Use a fresh abort controller for this request
      currentAbortController = new AbortController();
      abortControllerRef.current = currentAbortController;

      const result = await queuedFetch(`${settings.serverUrl}/send_streaming_message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
        signal: currentAbortController.signal,
        isStreaming: true
      });

      const response = result.response;
      cleanup = result.cleanup;

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Increment count immediately after successful response
      setScreenshotCount(prev => prev + 1);
      setLastProcessedTime(new Date().toISOString());
      setStatus('monitoring');
      setError(null);

      // Consume the streaming response to complete the request
      // This is done after incrementing the count to ensure count updates even if streaming fails
      try {
        if (response.body) {
          const reader = response.body.getReader();
          while (true) {
            const { done } = await reader.read();
            if (done) break;
          }
        }
      } catch (streamError) {
        // Log streaming error but don't fail the whole request since we already counted it
        console.warn('Error consuming streaming response:', streamError);
      }

      // Clear accumulated audio after successful send - COMMENTED OUT
      // if (voiceRecorderRef.current && typeof voiceRecorderRef.current.clearAccumulatedAudio === 'function') {
      //   voiceRecorderRef.current.clearAccumulatedAudio();
      // }

      return { success: true, shouldDelete: false };

    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(t('screenshot.errors.screenshotFailed', { error: err.message }));
      }
      return { success: false, shouldDelete: false };
    } finally {
      setIsRequestInProgress(false);
      // Clear the abort controller if it's still the current one
      if (abortControllerRef.current?.signal === currentAbortController?.signal) {
        abortControllerRef.current = null;
      }
      
      // Call cleanup to notify request queue
      if (cleanup) {
        cleanup();
      }
    }
  }, [settings.serverUrl, isRequestInProgress]);

  // Check which sources are visible (with caching)
  const checkVisibleSources = useCallback(async (sources) => {
    const now = Date.now();
    const timeSinceLastCheck = now - lastVisibilityCheckRef.current;
    
    // Use cached visibility if recent enough
    if (cachedVisibleSourcesRef.current && timeSinceLastCheck < VISIBILITY_CHECK_INTERVAL) {
      return cachedVisibleSourcesRef.current;
    }
    
    try {
      // Get visibility for selected sources
      const sourceIds = sources.map(s => s.id);
      const result = await window.electronAPI.getVisibleSources(sourceIds);
      
      // Also get ALL visible sources for logging
      const allVisibleResult = await window.electronAPI.getVisibleSources();
      
      if (result.success) {
        const visibleMap = {};
        result.sources.forEach(s => {
          visibleMap[s.id] = s.isVisible;
        });
        
        // Store all visible sources for logging
        if (allVisibleResult.success) {
          visibleMap._allVisible = allVisibleResult.sources;
        }
        
        cachedVisibleSourcesRef.current = visibleMap;
        lastVisibilityCheckRef.current = now;
        return visibleMap;
      }
    } catch (error) {
      console.error('[ScreenshotMonitor] Error checking visibility:', error);
    }
    
    // If check fails, assume all are visible (fallback)
    const fallbackMap = {};
    sources.forEach(s => {
      fallbackMap[s.id] = true;
    });
    return fallbackMap;
  }, []);

  // Take and process a screenshot
  const processScreenshot = useCallback(async () => {
    
    if (!window.electronAPI) {
      const errorMsg = t('screenshot.errors.desktopRequired');
      console.error('[ScreenshotMonitor] Error:', errorMsg);
      setError(errorMsg);
      return;
    }

    // Skip if already processing a screenshot or if a request is in progress
    if (isProcessingScreenshot || isRequestInProgress) {
      return;
    }

    // Add a minimum delay between captures to prevent system overload
    const now = Date.now();
    const timeSinceLastCapture = now - lastCaptureTimeRef.current;
    // Use the appropriate interval based on mode
    const currentInterval = monitorMode === 'selected' && selectedSources.length > 1 ? MULTI_APP_INTERVAL : BASE_INTERVAL;
    const MIN_CAPTURE_INTERVAL = currentInterval - 100; // Slightly less than interval to account for timer drift
    
    if (timeSinceLastCapture < MIN_CAPTURE_INTERVAL) {
      console.log(`[ScreenshotMonitor] Skipping capture - too soon (${timeSinceLastCapture}ms < ${MIN_CAPTURE_INTERVAL}ms)`);
      return;
    }
    
    const timestamp = new Date().toISOString();
    console.log(`[ScreenshotMonitor] Starting capture process at ${timestamp} (${now}ms)`);
    lastCaptureTimeRef.current = now;

    setIsProcessingScreenshot(true); // Set flag before try
    
    try {
      setStatus('capturing');

      let result;
      let sourceInfo = null;
      
      if (monitorMode === 'selected' && selectedSources.length > 0) {
        // Check which sources are visible before capturing
        const visibilityMap = await checkVisibleSources(selectedSources);
        const visibleSources = selectedSources.filter(s => visibilityMap[s.id]);
        
        if (visibleSources.length === 0) {
          console.log('[ScreenshotMonitor] No selected apps are currently visible, skipping capture');
          setStatus('monitoring');
          setCurrentAppName(t('screenshot.monitoring.noAppsVisible'));
          setIsProcessingScreenshot(false); // CRITICAL: Reset the processing flag!
          return;
        }
        
        console.log(`[ScreenshotMonitor] ${visibleSources.length}/${selectedSources.length} selected sources are visible:`, visibleSources.map(s => s.name));
        
        // Step 1: Capture only visible sources
        const capturePromises = visibleSources.map(async (source) => {
          try {
            const captureTime = new Date().toISOString();
            console.log(`[ScreenshotMonitor] ${captureTime} - Capturing visible source: ${source.name} (${source.id})`);
            const captureResult = await window.electronAPI.takeSourceScreenshot(source.id);
            if (captureResult.success) {
              console.log(`[ScreenshotMonitor] Successfully captured: ${source.name}`);
              return {
                source: source,
                captureResult: captureResult,
                success: true
              };
            }
            console.warn(`[ScreenshotMonitor] Capture failed for ${source.name}:`, captureResult.error);
            return { source: source, success: false };
          } catch (error) {
            console.error(`[ScreenshotMonitor] Exception capturing ${source.name}:`, error);
            return { source: source, success: false };
          }
        });
        
        const captureResults = await Promise.all(capturePromises);
        console.log(`[ScreenshotMonitor] Capture results:`, captureResults.map(r => ({ name: r.source?.name, success: r.success })));
        
        // Step 2: Process each captured image for similarity check
        const validImages = [];
        const validSources = [];
        
        
        for (const result of captureResults) {
          if (!result.success) {
            continue;
          }
          
          const { source, captureResult } = result;
          
          // Normalize source ID for storage - strip the changing number from virtual window IDs
          let storageKey = source.id;
          if (source.id.startsWith('virtual-window:')) {
            // Extract just the app name part for consistent storage
            const appNameMatch = source.id.match(/virtual-window:\d+-(.+)$/);
            if (appNameMatch) {
              storageKey = `app-${appNameMatch[1]}`;
            }
          }
          
          // Skip similarity check if disabled
          if (skipSimilarityCheck) {
            console.log(`⏭️ Similarity check disabled for ${source.name}`);
            // Still validate that the file exists and has content before adding to validImages
            try {
              const fileExists = await window.electronAPI.readImageAsBase64(captureResult.filepath);
              if (fileExists.success && captureResult.filepath) {
                validImages.push(captureResult.filepath);
                validSources.push(source.name);
                console.log(`📸 Screenshot saved (similarity check skipped): ${captureResult.filepath} (${source.name})`);
              } else {
                console.error(`❌ File validation failed for ${source.name}: ${captureResult.filepath}`);
              }
            } catch (error) {
              console.error(`❌ File validation error for ${source.name}:`, error);
            }
            continue;
          }
          
          // If this is the first image for this source, always include it (but validate first)
          console.log(`🔍 Checking if first capture for ${source.name} (storage key: ${storageKey}). Has stored data: ${!!sourceLastImageDataRef.current[storageKey]}`);
          if (!sourceLastImageDataRef.current[storageKey]) {
            console.log(`🆕 First capture for ${source.name} - skipping similarity check`);
            // Validate file exists and is readable before adding to validImages
            const imageResult = await window.electronAPI.readImageAsBase64(captureResult.filepath);
            if (!imageResult.success || !captureResult.filepath) {
              console.error(`❌ First capture validation failed for ${source.name}: ${captureResult.filepath}`);
              continue;
            }
            
            console.log(`📸 Screenshot saved: ${captureResult.filepath} (${source.name})`);
            validImages.push(captureResult.filepath);
            validSources.push(source.name);
            
            // Store image data for future comparisons (reuse the imageResult from validation above)
            try {
              if (imageResult.success) {
                console.log(`💾 Storing image data for future comparisons: ${source.name} (${source.id})`);
                const img = new Image();
                await new Promise((resolve, reject) => {
                  img.onload = () => {
                    try {
                      const canvas = document.createElement('canvas');
                      canvas.width = img.naturalWidth;
                      canvas.height = img.naturalHeight;
                      const ctx = canvas.getContext('2d');
                      ctx.drawImage(img, 0, 0);
                      const currentImageData = getImageDataFromCanvas(canvas);
                      sourceLastImageDataRef.current[storageKey] = currentImageData;
                      console.log(`✅ Stored image data for ${source.name} (key: ${storageKey}), data length: ${currentImageData.length}`);
                      resolve();
                    } catch (error) {
                      console.error(`❌ Error processing image for storage: ${source.name}:`, error);
                      reject(error);
                    }
                  };
                  img.onerror = (error) => {
                    console.error(`❌ Failed to load image: ${source.name}:`, error);
                    reject(new Error('Failed to load image for similarity check'));
                  };
                  img.src = imageResult.dataUrl;
                });
              } else {
                console.error(`❌ imageResult not successful for ${source.name}`);
              }
            } catch (error) {
              console.error(`❌ Failed to store image data for ${source.name}:`, error);
            }
            continue;
          }
          
          // Compare with last image for this source
          console.log(`🔍 Running similarity check for ${source.name} (using key: ${storageKey})`);
          try {
            const imageResult = await window.electronAPI.readImageAsBase64(captureResult.filepath);
            if (imageResult.success) {
              const img = new Image();
              await new Promise((resolve, reject) => {
                img.onload = () => {
                  try {
                    const canvas = document.createElement('canvas');
                    canvas.width = img.naturalWidth;
                    canvas.height = img.naturalHeight;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0);
                    const currentImageData = getImageDataFromCanvas(canvas);
                    
                    const similarity = calculateImageSimilarity(sourceLastImageDataRef.current[storageKey], currentImageData);
                    console.log(`🔍 Similarity score for ${source.name}: ${similarity.toFixed(4)} (threshold: ${SIMILARITY_THRESHOLD})`);
                    
                    // Debug comparison image saving disabled
                    // if (similarity === 0) {
                    //   console.warn(`🐛 Suspicious similarity score of 0.0000 for ${source.name}`);
                    // }
                    
                    if (similarity < SIMILARITY_THRESHOLD) {
                      // Image is different enough, include it
                      console.log(`📸 Screenshot saved: ${captureResult.filepath} (${source.name})`);
                      validImages.push(captureResult.filepath);
                      validSources.push(source.name);
                      sourceLastImageDataRef.current[storageKey] = currentImageData;
                    } else {
                      // Image is too similar, delete it
                      console.log(`🗑️ Deleting similar screenshot: ${captureResult.filepath} (${source.name})`);
                      deleteSimilarScreenshot(captureResult.filepath);
                    }
                    resolve();
                  } catch (error) {
                    reject(error);
                  }
                };
                img.onerror = (error) => {
                  reject(new Error('Failed to load image for similarity comparison'));
                };
                img.src = imageResult.dataUrl;
              });
            } else {
              // If we can't read the image, skip it (don't delete, don't send)
              console.warn(`Failed to read image for similarity check: ${source.name}`);
            }
          } catch (error) {
            console.error(`Failed to process similarity check for ${source.name}:`, error);
            // Don't delete on processing errors - just skip it (don't send to backend)
          }
        }
        
        // Step 3: Send all valid images to backend in one request (if any)
        
        if (validImages.length > 0) {
          const sendTime = new Date().toISOString();
          console.log(`📤 ${sendTime} - Sending ${validImages.length} images to backend:`, validImages);
          const sendResult = await sendScreenshotsToBackend(validImages, validSources);
          if (!sendResult || !sendResult.success) {
            // Backend sending failed - but don't delete the images since they're valid/non-similar
            // They will be kept on disk for potential retry or manual review
            console.warn(`[ScreenshotMonitor] Failed to send ${validImages.length} valid images to backend, but keeping images on disk`);
          } else {
            console.log(`[ScreenshotMonitor] Successfully sent ${validImages.length} images`);
          }
        } else {
          console.log('[ScreenshotMonitor] No valid images to send (all were too similar)');
        }
        
        const statusMessage = visibleSources.length === 1 
          ? visibleSources[0].name 
          : t('screenshot.monitoring.appsVisible', { visible: visibleSources.length, total: selectedSources.length, sent: validImages.length });
        setCurrentAppName(statusMessage);
        setStatus('monitoring');
        return;
      } else {
        console.log('[ScreenshotMonitor] Taking full screen screenshot');
        
        // Take full screen screenshot
        result = await window.electronAPI.takeScreenshot();
        setCurrentAppName(t('screenshot.monitoring.fullScreen'));
        
        if (result.success) {
          // Validate fullscreen file before sending
          try {
            const fileValidation = await window.electronAPI.readImageAsBase64(result.filepath);
            if (!fileValidation.success || !result.filepath) {
              console.error(`❌ Fullscreen validation failed: ${result.filepath}`);
              setStatus('monitoring');
              return;
            }
            
            console.log(`📸 Screenshot saved: ${result.filepath} (Full Screen)`);
            // For fullscreen, send as single image with "Full Screen" source
            console.log('📤 Sending fullscreen image to backend:', [result.filepath]);
            const sendResult = await sendScreenshotsToBackend([result.filepath], [t('screenshot.monitoring.fullScreen')]);
            
            if (!sendResult || sendResult.shouldDelete) {
              console.log(`🗑️ Deleting fullscreen screenshot: ${result.filepath}`);
              await deleteSimilarScreenshot(result.filepath);
            }
          } catch (validationError) {
            console.error(`❌ Fullscreen validation error:`, validationError);
            setStatus('monitoring');
            return;
          }
          setStatus('monitoring');
          return;
        }
      }

    } catch (err) {
      console.error('[ScreenshotMonitor] ERROR: Screenshot processing failed:', err);
      console.error('[ScreenshotMonitor] Error details:', {
        message: err.message,
        stack: err.stack,
        monitorMode,
        selectedSourcesCount: selectedSources.length,
        isProcessingScreenshot,
        isRequestInProgress
      });
      setError(t('screenshot.errors.screenshotProcessing', { error: err.message }));
      
      // Reset processing state to allow retry
      setIsProcessingScreenshot(false);
      setIsRequestInProgress(false);
      
      // Clear any pending abort controllers
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
      
      setStatus('monitoring');
    } finally {
      setIsProcessingScreenshot(false);
    }
  }, [calculateImageSimilarity, getImageDataFromCanvas, sendScreenshotToBackend, sendScreenshotsToBackend, deleteSimilarScreenshot, skipSimilarityCheck, isRequestInProgress, isProcessingScreenshot, monitorMode, selectedSources, checkVisibleSources, BASE_INTERVAL, MULTI_APP_INTERVAL]);

  // Start monitoring
  const startMonitoring = useCallback(async () => {
    console.log('[ScreenshotMonitor] startMonitoring called');
    
    if (isMonitoring) {
      console.log('[ScreenshotMonitor] Already monitoring, skipping');
      return;
    }

    // Check permissions first
    console.log('[ScreenshotMonitor] Checking permissions...');
    const hasPermission = await checkScreenPermissions();
    if (!hasPermission) {
      console.error('[ScreenshotMonitor] No screen recording permission');
      return;
    }
    console.log('[ScreenshotMonitor] Permissions granted');

    setIsMonitoring(true);
    setStatus('monitoring');
    setError(null);
    setScreenshotCount(0);
    lastImageDataRef.current = null;
    // Don't reset sourceLastImageDataRef here - we want to keep previous image data for similarity comparison
    // sourceLastImageDataRef.current = {};
    lastCaptureTimeRef.current = 0; // Reset capture time
    
    console.log('[ScreenshotMonitor] Monitor settings:', {
      monitorMode,
      selectedSourcesCount: selectedSources.length,
      selectedSourceNames: selectedSources.map(s => s.name)
    });
    
    // Notify parent component about monitoring status change
    if (onMonitoringStatusChange) {
      onMonitoringStatusChange(true);
    }

    // Start the interval - use longer interval for multiple apps
    const interval = monitorMode === 'selected' && selectedSources.length > 1 ? MULTI_APP_INTERVAL : BASE_INTERVAL;
    const startTime = new Date().toISOString();
    console.log(`[ScreenshotMonitor] ${startTime} - Starting interval with ${interval}ms (${interval/1000}s)`);
    intervalRef.current = setInterval(processScreenshot, interval);

    // Take first screenshot immediately
    console.log('[ScreenshotMonitor] Taking first screenshot immediately');
    processScreenshot();
  }, [isMonitoring, processScreenshot, checkScreenPermissions, onMonitoringStatusChange, monitorMode, selectedSources.length]);

  // Stop monitoring
  const stopMonitoring = useCallback(() => {
    
    if (!isMonitoring) {
      return;
    }

    setIsMonitoring(false);
    setStatus('idle');

    // Clear interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    // Abort any pending request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    // Clear image data references
    lastImageDataRef.current = null;
    sourceLastImageDataRef.current = {};
    setCurrentAppName('');
    
    // Clear visibility cache
    cachedVisibleSourcesRef.current = null;
    lastVisibilityCheckRef.current = 0;
    
    // Reset request and processing state
    setIsRequestInProgress(false);
    setIsProcessingScreenshot(false);
    
    // Notify parent component about monitoring status change
    if (onMonitoringStatusChange) {
      onMonitoringStatusChange(false);
    }
  }, [isMonitoring, onMonitoringStatusChange]);

  // Check permissions on mount
  React.useEffect(() => {
    console.log('[ScreenshotMonitor] Component mounted, checking permissions');
    checkScreenPermissions();
  }, [checkScreenPermissions]);

  // Restart monitoring when selectedSources changes (if currently monitoring)
  // DISABLED: This effect was causing constant re-runs and clearing stored image data
  // React.useEffect(() => {
  //   console.log(`🔄 selectedSources effect triggered. isMonitoring: ${isMonitoring}, monitorMode: ${monitorMode}, sources:`, selectedSources.map(s => s.name));
  //   if (isMonitoring && monitorMode === 'selected') {
  //     // Stop current monitoring and reset all state
  //     if (intervalRef.current) {
  //       clearInterval(intervalRef.current);
  //       intervalRef.current = null;
  //     }
  //     // Abort any pending request
  //     if (abortControllerRef.current) {
  //       abortControllerRef.current.abort();
  //       abortControllerRef.current = null;
  //     }
  //     
  //     // Reset all processing states for clean restart
  //     setIsProcessingScreenshot(false);
  //     setIsRequestInProgress(false);
  //     setError(null);
  //     
  //     // Clear image data references for fresh start
  //     console.log(`🗑️ Clearing stored image data due to source change`);
  //     sourceLastImageDataRef.current = {};
  //     
  //     // Restart with new selection
  //     if (selectedSources.length > 0) {
  //       setStatus('monitoring');
  //       const interval = selectedSources.length > 1 ? MULTI_APP_INTERVAL : BASE_INTERVAL;
  //       intervalRef.current = setInterval(processScreenshot, interval);
  //       // Take first screenshot immediately with new selection
  //       processScreenshot();
  //     } else {
  //       // No apps selected, stop monitoring
  //       setIsMonitoring(false);
  //       setStatus('idle');
  //       setCurrentAppName('');
  //     }
  //   }
  // }, [selectedSources.map(s => s.id).join(','), isMonitoring, monitorMode, processScreenshot]);

  // Cleanup on unmount
  React.useEffect(() => {
    return () => {
      stopMonitoring();
    };
  }, [stopMonitoring]);

  const getStatusIcon = () => {
    switch (status) {
      case 'monitoring': return '👁️';
      case 'capturing': return '📸';
      case 'sending': return '📤';
      default: return '⏹️';
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'monitoring': return '#28a745';
      case 'capturing': return '#ffc107';
      case 'sending': return '#17a2b8';
      default: return '#6c757d';
    }
  };

  return (
    <div className="screenshot-monitor">
      <div className="monitor-header">
        <h3>🎯 {t('screenshot.title')}</h3>
        <div className="monitor-controls">
          {monitorMode === 'selected' && selectedSources.length > 0 && (
            <div className="selected-sources-info">
              {selectedSources.length > 1 ? (
                <div>
                  <div>{t('screenshot.monitoring.multipleApps', { count: selectedSources.length })}</div>
                  {isMonitoring && currentAppName && (
                    <div style={{ fontSize: '12px', color: '#666', marginTop: '2px' }}>
                      {t('screenshot.monitoring.statusInfo', { status: currentAppName })}
                    </div>
                  )}
                </div>
              ) : (
                <div>{t('screenshot.monitoring.singleApp', { appName: selectedSources[0]?.name })}</div>
              )}
            </div>
          )}
          {hasScreenPermission === false && (
            <button
              className="open-prefs-button"
              onClick={openSystemPreferences}
              disabled={false}
              style={{
                backgroundColor: '#dc3545',
                color: 'white',
                border: 'none',
                padding: '8px 16px',
                borderRadius: '4px',
                                  cursor: 'pointer',
                marginRight: '8px'
              }}
            >
              ⚙️ {t('screenshot.controls.openSystemPrefs')}
            </button>
          )}
          <button
            className="select-apps-button"
            onClick={() => setShowAppSelector(true)}
            disabled={isMonitoring}
            style={{
              backgroundColor: '#17a2b8',
              color: 'white',
              border: 'none',
              padding: '8px 16px',
              borderRadius: '4px',
              cursor: isMonitoring ? 'not-allowed' : 'pointer',
              marginRight: '8px',
              opacity: isMonitoring ? 0.5 : 1
            }}
                      >
              📱 {t('screenshot.controls.selectApps')}
            </button>
          <button
            className={`monitor-toggle ${isMonitoring ? 'active' : ''}`}
            onClick={() => {
              if (isMonitoring) {
                stopMonitoring();
              } else {
                startMonitoring();
              }
            }}
            disabled={hasScreenPermission === false || (monitorMode === 'selected' && selectedSources.length === 0)}
            style={{
              backgroundColor: isMonitoring ? '#dc3545' : hasScreenPermission === false || (monitorMode === 'selected' && selectedSources.length === 0) ? '#6c757d' : '#28a745',
              color: 'white',
              cursor: hasScreenPermission === false || (monitorMode === 'selected' && selectedSources.length === 0) ? 'not-allowed' : 'pointer'
            }}
          >
            {hasScreenPermission === false ? `🔒 ${t('screenshot.controls.permissionRequired')}` :
             monitorMode === 'selected' && selectedSources.length === 0 ? `📱 ${t('screenshot.controls.selectAppsFirst')}` :
             isMonitoring ? `⏹️ ${t('screenshot.controls.stopMonitor')}` : `▶️ ${t('screenshot.controls.startMonitor')}`}
          </button>
        </div>
      </div>

      <div className="monitor-status">
        <div className="status-item">
          <span className="status-icon" style={{ color: getStatusColor() }}>
            {getStatusIcon()}
          </span>
          <span className="status-text">
            {t('screenshot.status.status')}: <strong style={{ color: getStatusColor() }}>{t(`screenshot.status.${status}`)}</strong>
          </span>
        </div>
        
        <div className="status-item">
          <span className="permission-status">
            📋 {t('screenshot.status.permissions')}: <strong style={{ 
              color: hasScreenPermission === true ? '#28a745' : 
                     hasScreenPermission === false ? '#dc3545' : '#ffc107' 
            }}>
              {isCheckingPermission ? `⏳ ${t('screenshot.status.checking')}` :
               hasScreenPermission === true ? `✅ ${t('screenshot.status.granted')}` : 
               hasScreenPermission === false ? `❌ ${t('screenshot.status.denied')}` : `⏳ ${t('screenshot.status.checking')}`}
            </strong>
          </span>
        </div>
        
        <div className="status-item">
          <span>📊 {t('screenshot.status.screenshotsSent')}: <strong>{screenshotCount}</strong></span>
        </div>
        
        {lastProcessedTime && (
          <div className="status-item">
            <span>🕒 {t('screenshot.status.lastSent')}: <strong>{new Date(lastProcessedTime).toLocaleTimeString()}</strong></span>
          </div>
        )}
      </div>

      {error && (
        <div className="monitor-error">
          ⚠️ {error}
          {error.includes('permission') && (
            <div className="permission-help" style={{ marginTop: '8px', fontSize: '14px', color: '#6c757d' }}>
              <strong>{t('screenshot.permissions.helpTitle')}</strong> 
              <br />{t('screenshot.permissions.helpStep1')}
              <br />{t('screenshot.permissions.helpStep2')}
              <br />{t('screenshot.permissions.helpStep3')}
            </div>
          )}
        </div>
      )}

      {hasScreenPermission === false && !error && (
        <div className="monitor-warning" style={{ 
          backgroundColor: '#fff3cd', 
          color: '#856404', 
          padding: '12px', 
          borderRadius: '4px', 
          border: '1px solid #ffeaa7',
          marginTop: '12px'
        }}>
          🔒 {t('screenshot.permissions.warningTitle')}
          <br />
          <strong>{t('screenshot.permissions.warningAction')}</strong>
        </div>
      )}

      {/* Voice Recording Component - COMMENTED OUT */}
      {/* <VoiceRecorder 
        ref={voiceRecorderRef}
        settings={settings}
        isMonitoring={isMonitoring}
        onVoiceData={handleVoiceData}
      /> */}
      
      {showAppSelector && (
        <AppSelector
          onSourcesSelected={(sources) => {
            setSelectedSources(sources);
            setMonitorMode(sources.length > 0 ? 'selected' : 'fullscreen');
            setShowAppSelector(false);
          }}
          onClose={() => setShowAppSelector(false)}
        />
      )}
    </div>
  );
};

export default ScreenshotMonitor; 