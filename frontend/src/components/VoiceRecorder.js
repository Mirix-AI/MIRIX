import React, { useState, useRef, useCallback, forwardRef, useImperativeHandle } from 'react';
import './VoiceRecorder.css';

const VoiceRecorder = forwardRef(({ settings, isMonitoring, onVoiceData }, ref) => {
  const [isRecording, setIsRecording] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [error, setError] = useState(null);
  const [totalRecorded, setTotalRecorded] = useState(0);
  const [accumulatedAudio, setAccumulatedAudio] = useState([]);
  
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const analyzerRef = useRef(null);
  const intervalRef = useRef(null);
  const recordingChunksRef = useRef([]);
  const recordingStartTimeRef = useRef(null);

  // Configuration
  const CHUNK_DURATION = 5000; // 5 seconds - shorter chunks for better accumulation
  const SILENCE_THRESHOLD = 0.01;
  const SILENCE_DURATION = 3000;

  // Get accumulated audio for sending with screenshots
  const getAccumulatedAudio = useCallback(() => {
    return accumulatedAudio;
  }, [accumulatedAudio]);

  // Clear accumulated audio (called after sending)
  const clearAccumulatedAudio = useCallback(() => {
    setAccumulatedAudio([]);
    setTotalRecorded(0);
  }, []);

  // Expose methods to parent component via ref
  useImperativeHandle(ref, () => ({
    getAccumulatedAudio,
    clearAccumulatedAudio
  }));

  // Initialize audio level monitoring
  const initializeAudioLevelMonitoring = useCallback((stream) => {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(stream);
    const analyzer = audioContext.createAnalyser();
    
    analyzer.fftSize = 256;
    source.connect(analyzer);
    
    analyzerRef.current = { analyzer, audioContext };
    
    const dataArray = new Uint8Array(analyzer.frequencyBinCount);
    
    const updateAudioLevel = () => {
      if (analyzerRef.current) {
        analyzer.getByteFrequencyData(dataArray);
        const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
        const level = average / 255; // Normalize to 0-1
        setAudioLevel(level);
      }
    };
    
    intervalRef.current = setInterval(updateAudioLevel, 100);
  }, []);

  // Accumulate audio chunk locally
  const accumulateAudioChunk = useCallback(async (audioBlob, duration) => {
    if (!audioBlob || audioBlob.size === 0) return;

    const timestamp = new Date().toISOString();
    const audioData = {
      blob: audioBlob,
      duration: duration,
      timestamp: timestamp,
      size: audioBlob.size
    };

    setAccumulatedAudio(prev => [...prev, audioData]);
    setTotalRecorded(prev => prev + duration);

    if (onVoiceData) {
      onVoiceData({
        audioData: audioData,
        totalChunks: accumulatedAudio.length + 1,
        totalDuration: totalRecorded + duration,
        timestamp: timestamp
      });
    }
  }, [accumulatedAudio.length, totalRecorded, onVoiceData]);

  // Process recorded chunks
  const processRecordedChunks = useCallback(async () => {
    if (recordingChunksRef.current.length === 0) return;

    const audioBlob = new Blob(recordingChunksRef.current, { type: 'audio/webm' });
    const duration = Date.now() - recordingStartTimeRef.current;
    
    recordingChunksRef.current = [];
    recordingStartTimeRef.current = Date.now();
    
    await accumulateAudioChunk(audioBlob, duration);
  }, [accumulateAudioChunk]);

  // Start recording
  const startRecording = useCallback(async () => {
    try {
      setError(null);
      
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        } 
      });
      
      streamRef.current = stream;
      
      // Initialize audio level monitoring
      initializeAudioLevelMonitoring(stream);
      
      // Setup media recorder
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm'
      });
      
      mediaRecorderRef.current = mediaRecorder;
      recordingChunksRef.current = [];
      recordingStartTimeRef.current = Date.now();
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordingChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        processRecordedChunks();
      };
      
      // Start recording with time slices for chunk management
      mediaRecorder.start(CHUNK_DURATION);
      setIsRecording(true);
      
      // Handle periodic chunk processing
      const chunkInterval = setInterval(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
          mediaRecorderRef.current.stop();
          mediaRecorderRef.current.start(CHUNK_DURATION);
        }
      }, CHUNK_DURATION);
      
      intervalRef.current = chunkInterval;
      
    } catch (err) {
      console.error('Error starting recording:', err);
      setError(`Failed to start recording: ${err.message}`);
    }
  }, [initializeAudioLevelMonitoring, processRecordedChunks]);

  // Stop recording
  const stopRecording = useCallback(() => {
    setIsRecording(false);
    
    // Stop media recorder
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    
    // Stop media stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    // Clean up audio context
    if (analyzerRef.current) {
      analyzerRef.current.audioContext.close();
      analyzerRef.current = null;
    }
    
    // Clear intervals
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    
    setAudioLevel(0);
  }, []);

  // Effect to sync with monitoring state
  React.useEffect(() => {
    if (isMonitoring && !isRecording) {
      startRecording();
    } else if (!isMonitoring && isRecording) {
      stopRecording();
    }
  }, [isMonitoring, isRecording, startRecording, stopRecording]);

  // Cleanup on unmount
  React.useEffect(() => {
    return () => {
      stopRecording();
    };
  }, [stopRecording]);

  const formatDuration = (ms) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const totalSize = accumulatedAudio.reduce((sum, audio) => sum + audio.size, 0);

  return (
    <div className="voice-recorder">
      <div className="recorder-header">
        <h4>🎤 Voice Monitor</h4>
        <div className="recorder-status">
          {isRecording ? (
            <span className="recording-indicator">🔴 Recording</span>
          ) : (
            <span className="idle-indicator">⏸️ Idle</span>
          )}
        </div>
      </div>

      <div className="audio-visualizer">
        <div className="audio-level-container">
          <div className="audio-level-label">Audio Level:</div>
          <div className="audio-level-bar">
            <div 
              className="audio-level-fill"
              style={{ width: `${audioLevel * 100}%` }}
            />
          </div>
          <div className="audio-level-text">{Math.round(audioLevel * 100)}%</div>
        </div>
      </div>

      <div className="recorder-stats">
        <div className="stat-item">
          <span>📊 Total recorded: <strong>{formatDuration(totalRecorded)}</strong></span>
        </div>
        <div className="stat-item">
          <span>📦 Audio chunks: <strong>{accumulatedAudio.length}</strong></span>
        </div>
        <div className="stat-item">
          <span>💾 Total size: <strong>{formatFileSize(totalSize)}</strong></span>
        </div>
        {isRecording && (
          <div className="stat-item">
            <span>⏱️ Current session: <strong>{new Date().toLocaleTimeString()}</strong></span>
          </div>
        )}
      </div>

      {error && (
        <div className="recorder-error">
          ⚠️ {error}
        </div>
      )}
    </div>
  );
});

VoiceRecorder.displayName = 'VoiceRecorder';

export default VoiceRecorder; 