import React, { useState, useRef, useEffect } from 'react';
import './MessageInput.css';
import { useTranslation } from 'react-i18next';

const MessageInput = ({ onSendMessage, disabled, onScreenshotTaken }) => {
  const [message, setMessage] = useState('');
  const [selectedFiles, setSelectedFiles] = useState([]);
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);
  const containerRef = useRef(null);
  const { t } = useTranslation();

  useEffect(() => {
    // Listen for custom addScreenshot events
    const handleAddScreenshot = (event) => {
      const screenshotFile = event.detail;
      setSelectedFiles(prev => [...prev, screenshotFile]);
    };

    const container = containerRef.current;
    if (container) {
      container.addEventListener('addScreenshot', handleAddScreenshot);
    }

    return () => {
      if (container) {
        container.removeEventListener('addScreenshot', handleAddScreenshot);
      }
    };
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    if ((message.trim() || selectedFiles.length > 0) && !disabled) {
      onSendMessage(message, selectedFiles);
      setMessage('');
      setSelectedFiles([]);
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyDown = (e) => {
    // Handle Ctrl+A / Cmd+A explicitly
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'a') {
      console.log('Select all triggered');
      // Explicitly select all text in the textarea
      e.preventDefault();
      if (textareaRef.current) {
        textareaRef.current.select();
        textareaRef.current.setSelectionRange(0, textareaRef.current.value.length);
      }
      return;
    }
    
    // Handle Enter key
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleTextareaChange = (e) => {
    setMessage(e.target.value);
    
    // Auto-resize textarea
    const textarea = e.target;
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
  };

  const handleFileSelect = async () => {
    if (window.electronAPI) {
      // Use Electron's file dialog
      try {
        const filePaths = await window.electronAPI.selectFiles();
        if (filePaths && filePaths.length > 0) {
          const files = filePaths.map(path => ({
            name: path.split(/[\\/]/).pop(),
            path: path,
            type: 'image' // Assume images for now
          }));
          setSelectedFiles(prev => [...prev, ...files]);
        }
      } catch (error) {
        console.error('Error selecting files:', error);
      }
    } else {
      // Use web file input
      fileInputRef.current?.click();
    }
  };

  const handleFileInputChange = (e) => {
    const files = Array.from(e.target.files);
    const fileObjects = files.map(file => ({
      name: file.name,
      file: file,
      url: URL.createObjectURL(file),
      type: file.type.startsWith('image/') ? 'image' : 'file'
    }));
    setSelectedFiles(prev => [...prev, ...fileObjects]);
    
    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const removeFile = (index) => {
    setSelectedFiles(prev => {
      const newFiles = [...prev];
      const removedFile = newFiles.splice(index, 1)[0];
      
      // Revoke object URL to prevent memory leaks
      if (removedFile.url && removedFile.url.startsWith('blob:')) {
        URL.revokeObjectURL(removedFile.url);
      }
      
      return newFiles;
    });
  };

  const getFileDisplayName = (file) => {
    if (file.isScreenshot) {
      return `📸 ${file.name}`;
    }
    return file.name;
  };

  const getFilePreview = (file) => {
    if (file.isScreenshot && file.url) {
      return file.url;
    }
    if (file.url) {
      return file.url;
    }
    return null;
  };

  return (
    <div ref={containerRef} className="message-input-container">
      {selectedFiles.length > 0 && (
        <div className="selected-files">
          {selectedFiles.map((file, index) => (
            <div key={index} className={`file-chip ${file.isScreenshot ? 'screenshot-chip' : ''}`}>
              {file.isScreenshot && (
                <span className="screenshot-indicator">📸</span>
              )}
              <span className="file-name">{getFileDisplayName(file)}</span>
              <button 
                type="button"
                className="remove-file"
                onClick={() => removeFile(index)}
                title={t('messageInput.removeFileTitle')}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}
      
      <form onSubmit={handleSubmit} className="message-form">
        <div className="input-row">
          <button
            type="button"
            className="attach-button"
            onClick={handleFileSelect}
            disabled={disabled}
            title={t('messageInput.attachFilesTitle')}
          >
            📎
          </button>
          
          <textarea
            ref={textareaRef}
            value={message}
            onChange={handleTextareaChange}
            onKeyDown={handleKeyDown}
            placeholder={t('messageInput.placeholder')}
            disabled={disabled}
            className="message-textarea"
            rows={1}
            tabIndex={0}
            autoComplete="off"
            spellCheck="true"
          />
          
          <button
            type="submit"
            disabled={disabled || (!message.trim() && selectedFiles.length === 0)}
            className="send-button"
            title={t('messageInput.sendTitle')}
          >
            {disabled ? '⏳' : '➤'}
          </button>
        </div>
      </form>
      
      {/* Hidden file input for web */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept="image/*"
        onChange={handleFileInputChange}
        style={{ display: 'none' }}
      />
    </div>
  );
};

export default MessageInput; 