import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import './ChatBubble.css';
import { useTranslation } from 'react-i18next';

const ChatBubble = ({ message }) => {
  const { type, content, timestamp, images, isStreaming, thinkingSteps } = message;
  const { t } = useTranslation();

  const formatTime = (date) => {
    // Handle both Date objects and ISO string timestamps
    const dateObj = typeof date === 'string' ? new Date(date) : date;
    return dateObj.toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit',
      hour12: true 
    });
  };

  const renderThinkingSteps = () => {
    if (!thinkingSteps || thinkingSteps.length === 0) return null;

    return (
      <div className="thinking-section">
        <div className="thinking-header">
          <span className="thinking-icon">🧠</span>
          <span className="thinking-title">{t('chat.thinkingTitle')}</span>
          <span className="thinking-count">{t('chat.steps', { count: thinkingSteps.length })}</span>
        </div>
        <div className="thinking-steps">
          {thinkingSteps.map((step, index) => (
            <div key={step.id} className="thinking-step">
              <div className="thinking-step-header">
                <span className="thinking-step-number">{index + 1}</span>
                <span className="thinking-step-time">{formatTime(step.timestamp)}</span>
              </div>
              <div className="thinking-step-content">
                <ReactMarkdown
                  components={{
                    code({node, inline, className, children, ...props}) {
                      const match = /language-(\w+)/.exec(className || '');
                      return !inline && match ? (
                        <SyntaxHighlighter
                          style={tomorrow}
                          language={match[1]}
                          PreTag="div"
                          {...props}
                        >
                          {String(children).replace(/\n$/, '')}
                        </SyntaxHighlighter>
                      ) : (
                        <code className={className} {...props}>
                          {children}
                        </code>
                      );
                    }
                  }}
                  className="thinking-markdown"
                >
                  {step.content}
                </ReactMarkdown>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderContent = () => {
    if (type === 'error') {
      return <div className="error-content">{content}</div>;
    }

    // Just render the markdown content directly since backend already sends markdown
    return (
      <ReactMarkdown
        components={{
          code({node, inline, className, children, ...props}) {
            const match = /language-(\w+)/.exec(className || '');
            return !inline && match ? (
              <SyntaxHighlighter
                style={tomorrow}
                language={match[1]}
                PreTag="div"
                {...props}
              >
                {String(children).replace(/\n$/, '')}
              </SyntaxHighlighter>
            ) : (
              <code className={className} {...props}>
                {children}
              </code>
            );
          }
        }}
        className="markdown-content"
      >
        {content || ''}
      </ReactMarkdown>
    );
  };

  return (
    <div className={`chat-bubble ${type}`}>
      <div className="bubble-header">
        <span className="sender">
          {type === 'user' ? `👤 ${t('chat.sender.you')}` : type === 'assistant' ? `🤖 ${t('chat.sender.assistant')}` : `❌ ${t('chat.sender.error')}`}
        </span>
        <span className="timestamp">{formatTime(timestamp)}</span>
        {isStreaming && <span className="streaming-indicator">●</span>}
      </div>
      
      {images && images.length > 0 && (
        <div className="message-images">
          {images.map((image, index) => {
            // Determine the correct image source
            let imageSrc;
            if (image.displayUrl) {
              // Use displayUrl if available (base64 data URL for secure display)
              imageSrc = image.displayUrl;
            } else if (image.url) {
              // Use existing URL (blob URLs, etc.)
              imageSrc = image.url;
            } else if (image.path) {
              // Check if path is a base64 data URL
              if (image.path.startsWith('data:')) {
                imageSrc = image.path; // Use base64 data URL directly
              } else {
                // For file paths, use file:// protocol (though this may be blocked)
                imageSrc = `file://${image.path}`;
              }
            } else {
              // Fallback to name
              imageSrc = image.name;
            }

            return (
              <div key={index} className="image-preview">
                <img 
                  src={imageSrc}
                  alt={t('chat.attachmentAlt', { index: index + 1 })}
                  onError={(e) => {
                    // If file:// URL doesn't work, try without protocol for electron
                    if (image.path && e.target.src.startsWith('file://') && !image.path.startsWith('data:')) {
                      e.target.src = image.path;
                    }
                  }}
                  onLoad={(e) => {
                    // Revoke object URL after loading to prevent memory leaks
                    if (image.url && image.url.startsWith('blob:')) {
                      URL.revokeObjectURL(image.url);
                    }
                  }}
                />
                <span className="image-name">{image.name}</span>
              </div>
            );
          })}
        </div>
      )}
      
      {renderThinkingSteps()}
      
      <div className="bubble-content">
        {renderContent()}
      </div>
    </div>
  );
};

export default ChatBubble; 