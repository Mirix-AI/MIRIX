import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import './ChatBubble.css';
import { useTranslation } from 'react-i18next';

const ChatBubble = ({ message }) => {
  const { type, content, timestamp, images, isStreaming, thinkingSteps, memoryReferences } = message;
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

  const renderMemoryReferences = () => {
    if (!memoryReferences || memoryReferences.length === 0) return null;

    return (
      <div className="memory-references-section">
        <div className="memory-references-header">
          <span className="memory-icon">📚</span>
          <span className="memory-title">{t('chat.memoryReferences', { defaultValue: 'Memory Sources' })}</span>
          <span className="memory-count">{memoryReferences.length}</span>
        </div>
        <div className="memory-badges">
          {memoryReferences.map((ref, index) => {
            // Extract domain from URL
            const getDomain = (url) => {
              if (!url) return null;
              try {
                const urlObj = new URL(url.startsWith('http') ? url : `https://${url}`);
                return urlObj.hostname || urlObj.pathname.split('/')[0];
              } catch {
                return url;
              }
            };

            const domain = getDomain(ref.source_url);
            const capturedDate = ref.captured_at ? new Date(ref.captured_at).toLocaleDateString() : null;

            return (
              <div key={ref.id || index} className="memory-badge">
                <div className="memory-badge-icon">
                  {ref.source_app === 'Chrome' ? '🌐' :
                   ref.source_app === 'Safari' ? '🧭' :
                   ref.source_app === 'Firefox' ? '🦊' :
                   ref.source_app === 'Notion' ? '📝' : '💻'}
                </div>
                <div className="memory-badge-content">
                  <div className="memory-badge-app">{ref.source_app}</div>
                  {domain && <div className="memory-badge-url">{domain}</div>}
                  {capturedDate && <div className="memory-badge-date">{capturedDate}</div>}
                </div>
                {ref.ocr_text && (
                  <div className="memory-badge-preview" title={ref.ocr_text}>
                    {ref.ocr_text.substring(0, 100)}...
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    );
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

      {renderMemoryReferences()}

      {renderThinkingSteps()}
      
      <div className="bubble-content">
        {renderContent()}
      </div>
    </div>
  );
};

export default ChatBubble; 