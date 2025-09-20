import React from 'react';
import './Logo.css';

const Logo = ({ 
  size = 'medium', 
  showText = true, 
  className = '', 
  style = {},
  textColor = 'inherit',
  theme = 'auto' // 'light', 'dark', or 'auto'
}) => {
  // Use relative path to work in both dev and production
  const logoSrc = process.env.NODE_ENV === 'production' ? './logo.png' : '/logo.png';

  const logoSizes = {
    icon: { width: 32, height: 13 },
    small: { width: 80, height: 32 },
    medium: { width: 120, height: 48 },
    large: { width: 200, height: 81 }
  };

  const textSizes = {
    icon: '1rem',
    small: '1.5rem',
    medium: '2rem',
    large: '3rem'
  };

  // Auto-detect theme based on system preference if theme is 'auto'
  const getThemeClass = () => {
    if (theme === 'auto') {
      return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches 
        ? 'dark-theme' 
        : 'light-theme';
    }
    return `${theme}-theme`;
  };

  const themeClass = getThemeClass();

  return (
    <div className={`logo-container ${themeClass} ${className}`} style={style}>
      <img 
        src={logoSrc} 
        alt="MIRIX Logo" 
        className="logo-image"
        style={{
          width: logoSizes[size].width,
          height: logoSizes[size].height,
          objectFit: 'contain'
        }}
      />
      {showText && (
        <span 
          className="logo-text"
          style={{
            fontSize: textSizes[size],
            color: textColor,
            fontWeight: 'bold',
            marginLeft: size === 'icon' ? '8px' : '12px'
          }}
        >
          MIRIX
        </span>
      )}
    </div>
  );
};

export default Logo; 