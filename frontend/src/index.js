import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import './i18n';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  // Temporarily disable StrictMode to avoid double-rendering issues with health checks
  // <React.StrictMode>
    <App />
  // </React.StrictMode>
); 