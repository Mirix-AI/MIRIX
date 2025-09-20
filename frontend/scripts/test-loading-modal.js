#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🧪 Testing MIRIX Backend Loading Modal...');
console.log('');

// Check if the frontend components exist
const frontendSrc = path.join(__dirname, '..', 'src');
const components = [
  'components/BackendLoadingModal.js',
  'components/BackendLoadingModal.css'
];

console.log('📁 Checking frontend components:');
let allComponentsExist = true;

components.forEach(component => {
  const componentPath = path.join(frontendSrc, component);
  if (fs.existsSync(componentPath)) {
    const stats = fs.statSync(componentPath);
    console.log(`   ✅ ${component} (${(stats.size / 1024).toFixed(2)} KB)`);
  } else {
    console.log(`   ❌ ${component} - MISSING`);
    allComponentsExist = false;
  }
});

console.log('');

if (!allComponentsExist) {
  console.log('❌ Some components are missing. Please ensure all files are created.');
  process.exit(1);
}

// Check if App.js imports the BackendLoadingModal
console.log('🔍 Checking App.js integration:');
const appJsPath = path.join(frontendSrc, 'App.js');

if (fs.existsSync(appJsPath)) {
  const appJsContent = fs.readFileSync(appJsPath, 'utf8');
  
  const checks = [
    {
      name: 'BackendLoadingModal import',
      pattern: /import BackendLoadingModal from/,
      found: /import BackendLoadingModal from/.test(appJsContent)
    },
    {
      name: 'useCallback import',
      pattern: /useCallback/,
      found: /useCallback/.test(appJsContent)
    },
    {
      name: 'Backend loading state',
      pattern: /backendLoading/,
      found: /backendLoading/.test(appJsContent)
    },
    {
      name: 'Health check function',
      pattern: /checkBackendHealth/,
      found: /checkBackendHealth/.test(appJsContent)
    },
    {
      name: 'BackendLoadingModal component',
      pattern: /<BackendLoadingModal/,
      found: /<BackendLoadingModal/.test(appJsContent)
    }
  ];
  
  checks.forEach(check => {
    if (check.found) {
      console.log(`   ✅ ${check.name}`);
    } else {
      console.log(`   ❌ ${check.name} - NOT FOUND`);
    }
  });
  
  const allChecksPass = checks.every(check => check.found);
  
  console.log('');
  if (allChecksPass) {
    console.log('✅ All integration checks passed!');
  } else {
    console.log('❌ Some integration checks failed.');
  }
  
} else {
  console.log('   ❌ App.js not found');
}

console.log('');
console.log('🎯 Testing Instructions:');
console.log('1. Start the frontend development server:');
console.log('   cd frontend && npm start');
console.log('');
console.log('2. Stop the backend server (if running)');
console.log('');
console.log('3. Open http://localhost:3000 in your browser');
console.log('');
console.log('4. You should see the loading modal appear immediately');
console.log('');
console.log('5. Start the backend server to see the modal disappear:');
console.log('   cd .. && python main.py');
console.log('');
console.log('6. The modal should automatically hide when backend becomes available');
console.log('');
console.log('📱 Expected Behavior:');
console.log('• Modal appears with spinning loader and animated progress');
console.log('• Phase messages cycle through connecting → initializing → loading');
console.log('• Retry button appears if connection fails');
console.log('• Modal disappears when backend is healthy');
console.log('• Background is blurred with shadow overlay');
console.log('');
console.log('🔄 Window Reopening Test:');
console.log('1. Build and install the DMG: npm run electron-pack');
console.log('2. Open the MIRIX app');
console.log('3. Close the window (click X button - do NOT quit the app)');
console.log('4. Click the MIRIX icon in the dock to reopen');
console.log('5. Should show "Reconnecting MIRIX" modal briefly');
console.log('6. Modal should disappear when backend reconnects');
console.log('');
console.log('🎯 Reconnection vs Initial Loading:');
console.log('• Initial: "Loading MIRIX" with "Starting/Connecting/Initializing"');
console.log('• Reconnection: "Reconnecting MIRIX" with "Checking/Verifying/Syncing"');
console.log('• Different tips: "First startup..." vs "Reconnection usually..."'); 