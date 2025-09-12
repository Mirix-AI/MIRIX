const { app, BrowserWindow, Menu, shell, ipcMain, dialog, systemPreferences } = require('electron');
const path = require('path');
const fs = require('fs');
const os = require('os');
const isDev = require('electron-is-dev');
const { spawn, exec } = require('child_process');
const screenshot = require('screenshot-desktop');
const http = require('http');
const https = require('https');
const NativeCaptureHelper = require('./nativeCaptureHelper');

// Override isDev for packaged apps
const isPackaged = app.isPackaged || 
                  (process.mainModule && process.mainModule.filename.indexOf('app.asar') !== -1) ||
                  (require.main && require.main.filename.indexOf('app.asar') !== -1) ||
                  process.execPath.indexOf('MIRIX.app') !== -1 ||
                  __dirname.indexOf('app.asar') !== -1;
const actuallyDev = isDev && !isPackaged;

const safeLog = {
  log: (...args) => {
    if (actuallyDev) {
      console.log(...args);
    }
  },
  error: (...args) => {
    if (actuallyDev) {
      console.error(...args);
    }
  },
  warn: (...args) => {
    if (actuallyDev) {
      console.warn(...args);
    }
  }
};

let mainWindow;
let backendProcess = null;
const backendPort = 47283;
let backendLogFile = null;
let nativeCaptureHelper = null;

// Create screenshots directory
function ensureScreenshotDirectory() {
  const mirixDir = path.join(os.homedir(), '.mirix');
  const tmpDir = path.join(mirixDir, 'tmp');
  const imagesDir = path.join(tmpDir, 'images');
    
  if (!fs.existsSync(mirixDir)) {
    fs.mkdirSync(mirixDir, { recursive: true });
  }
  if (!fs.existsSync(tmpDir)) {
    fs.mkdirSync(tmpDir, { recursive: true });
  }
  if (!fs.existsSync(imagesDir)) {
    fs.mkdirSync(imagesDir, { recursive: true });
  }
  
  return imagesDir;
}

// Create debug images directory for development - DISABLED
// function ensureDebugImagesDirectory() {
//   const mirixDir = path.join(os.homedir(), '.mirix');
//   const debugDir = path.join(mirixDir, 'debug');
//   const debugImagesDir = path.join(debugDir, 'images');
//     
//   if (!fs.existsSync(mirixDir)) {
//     fs.mkdirSync(mirixDir, { recursive: true });
//   }
//   if (!fs.existsSync(debugDir)) {
//     fs.mkdirSync(debugDir, { recursive: true });
//   }
//   if (!fs.existsSync(debugImagesDir)) {
//     fs.mkdirSync(debugImagesDir, { recursive: true });
//   }
//   
//   return debugImagesDir;
// }

// Create debug comparison directory - DISABLED
// function ensureDebugCompareDirectory() {
//   const debugImagesDir = ensureDebugImagesDirectory();
//   const compareDir = path.join(debugImagesDir, 'compare');
//   
//   if (!fs.existsSync(compareDir)) {
//     fs.mkdirSync(compareDir, { recursive: true });
//   }
//   
//   return compareDir;
// }

// Helper function to save debug copy of an image - DISABLED
// function saveDebugCopy(sourceFilePath, debugName, sourceName = '') {
//   try {
//     const debugImagesDir = ensureDebugImagesDirectory();
//     const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
//     const sanitizedSourceName = sourceName.replace(/[^a-zA-Z0-9\-_]/g, '_');
//     const debugFileName = `${timestamp}_${debugName}_${sanitizedSourceName}.png`;
//     const debugFilePath = path.join(debugImagesDir, debugFileName);
//     
//     if (fs.existsSync(sourceFilePath)) {
//       fs.copyFileSync(sourceFilePath, debugFilePath);
//       safeLog.log(`✅ Debug copy saved: ${debugFilePath}`);
//     } else {
//       safeLog.warn(`Source file does not exist for debug copy: ${sourceFilePath}`);
//     }
//   } catch (error) {
//     safeLog.warn(`Failed to save debug copy: ${error.message}`);
//     safeLog.warn(`Error stack: ${error.stack}`);
//   }
// }

// Create backend log file
function createBackendLogFile() {
  const debugLogDir = path.join(os.homedir(), '.mirix', 'debug');
  if (!fs.existsSync(debugLogDir)) {
    fs.mkdirSync(debugLogDir, { recursive: true });
  }
  
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const logFileName = `backend-${timestamp}.log`;
  const logFilePath = path.join(debugLogDir, logFileName);
  
  // Create the log file with initial headers
  const initialLog = `=== MIRIX Backend Debug Log ===
Started: ${new Date().toISOString()}
Platform: ${process.platform}
Architecture: ${process.arch}
Node version: ${process.version}
Electron version: ${process.versions.electron}
Process execPath: ${process.execPath}
Process cwd: ${process.cwd()}
__dirname: ${__dirname}
Resources path: ${process.resourcesPath}
Is packaged: ${isPackaged}
Actually dev: ${actuallyDev}
========================================

`;
  
  fs.writeFileSync(logFilePath, initialLog);
  safeLog.log(`Created backend log file: ${logFilePath}`);
  
  return logFilePath;
}

// Helper function to log to backend log file
function logToBackendFile(message) {
  if (!backendLogFile) {
    backendLogFile = createBackendLogFile();
  }
  
  const timestamp = new Date().toISOString();
  const logMessage = `[${timestamp}] ${message}`;
  
  safeLog.log(logMessage);
  
  try {
    fs.appendFileSync(backendLogFile, logMessage + '\n');
  } catch (error) {
    safeLog.error('Failed to write to backend log file:', error);
  }
}

// Check if backend is running and healthy
async function isBackendHealthy() {
  try {
    const healthCheckResult = await checkBackendHealth();
    return true;
  } catch (error) {
    return false;
  }
}

// Ensure backend is running (start if not running)
async function ensureBackendRunning() {
  if (actuallyDev) {
    safeLog.log('Development mode: Backend should be running separately');
    return;
  }
  
  // Check if backend process is still running
  if (backendProcess && backendProcess.exitCode === null) {
    // Process is still running, check if it's healthy
    const isHealthy = await isBackendHealthy();
    if (isHealthy) {
      logToBackendFile('Backend is already running and healthy');
      return;
    } else {
      logToBackendFile('Backend process is running but not healthy, restarting...');
      stopBackendServer();
    }
  } else {
    logToBackendFile('Backend process is not running, starting...');
  }
  
  // Start the backend
  try {
    await startBackendServer();
    logToBackendFile('Backend started successfully');
  } catch (error) {
    logToBackendFile(`Failed to start backend: ${error.message}`);
    throw error;
  }
}

function startBackendServer() {
  if (actuallyDev) {
    safeLog.log('Development mode: Backend should be running separately');
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    try {
      const executableName = 'main';
      
      // Fix resourcesPath for packaged apps with detailed logging
      let actualResourcesPath = process.resourcesPath;
      logToBackendFile(`Initial resources path: ${actualResourcesPath}`);
      
      if (__dirname.indexOf('app.asar') !== -1) {
        const appAsarPath = __dirname.substring(0, __dirname.indexOf('app.asar'));
        actualResourcesPath = appAsarPath;
        logToBackendFile(`Adjusted resources path from asar: ${actualResourcesPath}`);
      }
      
      // Try multiple possible backend paths
      const possiblePaths = [
        path.join(actualResourcesPath, 'backend', executableName),
        path.join(actualResourcesPath, 'app', 'backend', executableName),
        path.join(actualResourcesPath, 'Contents', 'Resources', 'backend', executableName),
        path.join(actualResourcesPath, 'Contents', 'Resources', 'app', 'backend', executableName),
        path.join(process.resourcesPath, 'backend', executableName),
        path.join(process.resourcesPath, 'app', 'backend', executableName),
      ];
      
      logToBackendFile(`Searching for backend executable in ${possiblePaths.length} locations:`);
      
      let backendPath = null;
      for (const candidatePath of possiblePaths) {
        logToBackendFile(`  Checking: ${candidatePath}`);
        if (fs.existsSync(candidatePath)) {
          const stats = fs.statSync(candidatePath);
          logToBackendFile(`  ✅ Found! Size: ${stats.size} bytes, Modified: ${stats.mtime}`);
          logToBackendFile(`  File mode: ${stats.mode.toString(8)} (executable: ${(stats.mode & parseInt('111', 8)) !== 0})`);
          
          // Make sure it's executable
          if ((stats.mode & parseInt('111', 8)) === 0) {
            try {
              fs.chmodSync(candidatePath, '755');
              logToBackendFile(`  Made executable: ${candidatePath}`);
            } catch (chmodError) {
              logToBackendFile(`  Failed to make executable: ${chmodError.message}`);
            }
          }
          
          backendPath = candidatePath;
          break;
        } else {
          logToBackendFile(`  ❌ Not found`);
        }
      }
      
      if (!backendPath) {
        const error = `Backend executable not found in any of the expected locations:\n${possiblePaths.join('\n')}`;
        logToBackendFile(error);
        reject(new Error(error));
        return;
      }
      
      logToBackendFile(`Starting backend server on port ${backendPort}: ${backendPath}`);
      
      // Use user's .mirix directory as working directory (for .env files and SQLite database)
      const userMirixDir = path.join(os.homedir(), '.mirix');
      if (!fs.existsSync(userMirixDir)) {
        fs.mkdirSync(userMirixDir, { recursive: true });
        logToBackendFile(`Created working directory: ${userMirixDir}`);
      }
      const workingDir = userMirixDir;
      logToBackendFile(`Using working directory: ${workingDir}`);
      
      // Copy config files to working directory
      const configsDir = path.join(workingDir, 'configs');
      if (!fs.existsSync(configsDir)) {
        fs.mkdirSync(configsDir, { recursive: true });
        logToBackendFile(`Created configs directory: ${configsDir}`);
      }
      
      const sourceConfigsDir = path.join(actualResourcesPath, 'backend', 'configs');
      if (fs.existsSync(sourceConfigsDir)) {
        logToBackendFile(`Copying config files from: ${sourceConfigsDir}`);
        const configFiles = fs.readdirSync(sourceConfigsDir);
        for (const configFile of configFiles) {
          const sourcePath = path.join(sourceConfigsDir, configFile);
          const targetPath = path.join(configsDir, configFile);
          try {
            fs.copyFileSync(sourcePath, targetPath);
            logToBackendFile(`✅ Copied config: ${configFile}`);
          } catch (error) {
            logToBackendFile(`❌ Failed to copy config ${configFile}: ${error.message}`);
          }
        }
      } else {
        logToBackendFile(`❌ Source configs directory not found: ${sourceConfigsDir}`);
      }
      
      // Prepare environment variables
      const env = {
        ...process.env,
        PORT: backendPort.toString(),
        PYTHONPATH: workingDir,
        MIRIX_PG_URI: '', // Force SQLite fallback
        DEBUG: 'true',
        MIRIX_DEBUG: 'true',
        MIRIX_LOG_LEVEL: 'DEBUG'
      };
      
      logToBackendFile(`Environment variables: PORT=${env.PORT}, PYTHONPATH=${env.PYTHONPATH}, MIRIX_PG_URI=${env.MIRIX_PG_URI}`);
      
      // Start backend with SQLite configuration
      backendProcess = spawn(backendPath, ['--host', '0.0.0.0', '--port', backendPort.toString()], {
        stdio: ['pipe', 'pipe', 'pipe'],
        detached: false,
        cwd: workingDir,
        env: env
      });

      let healthCheckStarted = false;

      backendProcess.stdout.on('data', (data) => {
        const output = data.toString().trim();
        logToBackendFile(`STDOUT: ${output}`);
        
        if (output.includes('Uvicorn running on') || 
            output.includes('Application startup complete') ||
            output.includes('Started server process')) {
          
          if (!healthCheckStarted) {
            healthCheckStarted = true;
            logToBackendFile('Backend server startup detected, starting health check...');
            setTimeout(() => {
              checkBackendHealth().then(() => {
                logToBackendFile('Backend health check passed, resolving startup');
                resolve();
              }).catch((healthError) => {
                logToBackendFile(`Backend health check failed: ${healthError.message}`);
                reject(healthError);
              });
            }, 3000);
          }
        }
      });

      backendProcess.stderr.on('data', (data) => {
        const output = data.toString();
        logToBackendFile(`STDERR: ${output}`);
        
        // Check stderr for startup messages too
        if (output.includes('Uvicorn running on') || 
            output.includes('Application startup complete') ||
            output.includes('Started server process')) {
          
          if (!healthCheckStarted) {
            healthCheckStarted = true;
            logToBackendFile('Backend server startup detected in stderr, starting health check...');
            setTimeout(() => {
              checkBackendHealth().then(() => {
                logToBackendFile('Backend health check passed, resolving startup');
                resolve();
              }).catch((healthError) => {
                logToBackendFile(`Backend health check failed: ${healthError.message}`);
                reject(healthError);
              });
            }, 3000);
          }
        }
      });

      backendProcess.on('close', (code) => {
        logToBackendFile(`Backend process exited with code ${code}`);
        if (code !== 0 && !healthCheckStarted) {
          reject(new Error(`Backend process exited with code ${code}`));
        }
      });

      backendProcess.on('error', (error) => {
        logToBackendFile(`Failed to start backend process: ${error.message}`);
        reject(error);
      });

      // Timeout fallback
      setTimeout(() => {
        if (backendProcess && backendProcess.exitCode === null && !healthCheckStarted) {
          logToBackendFile('Backend startup timeout, trying health check...');
          checkBackendHealth().then(() => {
            logToBackendFile('Health check passed despite timeout');
            resolve();
          }).catch((healthError) => {
            logToBackendFile(`Backend health check failed after timeout: ${healthError.message}`);
            reject(new Error(`Backend startup timeout: ${healthError.message}`));
          });
        }
      }, 30000);

      logToBackendFile('Backend server started');
    } catch (error) {
      safeLog.error('Failed to start backend server:', error);
      reject(error);
    }
  });
}

async function checkBackendHealth() {
  const maxRetries = 20;
  const retryDelay = 20000;
  
  for (let i = 0; i < maxRetries; i++) {
    try {
      logToBackendFile(`Health check attempt ${i + 1}/${maxRetries} - checking http://127.0.0.1:${backendPort}/health`);
      
      const healthCheckResult = await new Promise((resolve, reject) => {
        const req = http.get(`http://127.0.0.1:${backendPort}/health`, { timeout: 5000 }, (res) => {
          let data = '';
          
          res.on('data', chunk => {
            data += chunk;
          });
          
          res.on('end', () => {
            if (res.statusCode === 200) {
              logToBackendFile(`Health check response: ${data}`);
              resolve(data);
            } else {
              reject(new Error(`Health check failed with status: ${res.statusCode}, response: ${data}`));
            }
          });
        });
        
        req.on('error', (error) => {
          logToBackendFile(`Health check request error: ${error.message}`);
          reject(error);
        });
        
        req.setTimeout(5000, () => {
          req.destroy();
          reject(new Error('Health check timeout after 5 seconds'));
        });
      });
      
      logToBackendFile('✅ Backend health check passed');
      return healthCheckResult;
      
    } catch (error) {
      logToBackendFile(`❌ Health check attempt ${i + 1} failed: ${error.message} (code: ${error.code})`);
      
      if (i < maxRetries - 1) {
        logToBackendFile(`Retrying in ${retryDelay}ms...`);
        await new Promise(resolve => setTimeout(resolve, retryDelay));
      } else {
        logToBackendFile(`All health check attempts failed. Final error: ${error.message}`);
        throw error;
      }
    }
  }
}

function stopBackendServer() {
  if (backendProcess) {
    logToBackendFile('Stopping backend server...');
    backendProcess.kill();
    backendProcess = null;
    logToBackendFile('Backend server stopped');
  }
}

function createWindow() {
  ensureScreenshotDirectory();

  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      enableRemoteModule: false,
      preload: path.join(__dirname, 'preload.js')
    },
    icon: path.join(__dirname, 'icon.png'),
    titleBarStyle: 'default',
    show: false
  });

  const startUrl = actuallyDev 
    ? 'http://localhost:3000' 
    : `file://${path.join(__dirname, '../build/index.html')}`;
  
  mainWindow.loadURL(startUrl);

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    safeLog.log('MainWindow is ready to show');
    
    // Ensure backend is running when window is shown
    if (!actuallyDev) {
      ensureBackendRunning().catch((error) => {
        safeLog.error('Failed to ensure backend is running:', error);
      });
    }
  });

  // Listen for window show events
  mainWindow.on('show', () => {
    mainWindow.webContents.send('window-show');
  });

  if (actuallyDev) {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });
}

app.whenReady().then(async () => {
  safeLog.log('Electron ready - creating window immediately and starting backend in parallel...');
  
  createWindow();
  startBackendInBackground();
  
  // Initialize native capture helper on macOS
  if (process.platform === 'darwin') {
    try {
      nativeCaptureHelper = new NativeCaptureHelper();
      await nativeCaptureHelper.initialize();
      safeLog.log('✅ Native capture helper initialized');
    } catch (error) {
      safeLog.warn(`⚠️ Native capture helper failed to initialize: ${error.message}`);
      safeLog.warn('Falling back to Electron desktopCapturer');
      nativeCaptureHelper = null; // Clear the helper so fallback logic works
    }
  }
  
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    } else {
      // Window exists but user activated the app, ensure backend is running
      if (!actuallyDev) {
        ensureBackendRunning().catch((error) => {
          safeLog.error('Failed to ensure backend is running on activate:', error);
        });
      }
      
      // Notify renderer about app activation
      const focusedWindow = BrowserWindow.getFocusedWindow();
      if (focusedWindow) {
        focusedWindow.webContents.send('app-activate');
      }
    }
  });
});

async function cleanupOldTmpImages(maxAge = 7 * 24 * 60 * 60 * 1000) {
  try {
    const imagesDir = ensureScreenshotDirectory();
    const files = fs.readdirSync(imagesDir);
    const now = Date.now();
    let deletedCount = 0;

    for (const file of files) {
      if (!file.startsWith('screenshot-') && 
          (file.endsWith('.png') || file.endsWith('.jpg') || file.endsWith('.jpeg') || 
           file.endsWith('.gif') || file.endsWith('.bmp') || file.endsWith('.webp'))) {
        const filepath = path.join(imagesDir, file);
        const stats = fs.statSync(filepath);
        const age = now - stats.mtime.getTime();
        
        if (age > maxAge) {
          fs.unlinkSync(filepath);
          deletedCount++;
        }
      }
    }

    return {
      success: true,
      deletedCount: deletedCount
    };
  } catch (error) {
    safeLog.error('Failed to cleanup tmp images:', error);
    return {
      success: false,
      error: error.message
    };
  }
}

async function startBackendInBackground() {
  safeLog.log('Starting backend server in background...');
  
  try {
    logToBackendFile('Initial backend startup...');
    await ensureBackendRunning();
    logToBackendFile('✅ Backend initialization complete');
    
    // Schedule cleanup of old tmp images after backend starts
    setTimeout(async () => {
      try {
        const result = await cleanupOldTmpImages();
        if (result.success && result.deletedCount > 0) {
          logToBackendFile(`Cleaned up ${result.deletedCount} old tmp images on startup`);
        }
      } catch (error) {
        logToBackendFile(`Failed to cleanup tmp images on startup: ${error.message}`);
      }
    }, 5000);
    
  } catch (error) {
    logToBackendFile(`❌ Backend initialization failed: ${error.message}`);
    logToBackendFile(`Error stack: ${error.stack}`);
    
    if (!actuallyDev) {
      let errorMessage = error.message || 'Unknown error';
      
      if (error.message && error.message.includes('ECONNREFUSED')) {
        errorMessage = 'Backend server failed to start - connection refused';
      } else if (error.message && error.message.includes('EADDRINUSE')) {
        errorMessage = 'Backend server failed to start - port already in use';
      } else if (error.message && error.message.includes('Backend process exited')) {
        errorMessage = 'Backend server crashed during startup';
      }
      
      const fullErrorMessage = `Failed to start the backend server: ${errorMessage}\n\nBackend log saved to: ${backendLogFile}`;
      
      dialog.showErrorBox(
        'Backend Startup Error', 
        fullErrorMessage
      );
    }
    
    safeLog.error(`Backend log saved to: ${backendLogFile}`);
  }
}

app.on('window-all-closed', () => {
  // On macOS, keep the backend running when window is closed
  // Only stop backend on other platforms where the app actually quits
  if (process.platform !== 'darwin') {
    stopBackendServer();
    app.quit();
  }
});

app.on('before-quit', () => {
  stopBackendServer();
});

app.on('web-contents-created', (event, contents) => {
  contents.on('new-window', (event, navigationUrl) => {
    event.preventDefault();
    shell.openExternal(navigationUrl);
  });
});

// IPC handlers for file operations
ipcMain.handle('select-files', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile', 'multiSelections'],
    filters: [
      { name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'] },
      { name: 'All Files', extensions: ['*'] }
    ]
  });
  
  return result.filePaths;
});

ipcMain.handle('select-save-path', async (event, options = {}) => {
  const result = await dialog.showSaveDialog(mainWindow, {
    title: options.title || 'Save File',
    defaultPath: options.defaultName || 'memories_export.xlsx',
    filters: [
      { name: 'Excel Files', extensions: ['xlsx'] },
      { name: 'CSV Files', extensions: ['csv'] },
      { name: 'All Files', extensions: ['*'] }
    ]
  });
  
  return {
    canceled: result.canceled,
    filePath: result.filePath
  };
});



// IPC handler for checking screen recording permissions
ipcMain.handle('check-screen-permission', async () => {
  try {
    if (process.platform === 'darwin') {
      const hasScreenPermission = systemPreferences.getMediaAccessStatus('screen');
      return {
        success: true,
        hasPermission: hasScreenPermission === 'granted',
        status: hasScreenPermission
      };
    } else {
      // On non-macOS platforms, assume permissions are granted
      return {
        success: true,
        hasPermission: true,
        status: 'granted'
      };
    }
  } catch (error) {
    safeLog.error('Failed to check screen permission:', error);
    return {
      success: false,
      hasPermission: false,
      error: error.message
    };
  }
});

// IPC handler for opening System Preferences to Screen Recording
ipcMain.handle('open-screen-recording-prefs', async () => {
  try {
    if (process.platform === 'darwin') {
      // Open System Preferences to Privacy & Security > Screen Recording
      const { spawn } = require('child_process');
      
      // Try the new System Settings first (macOS 13+)
      try {
        spawn('open', ['x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture']);
      } catch (error) {
        // Fall back to old System Preferences (macOS 12 and earlier)
        spawn('open', ['x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture']);
      }
      
      return {
        success: true,
        message: 'Opening System Preferences...'
      };
    } else {
      return {
        success: false,
        message: 'System Preferences not available on this platform'
      };
    }
  } catch (error) {
    safeLog.error('Failed to open System Preferences:', error);
    return {
      success: false,
      message: error.message
    };
  }
});

// IPC handler for getting available windows and screens for capture
ipcMain.handle('get-capture-sources', async () => {
  try {
    const { desktopCapturer, nativeImage } = require('electron');
    
    // Get all available sources from desktopCapturer
    const sources = await desktopCapturer.getSources({
      types: ['window', 'screen'],
      thumbnailSize: { width: 256, height: 144 },
      fetchWindowIcons: true
    });
    
    // Format sources for the frontend
    const formattedSources = sources.map(source => {
      let displayName = source.name;
      
      // Debug logging - now that we do proper app matching later, just log basic info
      if (source.display_id) {
        safeLog.log(`📺 Screen: "${source.name}"`);
      } else {
        safeLog.log(`🪟 Window: "${source.name}"`);
      }
      
      return {
        id: source.id,
        name: displayName,
        type: source.display_id ? 'screen' : 'window',
        thumbnail: source.thumbnail.toDataURL(),
        appIcon: source.appIcon ? source.appIcon.toDataURL() : null,
        isVisible: true // desktopCapturer only returns visible windows
      };
    });
    
    // On macOS, try to get additional windows including minimized ones
    if (process.platform === 'darwin') {
      try {
        // For cross-space detection, always use macWindowManager which has Python/Quartz support
        // Native helper only detects visible windows on current space
        let allWindows = [];
        
        safeLog.log('Using macWindowManager for cross-space window detection');
        const macWindowManager = require('./macWindowManager');
        allWindows = await macWindowManager.getAllWindows();
        
        // Create a map to track windows by app name for better deduplication
        const windowsByApp = new Map();
        
        // Create a map to match desktopCapturer windows with their real app names from macWindowManager
        const realAppNames = new Map();
        
        // First pass: try to match desktopCapturer windows with macWindowManager data to get real app names
        for (const macWindow of allWindows) {
          const macTitle = macWindow.windowTitle.toLowerCase();
          const macApp = macWindow.appName;
          
          // Try to find matching desktopCapturer window
          const matchingDesktopSource = formattedSources.find(source => {
            if (source.type === 'screen') return false;
            const sourceTitle = source.name.toLowerCase();
            
            // Exact match
            if (sourceTitle === macTitle) return true;
            
            // For Cursor: match "filename — project" with "filename — project" 
            if (macApp === 'Cursor' && sourceTitle.includes('—') && macTitle.includes('—')) {
              return sourceTitle === macTitle;
            }
            
            // For other apps: try partial matching
            if (sourceTitle.includes(macTitle) || macTitle.includes(sourceTitle)) {
              return true;
            }
            
            return false;
          });
          
          if (matchingDesktopSource) {
            realAppNames.set(matchingDesktopSource.name, macApp);
            safeLog.log(`🔗 Matched: "${matchingDesktopSource.name}" -> App: ${macApp}`);
          }
        }
        
        // Second pass: add all desktopCapturer windows to the map with correct app names
        formattedSources
          .filter(s => s.type === 'window')
          .forEach(source => {
            // Use real app name if available, otherwise fall back to parsing window title
            const realApp = realAppNames.get(source.name);
            let appName = realApp || source.name.split(' - ')[0];
            
            // Apply Cursor-specific formatting if we know it's actually Cursor
            let displayName = source.name;
            if (realApp === 'Cursor') {
              if (source.name.includes(' — ')) {
                const parts = source.name.split(' — ');
                if (parts.length >= 2) {
                  const lastPart = parts[parts.length - 1];
                  if (!lastPart.includes('.') && lastPart.length < 30) {
                    displayName = `Cursor - ${lastPart}`;
                  }
                }
              }
            }
            
            if (!windowsByApp.has(appName)) {
              windowsByApp.set(appName, []);
            }
            windowsByApp.get(appName).push({
              ...source,
              name: displayName, // Use the corrected display name
              appName: appName, // Store the real app name
              fromDesktopCapturer: true
            });
          });
        
        // Process windows from native API
        for (const window of allWindows) {
          const appName = window.appName;
          
          // Skip Electron's own windows
          if (appName === 'MIRIX' || appName === 'Electron') continue;
          
          // Check if we already have windows from this app
          const existingWindows = windowsByApp.get(appName) || [];
          
          // For important apps, always include minimized windows
          const importantApps = [
            'Zoom', 'zoom.us', 'Slack', 'Microsoft Teams', 'MSTeams', 'Teams', 'Discord', 'Skype',
            'Microsoft PowerPoint', 'PowerPoint', 'Keynote', 'Presentation',
            'Notion', 'Obsidian', 'Roam Research', 'Logseq',
            'Visual Studio Code', 'Code', 'Xcode', 'IntelliJ IDEA', 'PyCharm',
            'Google Chrome', 'Safari', 'Firefox', 'Microsoft Edge',
            'Figma', 'Sketch', 'Adobe Photoshop', 'Adobe Illustrator',
            'Finder', 'System Preferences', 'Activity Monitor'
          ];
          const isImportantApp = window.isImportantApp || importantApps.includes(appName);
          
          // Check if this specific window already exists
          const windowExists = existingWindows.some(existing => {
            const existingTitle = existing.name.toLowerCase();
            const currentTitle = `${appName} - ${window.windowTitle}`.toLowerCase();
            return existingTitle === currentTitle;
          });
          
          // Add the window if it doesn't exist or if it's an important app that might be minimized
          if (!windowExists || (isImportantApp && !window.isOnScreen)) {
            // Debug logging for Teams
            if (window.appName.includes('Teams')) {
              safeLog.log(`🔍 Teams window detection: ${window.appName} - ${window.windowTitle}, isOnScreen: ${window.isOnScreen}, windowExists: ${windowExists}, isImportantApp: ${isImportantApp}`);
            }
            
            // Check if this window was already found by desktopCapturer (meaning it's visible)
            const foundByDesktopCapturer = formattedSources.some(source => {
              const sourceName = source.name.toLowerCase();
              const windowName = window.appName.toLowerCase();
              return sourceName.includes(windowName) || sourceName.includes('teams');
            });
            
            // Create a virtual source for this window
            const virtualSource = {
              id: `virtual-window:${window.windowId || Date.now()}-${encodeURIComponent(window.appName)}`,
              name: `${window.appName} - ${window.windowTitle}`,
              type: 'window',
              thumbnail: null, // Will be captured when selected
              appIcon: null,
              isVisible: foundByDesktopCapturer || window.isOnScreen || false,
              isVirtual: true,
              appName: window.appName,
              windowTitle: window.windowTitle,
              windowId: window.windowId
            };
            
            // Try to get a real thumbnail using desktopCapturer
            try {
              const electronSources = await desktopCapturer.getSources({
                types: ['window'],
                thumbnailSize: { width: 512, height: 288 },
                fetchWindowIcons: true
              });
              
              // Try multiple matching strategies to find the window
              let matchingSource = null;
              
              // Strategy 1: Exact app name match
              matchingSource = electronSources.find(source => 
                source.name.toLowerCase().includes(window.appName.toLowerCase())
              );
              
              // Strategy 2: Partial match
              if (!matchingSource) {
                matchingSource = electronSources.find(source => 
                  window.appName.toLowerCase().includes(source.name.toLowerCase().split(' ')[0]) ||
                  source.name.toLowerCase().split(' ')[0].includes(window.appName.toLowerCase())
                );
              }
              
              // Strategy 3: For specific known apps, try common variations
              if (!matchingSource && window.appName.includes('zoom')) {
                matchingSource = electronSources.find(source => 
                  source.name.toLowerCase().includes('zoom')
                );
              }
              
              if (matchingSource && matchingSource.thumbnail) {
                virtualSource.thumbnail = matchingSource.thumbnail.toDataURL();
                virtualSource.appIcon = matchingSource.appIcon ? matchingSource.appIcon.toDataURL() : null;
                safeLog.log(`Successfully got thumbnail from desktopCapturer for ${window.appName}`);
              } else {
                safeLog.log(`No matching desktopCapturer source for ${window.appName}`);
              }
            } catch (captureError) {
              safeLog.log(`desktopCapturer failed for ${window.appName}: ${captureError.message}`);
            }
            
            // Create a meaningful placeholder thumbnail if we couldn't capture one
            if (!virtualSource.thumbnail) {
              // Choose color and icon based on app name
              let bgColor = '#4a4a4a';
              let appIcon = '📱';
              let appNameShort = window.appName.substring(0, 3).toUpperCase();
              
              if (window.appName.toLowerCase().includes('zoom')) {
                bgColor = '#2D8CFF';
                appIcon = '📹';
              } else if (window.appName.toLowerCase().includes('powerpoint')) {
                bgColor = '#D24726';
                appIcon = '📊';
              } else if (window.appName.toLowerCase().includes('notion')) {
                bgColor = '#000000';
                appIcon = '📝';
              } else if (window.appName.toLowerCase().includes('slack')) {
                bgColor = '#4A154B';
                appIcon = '💬';
              } else if (window.appName.toLowerCase().includes('teams')) {
                bgColor = '#6264A7';
                appIcon = '👥';
              } else if (window.appName.toLowerCase().includes('chrome')) {
                bgColor = '#4285F4';
                appIcon = '🌐';
              } else if (window.appName.toLowerCase().includes('word')) {
                bgColor = '#2B579A';
                appIcon = '📄';
              } else if (window.appName.toLowerCase().includes('excel')) {
                bgColor = '#217346';
                appIcon = '📊';
              } else if (window.appName.toLowerCase().includes('wechat')) {
                bgColor = '#07C160';
                appIcon = '💬';
              }
              
              // Create SVG placeholder
              const svg = `
                <svg width="256" height="144" xmlns="http://www.w3.org/2000/svg">
                  <rect width="256" height="144" fill="${bgColor}"/>
                  <text x="128" y="60" font-family="Arial, sans-serif" font-size="32" text-anchor="middle" fill="white">${appIcon}</text>
                  <text x="128" y="85" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="white">${window.appName}</text>
                  <text x="128" y="100" font-family="Arial, sans-serif" font-size="10" text-anchor="middle" fill="#cccccc">Hidden</text>
                </svg>
              `;
              
              virtualSource.thumbnail = `data:image/svg+xml;base64,${Buffer.from(svg).toString('base64')}`;
            }
            
            formattedSources.push(virtualSource);
          }
        }
      } catch (macError) {
        safeLog.error('Error getting additional windows from macOS:', macError);
        // Continue with just the desktopCapturer sources
      }
    }
    
    return {
      success: true,
      sources: formattedSources
    };
  } catch (error) {
    safeLog.error('Failed to get capture sources:', error);
    return {
      success: false,
      error: error.message,
      sources: []
    };
  }
});

// IPC handler for getting currently active/focused sources
ipcMain.handle('get-visible-sources', async (event, sourceIds) => {
  try {
    const { desktopCapturer } = require('electron');
    
    if (sourceIds && sourceIds.length > 0) {
      safeLog.log('Checking active/focused apps for source IDs:', sourceIds);
      
      // Enhanced multi-space/multi-screen visibility detection
      let activeAppsOnAllSpaces = [];
      if (process.platform === 'darwin') {
        try {
          const { exec } = require('child_process');
          const { promisify } = require('util');
          const execAsync = promisify(exec);
          
          // Get apps that have visible windows on ANY space (not just current)
          const { stdout: visibleAppsStdout } = await execAsync(`osascript -e '
            tell application "System Events"
              set visibleApps to {}
              repeat with p in (every application process)
                try
                  -- Check if app has any windows
                  if (count of windows of p) > 0 then
                    set end of visibleApps to (name of p as string)
                  end if
                end try
              end repeat
              return my list_to_string(visibleApps, ",")
            end tell
            
            on list_to_string(lst, delim)
              set AppleScript's text item delimiters to delim
              set str to lst as string
              set AppleScript's text item delimiters to ""
              return str
            end list_to_string
          '`);
          
          if (visibleAppsStdout && visibleAppsStdout.trim()) {
            activeAppsOnAllSpaces = visibleAppsStdout.trim().toLowerCase().split(',').map(app => app.trim());
            safeLog.log(`Apps with windows on all spaces: [${activeAppsOnAllSpaces.join(', ')}]`);
          }
          
          // Also get the frontmost app on current space for additional context
          const { stdout: frontmostStdout } = await execAsync(`osascript -e 'tell application "System Events" to get name of first application process whose frontmost is true'`);
          const frontmostApp = frontmostStdout.trim().toLowerCase();
          safeLog.log(`Frontmost app on current space: "${frontmostApp}"`);
          
        } catch (error) {
          safeLog.log('Could not get apps with windows:', error.message);
          // Fallback to assume all apps are visible
          activeAppsOnAllSpaces = [];
        }
      }
      
      // Also get visible sources for fallback
      const visibleSources = await desktopCapturer.getSources({
        types: ['window', 'screen'],
        thumbnailSize: { width: 1, height: 1 },
        fetchWindowIcons: false
      });
      
      const results = sourceIds.map(id => {
        let isVisible = false;
        let name = 'Unknown';
        
        if (id.startsWith('virtual-window:')) {
          const appNameMatch = id.match(/virtual-window:\d+-(.+)$/);
          if (appNameMatch) {
            name = decodeURIComponent(appNameMatch[1]);
            
            // Enhanced visibility check: app is visible if it has windows on ANY space
            if (activeAppsOnAllSpaces.length > 0) {
              const appNameLower = name.toLowerCase();
              const hasWindowsOnAnySpace = activeAppsOnAllSpaces.some(activeApp => {
                return activeApp.includes(appNameLower) || 
                       appNameLower.includes(activeApp) ||
                       (appNameLower === 'msteams' && activeApp.includes('teams')) ||
                       (appNameLower === 'microsoft teams' && activeApp.includes('teams')) ||
                       (appNameLower === 'wechat' && (activeApp.includes('wechat') || activeApp.includes('weixin'))) ||
                       (appNameLower === 'google chrome' && activeApp.includes('chrome')) ||
                       (appNameLower === 'visual studio code' && (activeApp.includes('code') || activeApp.includes('visual studio'))) ||
                       (appNameLower === 'microsoft powerpoint' && (activeApp.includes('powerpoint') || activeApp.includes('microsoft powerpoint'))) ||
                       (appNameLower === 'microsoft word' && (activeApp.includes('word') || activeApp.includes('microsoft word'))) ||
                       (appNameLower === 'microsoft excel' && (activeApp.includes('excel') || activeApp.includes('microsoft excel')));
              });
              
              if (hasWindowsOnAnySpace) {
                isVisible = true;
                safeLog.log(`Virtual window has windows on some space: ${id} -> ${name}`);
              } else {
                safeLog.log(`Virtual window has no windows on any space: ${id} -> ${name}`);
              }
            } else {
              // Fallback: if we can't detect apps with windows, assume visible
              isVisible = true;
              safeLog.log(`Virtual window assumed visible (no space detection): ${id} -> ${name}`);
            }
          }
        } else {
          // For regular window IDs, check if they're actually visible
          const visibleSource = visibleSources.find(s => s.id === id);
          if (visibleSource) {
            isVisible = true;
            name = visibleSource.name;
            safeLog.log(`Regular window found visible: ${id} -> ${name}`);
          } else {
            safeLog.log(`Regular window NOT visible: ${id}`);
          }
        }
        
        return { id, isVisible, name };
      });
      
      return { success: true, sources: results };
    } else {
      // Return all available sources
      const visibleSources = await desktopCapturer.getSources({
        types: ['window', 'screen'],
        thumbnailSize: { width: 1, height: 1 },
        fetchWindowIcons: false
      });
      
      const allVisible = visibleSources.map(s => ({ 
        id: s.id, 
        name: s.name,
        isVisible: true 
      }));
      
      return { success: true, sources: allVisible };
    }
  } catch (error) {
    safeLog.error('Error checking source visibility:', error);
    return { success: false, error: error.message };
  }
});

// IPC handler for taking screenshot of specific source (window or screen)
ipcMain.handle('take-source-screenshot', async (event, sourceId) => {
  try {
    // Source screenshot logging disabled
    
    const imagesDir = ensureScreenshotDirectory();
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `screenshot-${sourceId}-${timestamp}.png`;
    const filepath = path.join(imagesDir, filename);
    
    // Check permissions on macOS
    if (process.platform === 'darwin') {
      const hasScreenPermission = systemPreferences.getMediaAccessStatus('screen');
      if (hasScreenPermission !== 'granted') {
        const permissionGranted = await systemPreferences.askForMediaAccess('screen');
        if (!permissionGranted) {
          throw new Error('Screen recording permission not granted. Please grant screen recording permissions in System Preferences > Security & Privacy > Screen Recording and restart the application.');
        }
      }
    }
    
    const { desktopCapturer } = require('electron');
    
    // Handle virtual windows (minimized or on other spaces)
    if (sourceId.startsWith('virtual-window:')) {
      // Extract app name from the source ID
      const appNameMatch = sourceId.match(/virtual-window:\d+-(.+)$/);
      const appName = appNameMatch ? decodeURIComponent(appNameMatch[1]) : null;
      
      // Declare matchingSource in the correct scope
      let matchingSource = null;
      
      // First, quickly check if the app might be visible on current desktop
      const quickSources = await desktopCapturer.getSources({
        types: ['window'],
        thumbnailSize: { width: 256, height: 144 }, // Small size for quick check
        fetchWindowIcons: false
      });
      
      // Quick check if app is likely on current desktop
      const quickMatch = quickSources.find(source => {
        const name = source.name.toLowerCase();
        const appLower = appName.toLowerCase();
        return name.includes(appLower) || 
               (appLower.includes('powerpoint') && (name.includes('powerpoint') || name.includes('ppt'))) ||
               (appLower.includes('wechat') && name.includes('weixin')) ||
               (appLower.includes('chrome') && name.includes('chrome'));
      });
      
      if (quickMatch) {
        // Disabled to reduce log spam during frequent captures
        // safeLog.log(`✅ ${appName} found on current desktop, getting high-quality thumbnail`);
        
        // Get high-quality capture since we know it's visible
        try {
          const sources = await desktopCapturer.getSources({
            types: ['window'],
            thumbnailSize: { width: 1920, height: 1080 },
            fetchWindowIcons: true
          });
          
          // Find the matching source again with better quality
          matchingSource = sources.find(s => s.id === quickMatch.id);
          
          if (matchingSource) {
            // Disabled to reduce log spam during frequent captures
            // safeLog.log(`✅ Got high-quality thumbnail for ${appName}`);
            
            const image = matchingSource.thumbnail;
            const buffer = image.toPNG();
            
            fs.writeFileSync(filepath, buffer);
            // saveDebugCopy(filepath, 'electron_selected_source', matchingSource.name);
            
            const stats = fs.statSync(filepath);
            
            return {
              success: true,
              filepath: filepath,
              filename: filename,
              size: stats.size,
              sourceName: matchingSource.name
            };
          }
        } catch (highQualityError) {
          safeLog.log(`Failed to get high-quality capture: ${highQualityError.message}`);
        }
      } else {
        // App not on current desktop
      }
      
      // Check variable state
      
      // Try Python-free native capture helper for screen capture
      if (nativeCaptureHelper && nativeCaptureHelper.isRunning && appName) {
        safeLog.log(`Attempting Python-free screen capture for ${appName}`);
        try {
          const captureResult = await nativeCaptureHelper.captureScreen(0);
          if (captureResult.success && captureResult.data) {
            // Pure JS screen capture successful
            
            fs.writeFileSync(filepath, captureResult.data);
            // saveDebugCopy(filepath, 'python_free_screen_capture', appName);
            
            const stats = fs.statSync(filepath);
            
            return {
              success: true,
              filepath: filepath,
              filename: filename,
              size: stats.size,
              sourceName: `${appName} (Screen Capture - Python Free)`,
              captureMethod: 'python_free_screen'
            };
          } else {
            safeLog.log(`❌ Python-free screen capture failed for ${appName}: ${captureResult.error}`);
          }
        } catch (nativeError) {
          safeLog.log(`❌ Python-free capture error for ${appName}: ${nativeError.message}`);
        }
      }
      
      // Fallback: Try advanced macOS capture methods for cross-desktop window capture
      if (appName && process.platform === 'darwin') {
        // Attempting cross-desktop capture
        
        try {
          // Do NOT activate the app - we want to capture silently in the background
          
          // Try enhanced Python-based cross-desktop capture
            const macWindowManager = require('./macWindowManager');
            const allWindows = await macWindowManager.getAllWindows();
            const targetWindow = allWindows.find(w => 
              w.appName.toLowerCase() === appName.toLowerCase() ||
              w.appName.toLowerCase().includes(appName.toLowerCase()) ||
              appName.toLowerCase().includes(w.appName.toLowerCase())
            );
            
            if (targetWindow && targetWindow.windowId) {
              // Found window ID
              
              try {
                const captureResult = await new Promise(async (resolve, reject) => {
              const pythonScript = `
import sys
try:
    from Quartz import CGWindowListCreateImage, CGRectNull, kCGWindowListOptionIncludingWindow, kCGWindowImageBoundsIgnoreFraming, kCGWindowImageShouldBeOpaque, CGWindowListCopyWindowInfo, kCGWindowListOptionAll, kCGNullWindowID
    from CoreFoundation import kCFNull
    import base64
    
    app_name = "${appName}"
    old_window_id = ${targetWindow.windowId}
    
    # Get fresh window list and find the app by name
    window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionAll, kCGNullWindowID)
    target_window = None
    window_id = None
    
    # Look for the app by name in current window list and find the LARGEST window
    candidate_windows = []
    all_matching_windows = []  # Track ALL windows for this app for debugging
    all_windows_debug = []  # Track ALL windows for debugging
    
    for window in window_list:
        owner_name = window.get('kCGWindowOwnerName', '').lower()
        window_name = window.get('kCGWindowName', '').lower()
        bounds = window.get('kCGWindowBounds', {})
        width = bounds.get('Width', 0)
        height = bounds.get('Height', 0)
        
        # Track all windows for debugging
        all_windows_debug.append((window.get('kCGWindowNumber'), width, height, owner_name, window_name))
        
        # More flexible matching for MSTeams and similar apps
        app_keywords = [app_name.lower()]
        if app_name.lower() == 'msteams' or app_name.lower() == 'microsoft teams':
            app_keywords.extend(['microsoft teams', 'teams', 'com.microsoft.teams', 'msteams', 'com.microsoft.teams2'])
        elif app_name.lower() == 'notion':
            app_keywords.extend(['notion', 'com.notion.notion'])
        elif app_name.lower() == 'microsoft powerpoint':
            app_keywords.extend(['powerpoint', 'com.microsoft.powerpoint'])
            
        matches = False
        for keyword in app_keywords:
            if (keyword in owner_name or keyword in window_name):
                matches = True
                break
        
        if matches:
            all_matching_windows.append((window.get('kCGWindowNumber'), width, height, owner_name, window_name))
            
            # Skip windows with very small bounds (likely not main windows)
            if width > 200 and height > 200:  # Increased minimum size
                candidate_windows.append((window, width * height))  # Store window and area
                print(f"DEBUG: Found candidate window {window.get('kCGWindowNumber')} for {app_name}: {width}x{height}", file=sys.stderr)
    
    # Debug: Show ALL matching windows regardless of size
    print(f"DEBUG: All windows found for {app_name}:", file=sys.stderr)
    for wid, w, h, owner, title in all_matching_windows:
        print(f"  Window {wid}: {w}x{h} owner='{owner}' title='{title}'", file=sys.stderr)
        
    # If no matches found, show some examples of available windows
    if not all_matching_windows:
        print(f"DEBUG: No matches for '{app_name}'. Sample of available windows:", file=sys.stderr)
        for wid, w, h, owner, title in all_windows_debug[:10]:  # Show first 10
            if w > 50 and h > 50:  # Only show reasonable sized windows
                print(f"  Available: {wid}: {w}x{h} owner='{owner}' title='{title}'", file=sys.stderr)
    
    # Sort by area (largest first) but prefer non-webview windows
    if candidate_windows:
        # Sort with custom logic: prefer non-webview windows, then by area
        def window_priority(item):
            window, area = item
            owner = window.get('kCGWindowOwnerName', '').lower()
            # Penalize webview windows
            is_webview = 'webview' in owner
            # Return tuple: (webview penalty, negative area for descending sort)
            return (is_webview, -area)
        
        candidate_windows.sort(key=window_priority)
        target_window = candidate_windows[0][0]
        window_id = target_window.get('kCGWindowNumber')
        bounds = target_window.get('kCGWindowBounds', {})
        owner_name = target_window.get('kCGWindowOwnerName', '')
        print(f"DEBUG: Selected window ID {window_id} for {app_name} (was {old_window_id}): {bounds.get('Width', 0)}x{bounds.get('Height', 0)}, owner='{owner_name}'", file=sys.stderr)
    
    if not target_window:
        # If no large windows found, pick the largest available window regardless of size
        print(f"DEBUG: No large windows found, selecting largest available window", file=sys.stderr)
        if all_matching_windows:
            # Sort by area but prefer non-webview windows
            def fallback_priority(window_info):
                wid, w, h, owner, title = window_info
                area = w * h
                is_webview = 'webview' in owner.lower()
                # Return tuple: (webview penalty, negative area for descending sort)
                return (is_webview, -area)
            
            all_matching_windows.sort(key=fallback_priority)
            wid, w, h, owner, title = all_matching_windows[0]
            
            # Find the actual window object
            for window in window_list:
                if window.get('kCGWindowNumber') == wid:
                    target_window = window
                    window_id = wid
                    print(f"DEBUG: Selected window ID {window_id} for {app_name}: {w}x{h}, owner='{owner}'", file=sys.stderr)
                    break
    
    if not target_window:
        print(f"ERROR: No suitable window found for {app_name} in current window list")
        sys.exit(1)
    
    # Check window properties that might affect capture
    window_layer = target_window.get('kCGWindowLayer', 'unknown')
    window_alpha = target_window.get('kCGWindowAlpha', 'unknown')
    window_bounds = target_window.get('kCGWindowBounds', {})
    
    print(f"DEBUG: Window layer: {window_layer}, alpha: {window_alpha}, bounds: {window_bounds}", file=sys.stderr)
    
    # Try different capture options
    capture_options = [
        kCGWindowImageBoundsIgnoreFraming | kCGWindowImageShouldBeOpaque,
        kCGWindowImageBoundsIgnoreFraming,
        kCGWindowImageShouldBeOpaque,
        0  # No special options
    ]
    
    image = None
    for i, options in enumerate(capture_options):
        print(f"DEBUG: Trying capture option {i+1}/4", file=sys.stderr)
        image = CGWindowListCreateImage(
            CGRectNull,
            kCGWindowListOptionIncludingWindow,
            window_id,
            options
        )
        if image:
            print(f"DEBUG: Capture succeeded with option {i+1}", file=sys.stderr)
            break
    
    if image:
        # Convert to PNG data
        from Quartz import CGImageDestinationCreateWithData, CGImageDestinationAddImage, CGImageDestinationFinalize
        from CoreFoundation import CFDataCreateMutable, kCFAllocatorDefault
        
        data = CFDataCreateMutable(kCFAllocatorDefault, 0)
        dest = CGImageDestinationCreateWithData(data, 'public.png', 1, None)
        CGImageDestinationAddImage(dest, image, None)
        CGImageDestinationFinalize(dest)
        
        # Convert to base64 and print
        import base64
        png_data = bytes(data)
        print(base64.b64encode(png_data).decode('utf-8'))
    else:
        # If direct window capture fails, try screen capture with cropping
        print("DEBUG: Direct window capture failed, trying screen capture with cropping", file=sys.stderr)
        try:
            from Quartz import CGDisplayCreateImage, CGMainDisplayID, CGImageCreateWithImageInRect
            from CoreGraphics import CGRectMake
            
            # Get the window bounds
            bounds = target_window.get('kCGWindowBounds', {})
            x = bounds.get('X', 0)
            y = bounds.get('Y', 0) 
            width = bounds.get('Width', 0)
            height = bounds.get('Height', 0)
            
            if width > 0 and height > 0:
                # Capture entire screen
                screen_image = CGDisplayCreateImage(CGMainDisplayID())
                if screen_image:
                    # Crop to window bounds
                    crop_rect = CGRectMake(x, y, width, height)
                    cropped_image = CGImageCreateWithImageInRect(screen_image, crop_rect)
                    
                    if cropped_image:
                        # Convert to PNG
                        data = CFDataCreateMutable(kCFAllocatorDefault, 0)
                        dest = CGImageDestinationCreateWithData(data, 'public.png', 1, None)
                        CGImageDestinationAddImage(dest, cropped_image, None)
                        CGImageDestinationFinalize(dest)
                        
                        png_data = bytes(data)
                        print(base64.b64encode(png_data).decode('utf-8'))
                        print("DEBUG: Screen capture + crop succeeded", file=sys.stderr)
                    else:
                        print("ERROR: Failed to crop screen image")
                else:
                    print("ERROR: Failed to capture screen")
            else:
                print("ERROR: Invalid window bounds for cropping")
        except Exception as crop_error:
            print(f"ERROR: Screen capture fallback failed: {crop_error}")
            print("ERROR: Failed to create image with all capture options")
        
except Exception as e:
    print(f"ERROR: {e}")
`;
              
              const { spawn } = require('child_process');
              
              // Function to find available Python executable
              const findPython = async () => {
                const possiblePaths = [
                  'python3',  // Use system PATH first
                  'python',   // Fallback to python
                  '/usr/bin/python3',
                  '/usr/local/bin/python3',
                  '/opt/homebrew/bin/python3',
                  `${process.env.HOME}/anaconda3/bin/python3`,
                  `${process.env.HOME}/miniconda3/bin/python3`,
                  '/System/Library/Frameworks/Python.framework/Versions/Current/bin/python3'
                ];
                
                // Test each path to find one that works
                for (const pythonPath of possiblePaths) {
                  try {
                    const testResult = await new Promise((resolve) => {
                      const testProcess = spawn(pythonPath, ['-c', 'import sys; print("OK")'], {
                        stdio: ['pipe', 'pipe', 'pipe']
                      });
                      
                      let output = '';
                      testProcess.stdout.on('data', (data) => {
                        output += data.toString();
                      });
                      
                      testProcess.on('close', (code) => {
                        resolve({ success: code === 0 && output.trim() === 'OK', pythonPath });
                      });
                      
                      testProcess.on('error', () => {
                        resolve({ success: false, pythonPath });
                      });
                      
                      // Timeout after 3 seconds
                      setTimeout(() => {
                        testProcess.kill();
                        resolve({ success: false, pythonPath });
                      }, 3000);
                    });
                    
                    if (testResult.success) {
                      console.log(`[ScreenCapture] Found working Python: ${pythonPath}`);
                      return pythonPath;
                    }
                  } catch (error) {
                    // Continue to next path
                  }
                }
                
                console.error('[ScreenCapture] No working Python found, falling back to python3');
                return 'python3'; // Final fallback
              };
              
              const pythonCmd = await findPython();
              const python = spawn(pythonCmd, ['-c', pythonScript]);
              
              let output = '';
              let error = '';
              
              python.stdout.on('data', (data) => {
                output += data.toString();
              });
              
              python.stderr.on('data', (data) => {
                error += data.toString();
              });
              
              python.on('close', (code) => {
                if (code === 0 && output.trim() && !output.startsWith('ERROR:')) {
                  try {
                    const base64Data = output.trim();
                    const imageBuffer = Buffer.from(base64Data, 'base64');
                    resolve(imageBuffer);
                  } catch (parseError) {
                    reject(new Error(`Failed to parse image data: ${parseError.message}`));
                  }
                } else {
                  reject(new Error(`Python capture failed: ${error || output}`));
                }
              });
              
              python.on('error', reject);
                });
                
                if (captureResult && captureResult.length > 1000) {
                  fs.writeFileSync(filepath, captureResult);
                  const stats = fs.statSync(filepath);
                  
                  // CGWindowListCreateImage capture successful
                  // saveDebugCopy(filepath, 'cg_window_capture', targetWindow ? targetWindow.name : appName);
                  
                  return {
                    success: true,
                    filepath: filepath,
                    filename: filename,
                    size: stats.size,
                    sourceName: appName,
                    isCGWindowCapture: true
                  };
                }
              } catch (cgWindowError) {
                safeLog.log(`❌ Python fallback capture failed for ${appName}: ${cgWindowError.message}`);
              }
            }
        } catch (outerError) {
          safeLog.log(`❌ Cross-desktop capture failed for ${appName}: ${outerError.message}`);
        }
      }
      
      // Fallback: Create a more informative placeholder image for failed capture
      safeLog.log(`Creating placeholder for virtual window: ${appName}`);
      
      // Create a better placeholder that indicates the app is hidden
      const placeholderSvg = `
        <svg width="512" height="288" xmlns="http://www.w3.org/2000/svg">
          <rect width="512" height="288" fill="#2a2a2a"/>
          <text x="256" y="120" font-family="Arial, sans-serif" font-size="48" text-anchor="middle" fill="#888">📱</text>
          <text x="256" y="170" font-family="Arial, sans-serif" font-size="20" text-anchor="middle" fill="#ccc">${appName || 'App'}</text>
          <text x="256" y="200" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="#888">Window not visible</text>
          <text x="256" y="220" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#666">May be minimized or on another desktop</text>
        </svg>
      `;
      
      // Convert SVG to PNG using a minimal PNG fallback for now
      const minimalPng = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==', 'base64');
      fs.writeFileSync(filepath, minimalPng);
      
      const stats = fs.statSync(filepath);
      
      return {
        success: true,
        filepath: filepath,
        filename: filename,
        size: stats.size,
        sourceName: appName || 'Virtual Window',
        isPlaceholder: true,
        placeholderReason: 'Window not accessible - may be minimized or on another desktop'
      };
    }
    
    // For regular window capture, use desktopCapturer thumbnail directly
    else if (sourceId.startsWith('window:')) {
      const sources = await desktopCapturer.getSources({
        types: ['window'],
        thumbnailSize: { width: 1920, height: 1080 },
        fetchWindowIcons: true
      });
      
      const source = sources.find(s => s.id === sourceId);
      if (!source) {
        throw new Error(`Window with ID ${sourceId} not found`);
      }
      
      // Get the thumbnail image and convert to PNG buffer
      const image = source.thumbnail;
      const buffer = image.toPNG();
      
      // Write to file
      fs.writeFileSync(filepath, buffer);
      
      // Save debug copy
      // saveDebugCopy(filepath, 'electron_window', source.name);
      
      const stats = fs.statSync(filepath);
      
      return {
        success: true,
        filepath: filepath,
        filename: filename,
        size: stats.size,
        sourceName: source.name
      };
    } else {
      // For screens, use the regular approach
      const sources = await desktopCapturer.getSources({
        types: ['screen'],
        thumbnailSize: { width: 1920, height: 1080 }
      });
      
      const source = sources.find(s => s.id === sourceId);
      if (!source) {
        throw new Error(`Screen with ID ${sourceId} not found`);
      }
      
      // Get the full-size image from the source
      const image = source.thumbnail;
      const buffer = image.toPNG();
      
      // Write to file
      fs.writeFileSync(filepath, buffer);
      
      // Save debug copy
      // saveDebugCopy(filepath, 'electron_screen', `Display ${source.display_id}`);
      
      const stats = fs.statSync(filepath);
      
      return {
        success: true,
        filepath: filepath,
        filename: filename,
        size: stats.size,
        sourceName: source.name
      };
    }
  } catch (error) {
    // Silent error handling for missing windows
    return {
      success: false,
      error: error.message
    };
  }
});

// IPC handler for taking screenshot (full screen - backward compatibility)
ipcMain.handle('take-screenshot', async () => {
  try {
    const imagesDir = ensureScreenshotDirectory();
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `screenshot-${timestamp}.png`;
    const filepath = path.join(imagesDir, filename);
    
    // Check if we're on macOS and ask for screen recording permissions
    if (process.platform === 'darwin') {
      // Check if we have screen recording permissions
      const hasScreenPermission = systemPreferences.getMediaAccessStatus('screen');
      
      if (hasScreenPermission !== 'granted') {
        // Request screen recording permissions
        const permissionGranted = await systemPreferences.askForMediaAccess('screen');
        
        if (!permissionGranted) {
          throw new Error('Screen recording permission not granted. Please grant screen recording permissions in System Preferences > Security & Privacy > Screen Recording and restart the application.');
        }
      }
    }
    
    // Try to take screenshot with better error handling
    try {
      const imgBuffer = await screenshot();
      
      // Write the buffer to file
      fs.writeFileSync(filepath, imgBuffer);
      
      // Save debug copy
      // saveDebugCopy(filepath, 'fullscreen', 'primary_display');
      
    } catch (screenshotError) {
      safeLog.error('Screenshot capture failed:', screenshotError);
      
      // Try alternative method if the first one fails
      try {
        safeLog.log('Trying alternative screenshot method...');
        await screenshot(filepath);
      } catch (altError) {
        safeLog.error('Alternative screenshot method also failed:', altError);
        throw new Error(`Screenshot capture failed: ${screenshotError.message}. Alternative method error: ${altError.message}`);
      }
    }
    
    // Verify the file was created
    if (!fs.existsSync(filepath)) {
      throw new Error(`Screenshot file was not created: ${filepath}`);
    }
    
    const stats = fs.statSync(filepath);
    
    return {
      success: true,
      filepath: filepath,
      filename: filename,
      size: stats.size
    };
  } catch (error) {
    safeLog.error('Failed to take screenshot:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

// IPC handler for taking screenshot of specific display
ipcMain.handle('take-screenshot-display', async (event, displayId = 0) => {
  try {
    const imagesDir = ensureScreenshotDirectory();
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `screenshot-display-${displayId}-${timestamp}.png`;
    const filepath = path.join(imagesDir, filename);

    // Get list of displays and take screenshot of specific display
    const displays = await screenshot.listDisplays();
    if (displayId >= displays.length) {
      throw new Error(`Display ${displayId} not found. Available displays: ${displays.length}`);
    }

    const imgBuffer = await screenshot({ screen: displays[displayId].id });
    
    // Save screenshot
    fs.writeFileSync(filepath, imgBuffer);
    
    // Save debug copy
    // saveDebugCopy(filepath, 'display_capture', `display_${displayId}`);
    
    safeLog.log(`Screenshot of display ${displayId} saved: ${filepath}`);
    
    return {
      success: true,
      filepath: filepath,
      filename: filename,
      size: imgBuffer.length,
      displayId: displayId
    };
  } catch (error) {
    safeLog.error('Failed to take screenshot of display:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

// IPC handler for saving debug comparison images - DISABLED
// ipcMain.handle('save-debug-comparison-image', async (event, imageBuffer, filename) => {
//   try {
//     const compareDir = ensureDebugCompareDirectory();
//     const filepath = path.join(compareDir, filename);
//     
//     fs.writeFileSync(filepath, Buffer.from(imageBuffer));
//     console.log(`💾 Saved comparison image: ${filepath}`);
//     
//     return {
//       success: true,
//       filepath: filepath
//     };
//   } catch (error) {
//     console.error('Failed to save comparison image:', error);
//     return {
//       success: false,
//       error: error.message
//     };
//   }
// });

// IPC handler for getting available displays
ipcMain.handle('list-displays', async () => {
  try {
    const displays = await screenshot.listDisplays();
    return {
      success: true,
      displays: displays.map((display, index) => ({
        id: display.id,
        index: index,
        name: display.name || `Display ${index + 1}`,
        bounds: display.bounds
      }))
    };
  } catch (error) {
    safeLog.error('Failed to list displays:', error);
    return {
      success: false,
      error: error.message,
      displays: []
    };
  }
});

// IPC handler for cleaning up old screenshots
ipcMain.handle('cleanup-screenshots', async (event, maxAge = 24 * 60 * 60 * 1000) => {
  try {
    const imagesDir = ensureScreenshotDirectory();
    const files = fs.readdirSync(imagesDir);
    const now = Date.now();
    let deletedCount = 0;

    for (const file of files) {
      if (file.startsWith('screenshot-') && file.endsWith('.png')) {
        const filepath = path.join(imagesDir, file);
        const stats = fs.statSync(filepath);
        const age = now - stats.mtime.getTime();
        
        if (age > maxAge) {
          fs.unlinkSync(filepath);
          deletedCount++;
        }
      }
    }

    return {
      success: true,
      deletedCount: deletedCount
    };
  } catch (error) {
    safeLog.error('Failed to cleanup screenshots:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

// IPC handler for reading image as base64 (for similarity comparison)
ipcMain.handle('read-image-base64', async (event, filepath) => {
  try {
    
    if (!fs.existsSync(filepath)) {
      throw new Error(`File does not exist: ${filepath}`);
    }

    const stats = fs.statSync(filepath);
    
    const imageBuffer = fs.readFileSync(filepath);
    const base64Data = imageBuffer.toString('base64');
    const mimeType = 'image/png'; // Assuming PNG format for screenshots
    const dataUrl = `data:${mimeType};base64,${base64Data}`;

    return {
      success: true,
      dataUrl: dataUrl,
      base64: base64Data,
      size: imageBuffer.length
    };
  } catch (error) {
    safeLog.error('Failed to read image as base64:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

// IPC handler for deleting screenshot files (used when screenshots are too similar)
ipcMain.handle('delete-screenshot', async (event, filepath) => {
  try {
    if (!fs.existsSync(filepath)) {
      // Don't log for non-existent files - this is normal for placeholders
      return {
        success: true,
        message: 'File does not exist'
      };
    }
    
    // Check if it's a tiny placeholder image - don't bother deleting these
    const stats = fs.statSync(filepath);
    if (stats.size < 200) { // Placeholder images are very small
      safeLog.log(`Skipping deletion of placeholder image: ${filepath} (${stats.size} bytes)`);
      return {
        success: true,
        message: 'Placeholder image, skipping deletion'
      };
    }
    
    safeLog.log(`Attempting to delete screenshot: ${filepath} (${stats.size} bytes)`);

    // Only allow deletion of files in the screenshots directory for security
    const imagesDir = ensureScreenshotDirectory();
    const normalizedFilepath = path.resolve(filepath);
    const normalizedImagesDir = path.resolve(imagesDir);
    
    if (!normalizedFilepath.startsWith(normalizedImagesDir)) {
      throw new Error('Can only delete files in the screenshots directory');
    }

    fs.unlinkSync(filepath);
    safeLog.log(`Screenshot deleted successfully: ${filepath}`);

    return {
      success: true,
      message: 'File deleted successfully'
    };
  } catch (error) {
    safeLog.error('Failed to delete screenshot:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

// IPC handler for saving image files to tmp directory
ipcMain.handle('save-image-to-tmp', async (event, sourcePath, filename) => {
  try {
    const imagesDir = ensureScreenshotDirectory();
    const targetPath = path.join(imagesDir, filename);

    // Check if source file exists
    if (!fs.existsSync(sourcePath)) {
      throw new Error(`Source file does not exist: ${sourcePath}`);
    }

    // Copy the file to the tmp directory
    fs.copyFileSync(sourcePath, targetPath);
    
    safeLog.log(`Image saved to tmp directory: ${targetPath}`);

    return targetPath;
  } catch (error) {
    safeLog.error('Failed to save image to tmp directory:', error);
    throw error;
  }
});

// IPC handler for saving image buffer to tmp directory
ipcMain.handle('save-image-buffer-to-tmp', async (event, arrayBuffer, filename) => {
  try {
    const imagesDir = ensureScreenshotDirectory();
    const targetPath = path.join(imagesDir, filename);

    // Convert ArrayBuffer to Buffer
    const buffer = Buffer.from(arrayBuffer);
    
    // Write the buffer to file
    fs.writeFileSync(targetPath, buffer);
    
    safeLog.log(`Image buffer saved to tmp directory: ${targetPath}`);

    return targetPath;
  } catch (error) {
    safeLog.error('Failed to save image buffer to tmp directory:', error);
    throw error;
  }
});

// IPC handler for cleaning up old tmp images
ipcMain.handle('cleanup-tmp-images', async (event, maxAge = 7 * 24 * 60 * 60 * 1000) => {
  try {
    const imagesDir = ensureScreenshotDirectory();
    const files = fs.readdirSync(imagesDir);
    const now = Date.now();
    let deletedCount = 0;

    for (const file of files) {
      // Clean up any image files older than maxAge, but skip screenshot files
      if (!file.startsWith('screenshot-') && 
          (file.endsWith('.png') || file.endsWith('.jpg') || file.endsWith('.jpeg') || 
           file.endsWith('.gif') || file.endsWith('.bmp') || file.endsWith('.webp'))) {
        const filepath = path.join(imagesDir, file);
        const stats = fs.statSync(filepath);
        const age = now - stats.mtime.getTime();
        
        if (age > maxAge) {
          fs.unlinkSync(filepath);
          deletedCount++;
        }
      }
    }

    safeLog.log(`Cleaned up ${deletedCount} old tmp images`);

    return {
      success: true,
      deletedCount: deletedCount
    };
  } catch (error) {
    safeLog.error('Failed to cleanup tmp images:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

// Handle app protocol for deep linking (optional)
if (process.defaultApp) {
  if (process.argv.length >= 2) {
    app.setAsDefaultProtocolClient('mirix', process.execPath, [path.resolve(process.argv[1])]);
  }
} else {
  app.setAsDefaultProtocolClient('mirix');
}

const createMenu = () => {
  const template = [
    {
      label: 'File',
      submenu: [
        {
          label: 'Quit',
          accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Ctrl+Q',
          click: () => {
            app.quit();
          }
        }
      ]
    },
    {
      label: 'Edit',
      submenu: [
        { label: 'Undo', accelerator: 'CmdOrCtrl+Z', role: 'undo' },
        { label: 'Redo', accelerator: 'Shift+CmdOrCtrl+Z', role: 'redo' },
        { type: 'separator' },
        { label: 'Cut', accelerator: 'CmdOrCtrl+X', role: 'cut' },
        { label: 'Copy', accelerator: 'CmdOrCtrl+C', role: 'copy' },
        { label: 'Paste', accelerator: 'CmdOrCtrl+V', role: 'paste' }
      ]
    },
    {
      label: 'View',
      submenu: [
        { label: 'Reload', accelerator: 'CmdOrCtrl+R', role: 'reload' },
        { label: 'Force Reload', accelerator: 'CmdOrCtrl+Shift+R', role: 'forceReload' },
        { label: 'Toggle Developer Tools', accelerator: process.platform === 'darwin' ? 'Alt+Cmd+I' : 'Ctrl+Shift+I', role: 'toggleDevTools' },
        { type: 'separator' },
        { label: 'Actual Size', accelerator: 'CmdOrCtrl+0', role: 'resetZoom' },
        { label: 'Zoom In', accelerator: 'CmdOrCtrl+Plus', role: 'zoomIn' },
        { label: 'Zoom Out', accelerator: 'CmdOrCtrl+-', role: 'zoomOut' },
        { type: 'separator' },
        { label: 'Toggle Fullscreen', accelerator: process.platform === 'darwin' ? 'Ctrl+Cmd+F' : 'F11', role: 'togglefullscreen' }
      ]
    },
    {
      label: 'Window',
      submenu: [
        { label: 'Minimize', accelerator: 'CmdOrCtrl+M', role: 'minimize' },
        { label: 'Close', accelerator: 'CmdOrCtrl+W', role: 'close' }
      ]
    }
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
};

app.whenReady().then(() => {
  createMenu();
}); 