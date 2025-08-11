const { app, BrowserWindow, Menu, shell, ipcMain, dialog, systemPreferences } = require('electron');
const path = require('path');
const fs = require('fs');
const os = require('os');
const isDev = require('electron-is-dev');
const { spawn, exec } = require('child_process');
const screenshot = require('screenshot-desktop');
const http = require('http');
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
    
    // Log all sources to debug Zoom detection
    console.log('[getCaptureSources] All available windows:');
    sources.forEach(source => {
      if (!source.display_id) {
        console.log(`  - ${source.name} (ID: ${source.id})`);
      }
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
        // Try native capture helper first
        let allWindows = [];
        
        if (nativeCaptureHelper && nativeCaptureHelper.isRunning) {
          safeLog.log('Using native capture helper for window detection');
          try {
            allWindows = await nativeCaptureHelper.getAllWindows();
          } catch (error) {
            safeLog.log(`Native helper failed: ${error.message}, falling back to macWindowManager`);
            const macWindowManager = require('./macWindowManager');
            allWindows = await macWindowManager.getAllWindows();
          }
        } else {
          safeLog.log('Falling back to macWindowManager for window detection');
          const macWindowManager = require('./macWindowManager');
          allWindows = await macWindowManager.getAllWindows();
        }
        
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
    
    // Add virtual browser-based app entries for popular web services
    // This allows users to select "Zoom in Browser", "Teams in Browser", etc.
    const browserBasedApps = [
      { name: 'Zoom (Browser)', service: 'zoom', icon: '📹', color: '#2D8CFF' },
      { name: 'Microsoft Teams (Browser)', service: 'teams', icon: '👥', color: '#6264A7' },
      { name: 'Slack (Browser)', service: 'slack', icon: '💬', color: '#4A154B' },
      { name: 'Notion (Browser)', service: 'notion', icon: '📝', color: '#000000' },
      { name: 'Discord (Browser)', service: 'discord', icon: '🎮', color: '#5865F2' },
      { name: 'Figma (Browser)', service: 'figma', icon: '🎨', color: '#F24E1E' },
      { name: 'Miro (Browser)', service: 'miro', icon: '📋', color: '#FFD02F' },
      { name: 'GitHub (Browser)', service: 'github', icon: '🐙', color: '#24292e' },
      { name: 'Gmail (Browser)', service: 'gmail', icon: '📧', color: '#EA4335' },
      { name: 'Google Calendar (Browser)', service: 'calendar', icon: '📅', color: '#4285F4' },
      { name: 'Google Docs (Browser)', service: 'docs', icon: '📄', color: '#4285F4' }
    ];
    
    // Check if browsers are available before adding browser-based apps
    const hasBrowsers = formattedSources.some(source => {
      const name = source.name.toLowerCase();
      return name.includes('chrome') || name.includes('safari') || name.includes('firefox');
    });
    
    if (hasBrowsers) {
      for (const app of browserBasedApps) {
        const virtualId = `virtual-browser-app:${app.service}`;
        
        // Create SVG placeholder for browser-based app
        const svg = `
          <svg width="256" height="144" xmlns="http://www.w3.org/2000/svg">
            <rect width="256" height="144" fill="${app.color}"/>
            <text x="128" y="60" font-family="Arial, sans-serif" font-size="32" text-anchor="middle" fill="white">${app.icon}</text>
            <text x="128" y="85" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="white">${app.name}</text>
            <text x="128" y="100" font-family="Arial, sans-serif" font-size="10" text-anchor="middle" fill="#cccccc">Web App</text>
          </svg>
        `;
        
        formattedSources.push({
          id: virtualId,
          name: app.name,
          type: 'window',
          thumbnail: `data:image/svg+xml;base64,${Buffer.from(svg).toString('base64')}`,
          appIcon: null,
          isVisible: true // These are always selectable
        });
      }
      
      safeLog.log(`Added ${browserBasedApps.length} browser-based app entries`);
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

// IPC handler for requesting accessibility permissions for enhanced screen sharing detection
ipcMain.handle('request-screen-sharing-detection-permissions', async () => {
  try {
    safeLog.log('[Enhanced Permissions] Requesting accessibility permissions for screen sharing detection...');
    
    const { exec } = require('child_process');
    const { promisify } = require('util');
    const execAsync = promisify(exec);
    
    try {
      // Test accessibility permissions by trying to get window information
      // This will trigger the accessibility permission dialog if not granted
      const testScript = `
        tell application "System Events"
          try
            set zoomApps to (every process whose name contains "zoom")
            if (count of zoomApps) > 0 then
              set zoomApp to item 1 of zoomApps
              set windowList to (name of every window of zoomApp)
              return "SUCCESS: " & (count of windowList) as string
            else
              return "SUCCESS: No Zoom found"
            end if
          on error errMsg
            return "ERROR: " & errMsg
          end try
        end tell
      `;
      
      safeLog.log('[Enhanced Permissions] Testing accessibility access with AppleScript...');
      const { stdout } = await execAsync(`osascript -e '${testScript}'`, { timeout: 10000 });
      
      safeLog.log('[Enhanced Permissions] AppleScript result:', stdout.trim());
      
      if (stdout.includes('SUCCESS:')) {
        safeLog.log('[Enhanced Permissions] Accessibility permissions verified');
        return {
          success: true,
          permissions_requested: true,
          can_capture: true,
          message: 'Accessibility permissions granted - enhanced detection available'
        };
      } else if (stdout.includes('ERROR:') && stdout.includes('not allowed assistive access')) {
        safeLog.log('[Enhanced Permissions] Accessibility permission denied - opening System Preferences');
        
        // Open System Preferences to Accessibility settings
        try {
          await execAsync('open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"');
        } catch (openError) {
          safeLog.warn('[Enhanced Permissions] Failed to open System Preferences:', openError.message);
        }
        
        return {
          success: true,
          permissions_requested: true,
          can_capture: false,
          message: 'Accessibility permission required. Please enable MIRIX in System Preferences > Privacy & Security > Accessibility'
        };
      } else {
        return {
          success: false,
          permissions_requested: false,
          error: 'Unexpected AppleScript result',
          message: 'Failed to test accessibility permissions'
        };
      }
      
    } catch (error) {
      safeLog.error('[Enhanced Permissions] Accessibility test failed:', error.message);
      
      // If the error suggests permission issue, try to open System Preferences
      if (error.message.includes('not allowed assistive access') || 
          error.message.includes('accessibility')) {
        try {
          await execAsync('open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"');
        } catch (openError) {
          safeLog.warn('[Enhanced Permissions] Failed to open System Preferences:', openError.message);
        }
        
        return {
          success: true,
          permissions_requested: true,
          can_capture: false,
          message: 'Accessibility permission required. Please enable MIRIX in System Preferences > Privacy & Security > Accessibility'
        };
      }
      
      return {
        success: false,
        permissions_requested: false,
        error: error.message,
        message: 'Failed to request accessibility permissions'
      };
    }
    
  } catch (error) {
    safeLog.error('[Enhanced Permissions] Error requesting permissions:', error);
    return {
      success: false,
      permissions_requested: false,
      error: error.message,
      message: 'Enhanced permissions unavailable'
    };
  }
});

// Helper function to detect Google Meet screen sharing regardless of focused app
async function detectGoogleMeetScreenSharing() {
  try {
    const { exec } = require('child_process');
    const { promisify } = require('util');
    const execAsync = promisify(exec);
    
    console.log(`\n[Google Meet Independent Detection] ========================================`);
    console.log(`[Google Meet Independent] Checking for screen sharing regardless of focused app...`);
    
    // Method 0: Precise screen sharing detection to avoid false positives
    let systemScreenRecordingActive = false;
    try {
      // Check for ACTIVE screen sharing processes more precisely
      // Focus on processes that indicate actual active screen recording/sharing
      const activeRecordingCheck = await execAsync('ps aux | grep -E "(ScreenCaptureKit.*Chrome|screencapturingagent.*--enable-screen-capture)" | grep -v grep | wc -l');
      const activeRecordingCount = parseInt(activeRecordingCheck.stdout.trim()) || 0;
      
      // Check Chrome Video Capture CPU usage - only consider significant activity
      const chromeVidCheck = await execAsync('ps aux | grep "video_capture.mojom.VideoCaptureService" | grep -v grep | awk \'{print $3}\' | head -1');
      const chromeVidCpu = parseFloat(chromeVidCheck.stdout.trim()) || 0;
      
      // More restrictive: require significant Chrome video CPU OR multiple active recording processes
      if (activeRecordingCount >= 2 || chromeVidCpu > 3.0) {
        systemScreenRecordingActive = true;
        console.log(`[Google Meet Independent] Active screen recording detected (active processes: ${activeRecordingCount}, Chrome video CPU: ${chromeVidCpu}%)`);
      } else {
        console.log(`[Google Meet Independent] No significant screen recording activity (active processes: ${activeRecordingCount}, Chrome video CPU: ${chromeVidCpu}%)`);
      }
    } catch (e) {
      console.log(`[Google Meet Independent] Screen sharing check failed: ${e.message}`);
    }
    
    // Method 1: Check for ControlCenter menu bar icon via process activity
    let controlCenterActive = false;
    try {
      // Simple check for ControlCenter activity without complex AppleScript
      const ccCheck = await execAsync('ps aux | grep "ControlCenter" | grep -v grep | wc -l');
      const ccCount = parseInt(ccCheck.stdout.trim()) || 0;
      
      if (ccCount > 0) {
        controlCenterActive = true;
        console.log(`[Google Meet Independent] ControlCenter process active (likely showing screen sharing indicator)`);
      }
    } catch (e) {
      console.log(`[Google Meet Independent] ControlCenter check failed: ${e.message}`);
    }

    // Method 2: Check for Chrome Video Capture Service (restrictive to avoid false positives)
    let videoCaptureServiceDetected = false;
    try {
      const videoCaptureCheck = await execAsync('ps aux | grep -i chrome | grep -i "video.*capture\\|screen.*capture\\|media.*stream"');
      if (videoCaptureCheck.stdout.includes('video_capture.mojom.VideoCaptureService')) {
        // Extract CPU usage from the video capture process
        const videoCaptureProcess = videoCaptureCheck.stdout.split('\n').find(line => 
          line.includes('video_capture.mojom.VideoCaptureService')
        );
        
        if (videoCaptureProcess) {
          const cpuMatch = videoCaptureProcess.match(/\s+(\d+\.\d+)\s+/);
          const cpuUsage = cpuMatch ? parseFloat(cpuMatch[1]) : 0;
          
          // Much more restrictive: only consider active if significant CPU usage is detected
          // This avoids false positives from idle video capture services
          if (cpuUsage > 2.0) {
            videoCaptureServiceDetected = true;
            console.log(`[Google Meet Independent] Chrome Video Capture Service actively processing (CPU: ${cpuUsage}%)`);
          } else {
            console.log(`[Google Meet Independent] Chrome Video Capture Service idle (CPU: ${cpuUsage}% - too low for active sharing)`);
          }
        }
      } else {
        console.log(`[Google Meet Independent] Chrome Video Capture Service not found`);
      }
    } catch (e) {
      console.log(`[Google Meet Independent] Video capture service check failed: ${e.message}`);
    }
    
    // Method 3: Very precise Google Meet detection - require multiple strong indicators
    let googleMeetDetected = false;
    let meetingTitle = '';
    try {
      // Only detect Google Meet if we have STRONG evidence: active video capture AND system-level indicators
      if (videoCaptureServiceDetected && systemScreenRecordingActive && controlCenterActive) {
        const chromeProcesses = await execAsync('ps aux | grep -i "Google Chrome" | grep -v grep | wc -l');
        const chromeCount = parseInt(chromeProcesses.stdout.trim()) || 0;
        
        if (chromeCount > 0) {
          googleMeetDetected = true;
          meetingTitle = 'Google Meet (Active screen sharing confirmed)';
          console.log(`[Google Meet Independent] Google Meet confirmed with multiple strong indicators`);
        }
      } else {
        console.log(`[Google Meet Independent] Insufficient evidence for active Google Meet screen sharing`);
        console.log(`  - Video Capture Active: ${videoCaptureServiceDetected}`);
        console.log(`  - System Recording Active: ${systemScreenRecordingActive}`);
        console.log(`  - ControlCenter Active: ${controlCenterActive}`);
      }
    } catch (e) {
      console.log(`[Google Meet Independent] Google Meet detection failed: ${e.message}`);
    }
    
    // Determine screen sharing status - very restrictive approach to eliminate false positives
    const hasScreenSharingIndicators = videoCaptureServiceDetected && systemScreenRecordingActive && controlCenterActive;
    
    // Method 4: Detect which display is being shared (for multi-monitor setups)
    let detectedSharedDisplay = null;
    if (hasScreenSharingIndicators) {
      try {
        // Get all displays
        const displayScript = `
          tell application "System Events"
            try
              set displayCount to (do shell script "system_profiler SPDisplaysDataType | grep 'Display Type' | wc -l")
              return "DISPLAYS:" & displayCount
            on error
              return "DISPLAYS:1"
            end try
          end tell
        `;
        
        const displayResult = await execAsync(`osascript -e '${displayScript}'`);
        let displayCount = 1;
        if (displayResult.stdout.includes('DISPLAYS:')) {
          displayCount = parseInt(displayResult.stdout.split(':')[1]) || 1;
        }
        
        console.log(`[Google Meet Independent] System has ${displayCount} display(s)`);
        
        if (displayCount > 1) {
          // Try to detect which display is being shared by checking Chrome window positions
          const chromeDisplayScript = `
            tell application "System Events"
              try
                set chromeProcesses to (every process whose name contains "Chrome")
                set displayResults to {}
                
                repeat with chromeProcess in chromeProcesses
                  try
                    set chromeWindows to (every window of chromeProcess)
                    repeat with chromeWindow in chromeWindows
                      try
                        set windowName to (name of chromeWindow) as string
                        if windowName contains "Google Meet" or windowName contains "meet.google.com" then
                          set windowPosition to (position of chromeWindow)
                          set windowX to (item 1 of windowPosition)
                          -- Determine which display based on X position
                          if windowX > 1440 then
                            set end of displayResults to "Display 2"
                          else
                            set end of displayResults to "Display 1"  
                          end if
                        end if
                      end try
                    end repeat
                  end try
                end repeat
                
                return "WINDOW_DISPLAYS:" & (displayResults as string)
              on error errMsg
                return "ERROR:" & errMsg
              end try
            end tell
          `;
          
          const chromeDisplayResult = await execAsync(`osascript -e '${chromeDisplayScript}'`);
          
          if (chromeDisplayResult.stdout.includes('WINDOW_DISPLAYS:')) {
            const displayInfo = chromeDisplayResult.stdout.split(':')[1];
            if (displayInfo.includes('Display 2')) {
              detectedSharedDisplay = {
                id: '1', // Secondary display typically has ID 1 in Electron
                name: 'Display 2'
              };
            } else {
              detectedSharedDisplay = {
                id: '0', // Primary display typically has ID 0 in Electron
                name: 'Display 1'
              };
            }
            console.log(`[Google Meet Independent] Detected shared display from window position: ${detectedSharedDisplay.name}`);
          }
        } else {
          // Single display setup
          detectedSharedDisplay = {
            id: '0',
            name: 'Display 1'
          };
          console.log(`[Google Meet Independent] Single display setup, using primary display`);
        }
      } catch (e) {
        console.log(`[Google Meet Independent] Display detection failed: ${e.message}`);
        // Fallback to primary display
        detectedSharedDisplay = {
          id: '0',
          name: 'Primary Display'
        };
      }
    }
    
    console.log(`[Google Meet Independent] Detection results:`);
    console.log(`  - System Screen Recording Processes: ${systemScreenRecordingActive}`);
    console.log(`  - ControlCenter Active: ${controlCenterActive}`);
    console.log(`  - Chrome Video Capture Service: ${videoCaptureServiceDetected}`);
    console.log(`  - Google Meet Detected: ${googleMeetDetected}`);
    console.log(`  - Screen sharing detected: ${hasScreenSharingIndicators}`);
    console.log(`  - Shared display detected: ${detectedSharedDisplay ? detectedSharedDisplay.name : 'None'}`);
    
    // Return results
    if (googleMeetDetected && hasScreenSharingIndicators) {
      console.log(`[Google Meet Independent] ✅ GOOGLE MEET WITH SCREEN SHARING DETECTED`);
      return {
        detected: true,
        isScreenSharing: true,
        meetingStatus: 'In Meeting - Screen Sharing',
        screenSharingStatus: 'active',
        displayInfo: 'Google Meet - Screen Sharing',
        windowTitle: meetingTitle || 'Google Meet',
        sharedDisplay: detectedSharedDisplay
      };
    } else if (googleMeetDetected) {
      console.log(`[Google Meet Independent] ✅ GOOGLE MEET DETECTED (no screen sharing)`);
      return {
        detected: true,
        isScreenSharing: false,
        meetingStatus: 'In Meeting',
        screenSharingStatus: 'inactive',
        displayInfo: 'Google Meet',
        windowTitle: meetingTitle || 'Google Meet',
        sharedDisplay: null
      };
    } else if (hasScreenSharingIndicators) {
      console.log(`[Google Meet Independent] ⚠️ SCREEN SHARING DETECTED (no Google Meet confirmation)`);
      return {
        detected: false,
        isScreenSharing: true,
        meetingStatus: 'Unknown',
        screenSharingStatus: 'active',
        displayInfo: 'Screen Sharing Active',
        windowTitle: 'Unknown',
        sharedDisplay: detectedSharedDisplay
      };
    }
    
    console.log(`[Google Meet Independent] ❌ No Google Meet screen sharing detected`);
    return {
      detected: false,
      isScreenSharing: false,
      meetingStatus: null,
      screenSharingStatus: 'inactive',
      displayInfo: null,
      windowTitle: null
    };
    
  } catch (error) {
    console.log(`[Google Meet Independent] Error in detection: ${error.message}`);
    return {
      detected: false,
      isScreenSharing: false,
      meetingStatus: null,
      screenSharingStatus: 'unknown',
      displayInfo: null,
      windowTitle: null,
      error: error.message
    };
  }
}

// IPC handler for getting the currently focused app
ipcMain.handle('get-focused-app', async () => {
  try {
    if (process.platform === 'darwin') {
      const { exec } = require('child_process');
      const { promisify } = require('util');
      const execAsync = promisify(exec);
      const { desktopCapturer } = require('electron');
      
      // Get the frontmost application and its window title
      const appScript = `osascript -e 'tell application "System Events" to get name of first application process whose frontmost is true'`;
      const windowScript = `osascript -e '
        tell application "System Events"
          set frontApp to first application process whose frontmost is true
          try
            set windowTitle to title of first window of frontApp
            return windowTitle
          on error
            return ""
          end try
        end tell'`;
      
      const [appResult, windowResult] = await Promise.allSettled([
        execAsync(appScript),
        execAsync(windowScript)
      ]);
      
      let activeAppName = null;
      let activeWindowTitle = null;
      
      if (appResult.status === 'fulfilled') {
        activeAppName = appResult.value.stdout.trim();
      }
      if (windowResult.status === 'fulfilled') {
        activeWindowTitle = windowResult.value.stdout.trim();
      }
      
      // Enhanced information for different apps
      let displayInfo = activeWindowTitle || '';
      let tabInfo = null;
      let meetingStatus = null;
      let screenSharingStatus = null;
      
      // For Zoom, determine if in a meeting using medium-permission approach
      if (activeAppName && activeAppName.toLowerCase().includes('zoom')) {
        console.log(`\n[Zoom Detection - Medium Permission] ========================================`);
        console.log(`[Zoom Detection] Focused App: ${activeAppName}`);
        
        try {
          // Process analysis (no permissions needed)
          const psOutput = await execAsync('ps aux | grep -i zoom | grep -v grep');
          const zoomProcesses = psOutput.stdout.trim().split('\n').filter(line => line.trim());
          
          const meetingProcesses = zoomProcesses.filter(proc => {
            const lower = proc.toLowerCase();
            return lower.includes('cpthost') || 
                   lower.includes('caphost') || 
                   lower.includes('aomhost') ||
                   lower.includes('meeting');
          });
          
          console.log(`[Zoom Detection] Total processes: ${zoomProcesses.length}`);
          console.log(`[Zoom Detection] Meeting processes: ${meetingProcesses.length}`);
          
          // Resource usage
          const cpuOutput = await execAsync('ps -p $(pgrep zoom.us) -o %cpu= 2>/dev/null || echo "0"');
          const cpuUsage = parseFloat(cpuOutput.stdout.trim()) || 0;
          console.log(`[Zoom Detection] CPU usage: ${cpuUsage}%`);
          
          // Network activity
          const netOutput = await execAsync('lsof -c zoom | grep -E "TCP|UDP" | wc -l 2>/dev/null || echo "0"');
          const networkConnections = parseInt(netOutput.stdout.trim()) || 0;
          console.log(`[Zoom Detection] Network connections: ${networkConnections}`);
          
          // ENHANCED SCREEN SHARING DETECTION
          // Use multiple detection methods with elevated permissions for reliable detection
          const isScreenSharing = await (async () => {
            try {
              console.log('[Zoom Screen Share Detection] Starting enhanced detection...');
              let detectionScores = [];
              
              // Method 1: Check for Zoom screen sharing processes (more specific patterns)
              try {
                const { stdout: zoomProcesses } = await execAsync(
                  `ps aux | grep -E "zoom.us|ZoomOpener" | grep -v grep`
                );
                const hasShareProcess = zoomProcesses.toLowerCase().includes('screenshare') ||
                                       zoomProcesses.toLowerCase().includes('share') ||
                                       zoomProcesses.toLowerCase().includes('capture');
                if (hasShareProcess) detectionScores.push('zoom_share_process');
                console.log(`[Detection Method 1] Zoom share processes: ${hasShareProcess}`);
              } catch (e) {}
              
              // Method 2: Check for system screen capture services (more specific)
              try {
                const { stdout: screenServices } = await execAsync(
                  `ps aux | grep -E "screencapturingagent.*zoom|ScreenSearch.*zoom" | grep -v grep | wc -l`
                );
                const screenServiceCount = parseInt(screenServices.trim()) || 0;
                // Only count if there are Zoom-related screen capture services
                if (screenServiceCount > 0) detectionScores.push('system_screen_services');
                console.log(`[Detection Method 2] Zoom-related screen services count: ${screenServiceCount}`);
              } catch (e) {}
              
              // Method 3: Enhanced Zoom window analysis with accessibility APIs
              try {
                const zoomWindowScript = `
                  tell application "System Events"
                    try
                      set zoomProcesses to (every process whose name contains "zoom")
                      set shareIndicators to {}
                      
                      repeat with zoomProcess in zoomProcesses
                        try
                          set windowNames to (name of every window of zoomProcess)
                          set processName to (name of zoomProcess)
                          
                          repeat with windowName in windowNames
                            set windowNameStr to windowName as string
                            -- Be extremely specific about sharing indicators
                            if windowNameStr is equal to "You are sharing your screen" or windowNameStr is equal to "Stop Share" or windowNameStr contains "Stop sharing" then
                              set end of shareIndicators to ("SHARING:" & processName & ":" & windowNameStr)
                            end if
                            
                            -- Check for specific Zoom sharing UI elements
                            if windowNameStr contains "Zoom Meeting" then
                              try
                                set windowElements to (UI elements of window windowName of zoomProcess)
                                repeat with element in windowElements
                                  set elementDescription to (description of element) as string
                                  if elementDescription contains "sharing" or elementDescription contains "Stop Share" then
                                    set end of shareIndicators to ("UI_SHARING:" & processName & ":" & elementDescription)
                                  end if
                                end repeat
                              end try
                            end if
                          end repeat
                        end try
                      end repeat
                      
                      return "RESULT:" & (count of shareIndicators) & ":" & (shareIndicators as string)
                    on error errMsg
                      return "ERROR:" & errMsg
                    end try
                  end tell
                `;
                
                const { stdout: windowResult } = await execAsync(`osascript -e '${zoomWindowScript}'`);
                
                let hasShareIndicator = false;
                if (windowResult.includes('RESULT:')) {
                  const parts = windowResult.split(':');
                  const shareCount = parseInt(parts[1]) || 0;
                  hasShareIndicator = shareCount > 0;
                  
                  console.log(`[Detection Method 3] Advanced window analysis: ${shareCount} sharing indicators found`);
                  if (shareCount > 0) {
                    console.log(`[Detection Method 3] Sharing details: ${parts.slice(2).join(':')}`);
                  }
                } else {
                  console.log(`[Detection Method 3] Accessibility check failed, falling back to basic detection`);
                  // Fallback to very specific basic window name checking
                  const hasBasicIndicator = windowResult.includes('You are sharing your screen') ||
                                           windowResult.includes('Stop Share') ||
                                           windowResult.includes('Stop sharing');
                  hasShareIndicator = hasBasicIndicator;
                  console.log(`[Detection Method 3] Fallback basic detection result: ${hasBasicIndicator}`);
                }
                
                if (hasShareIndicator) detectionScores.push('zoom_window_indicator');
                console.log(`[Detection Method 3] Enhanced window sharing indicator: ${hasShareIndicator}`);
              } catch (e) {
                console.log(`[Detection Method 3] Enhanced window detection failed: ${e.message}`);
              }
              
              // Method 4: Advanced system monitoring (requires elevated permissions)
              try {
                // Check for CoreGraphics display stream creation (indicates screen capture)
                const { stdout: displayStreams } = await execAsync(
                  `lsof -c zoom | grep -i display | wc -l 2>/dev/null || echo "0"`
                );
                const streamCount = parseInt(displayStreams.trim()) || 0;
                if (streamCount > 0) detectionScores.push('display_streams');
                console.log(`[Detection Method 4] Display streams: ${streamCount}`);
              } catch (e) {}
              
              // Method 5: Check for IOSurface usage (screen sharing uses IOSurface)
              try {
                const { stdout: ioSurface } = await execAsync(
                  `lsof -c zoom | grep -i iosurface | wc -l 2>/dev/null || echo "0"`
                );
                const surfaceCount = parseInt(ioSurface.trim()) || 0;
                if (surfaceCount > 2) detectionScores.push('iosurface_usage');
                console.log(`[Detection Method 5] IOSurface usage: ${surfaceCount}`);
              } catch (e) {}
              
              // Method 6: Monitor WindowServer connections (screen sharing creates specific connections)
              try {
                const { stdout: windowServer } = await execAsync(
                  `lsof -p $(pgrep zoom.us) | grep -i windowserver | wc -l 2>/dev/null || echo "0"`
                );
                const wsConnections = parseInt(windowServer.trim()) || 0;
                if (wsConnections > 1) detectionScores.push('windowserver_connections');
                console.log(`[Detection Method 6] WindowServer connections: ${wsConnections}`);
              } catch (e) {}
              
              // Method 7: CPU and memory pattern analysis for screen sharing (more conservative)
              const highCpuForSharing = cpuUsage > 45; // Raised threshold to be more conservative
              try {
                const { stdout: zoomMemory } = await execAsync(
                  `ps -o pid,rss,comm -p $(pgrep zoom.us) | awk 'NR>1 {sum+=$2} END {print sum}'`
                );
                const memoryUsage = parseInt(zoomMemory.trim()) || 0;
                // Screen sharing typically uses significantly more memory AND high network connections
                if (highCpuForSharing && memoryUsage > 200000 && networkConnections > 15) { // Stricter criteria
                  detectionScores.push('resource_pattern');
                }
                console.log(`[Detection Method 7] CPU: ${cpuUsage}%, Memory: ${memoryUsage}KB, Network: ${networkConnections}, High resource pattern: ${highCpuForSharing && memoryUsage > 200000 && networkConnections > 15}`);
              } catch (e) {}
              
              // Method 8: Advanced accessibility-based screen sharing detection with display identification
              try {
                const accessibilityScript = `
                  tell application "System Events"
                    try
                      set shareDetected to false
                      set detectionDetails to ""
                      set sharedDisplayInfo to ""
                      
                      -- Look for Zoom processes
                      set zoomProcesses to (every process whose name contains "zoom")
                      
                      repeat with zoomProcess in zoomProcesses
                        try
                          set processName to (name of zoomProcess)
                          
                          -- Method 8a: Check for green sharing indicator in menu bar
                          try
                            set menuBarItems to (UI elements of menu bar 1 of zoomProcess)
                            repeat with menuItem in menuBarItems
                              set menuDesc to (description of menuItem) as string
                              if menuDesc contains "sharing" or menuDesc contains "Share Screen" then
                                set shareDetected to true
                                set detectionDetails to detectionDetails & "MENUBAR_SHARE;"
                              end if
                            end repeat
                          end try
                          
                          -- Method 8b: Check window controls for sharing buttons and display info
                          set windows to (every window of zoomProcess)
                          repeat with zoomWindow in windows
                            try
                              set windowName to (name of zoomWindow) as string
                              if windowName contains "Zoom Meeting" or windowName contains "zoom.us" then
                                
                                -- Look for sharing control buttons
                                set buttons to (every button of zoomWindow)
                                repeat with btn in buttons
                                  try
                                    set buttonTitle to (title of btn) as string
                                    set buttonDesc to (description of btn) as string
                                    
                                    if buttonTitle contains "Stop Share" or buttonTitle contains "sharing" or buttonDesc contains "Stop sharing" then
                                      set shareDetected to true
                                      set detectionDetails to detectionDetails & "STOP_SHARE_BUTTON;"
                                    end if
                                    
                                    if buttonTitle contains "Share Screen" and (buttonDesc contains "selected" or buttonDesc contains "active") then
                                      set shareDetected to true
                                      set detectionDetails to detectionDetails & "ACTIVE_SHARE_BUTTON;"
                                    end if
                                  end try
                                end repeat
                                
                                -- Look for sharing status indicators and display information
                                set staticTexts to (every static text of zoomWindow)
                                repeat with textElement in staticTexts
                                  try
                                    set textValue to (value of textElement) as string
                                    if textValue contains "You are sharing" or textValue contains "sharing your screen" then
                                      set shareDetected to true
                                      set detectionDetails to detectionDetails & "SHARING_STATUS_TEXT;"
                                      
                                      -- Try to extract display information from sharing status
                                      if textValue contains "Display" or textValue contains "Screen" then
                                        set sharedDisplayInfo to textValue
                                      end if
                                    end if
                                    
                                    -- Look for specific display sharing indicators
                                    if textValue contains "Sharing Screen" and (textValue contains "1" or textValue contains "2" or textValue contains "3") then
                                      set shareDetected to true
                                      set sharedDisplayInfo to textValue
                                      set detectionDetails to detectionDetails & "DISPLAY_INFO;"
                                    end if
                                  end try
                                end repeat
                                
                              end if
                            end try
                          end repeat
                        end try
                      end repeat
                      
                      if shareDetected then
                        return "SHARING_DETECTED:" & detectionDetails & "|DISPLAY:" & sharedDisplayInfo
                      else
                        return "NO_SHARING_DETECTED"
                      end if
                      
                    on error errMsg
                      return "ERROR:" & errMsg
                    end try
                  end tell
                `;
                
                const { stdout: accessibilityResult } = await execAsync(`osascript -e '${accessibilityScript}'`);
                
                let isAccessibilitySharing = false;
                let sharedDisplayInfo = '';
                
                if (accessibilityResult.includes('SHARING_DETECTED:')) {
                  isAccessibilitySharing = true;
                  
                  // Parse the result to extract details and display info
                  const parts = accessibilityResult.split('|');
                  const details = parts[0].split(':')[1] || '';
                  
                  if (parts.length > 1 && parts[1].startsWith('DISPLAY:')) {
                    sharedDisplayInfo = parts[1].substring(8); // Remove 'DISPLAY:' prefix
                  }
                  
                  console.log(`[Detection Method 8] Accessibility-based sharing detected: ${details}`);
                  if (sharedDisplayInfo) {
                    console.log(`[Detection Method 8] Shared display info: ${sharedDisplayInfo}`);
                  }
                  
                  detectionScores.push('accessibility_detection');
                } else if (accessibilityResult.includes('ERROR:')) {
                  console.log(`[Detection Method 8] Accessibility API error (permissions may be needed): ${accessibilityResult}`);
                } else {
                  console.log(`[Detection Method 8] No accessibility-based sharing detected`);
                }
                
                console.log(`[Detection Method 8] Accessibility sharing detection: ${isAccessibilitySharing}`);
              } catch (e) {
                console.log(`[Detection Method 8] Accessibility detection failed: ${e.message}`);
              }
              
              // Method 9: Display-specific screen sharing detection
              let detectedSharedDisplay = null;
              try {
                // Get all available displays
                const displaysResult = await window.electronAPI.listDisplays();
                
                if (displaysResult.success && displaysResult.displays.length > 1) {
                  console.log(`[Detection Method 9] Multiple displays detected: ${displaysResult.displays.length}`);
                  
                  // Check for display capture activity using lsof
                  const { stdout: displayCapture } = await execAsync(
                    `lsof -c zoom | grep -E "CGDisplay|IOSurface" | head -5`
                  );
                  
                  if (displayCapture.trim()) {
                    console.log(`[Detection Method 9] Display capture activity detected`);
                    
                    // Try to identify which display is being captured
                    for (const display of displaysResult.displays) {
                      try {
                        // Check if this specific display is being captured
                        const { stdout: displaySpecific } = await execAsync(
                          `lsof -c zoom | grep "${display.id}" | wc -l`
                        );
                        const captureCount = parseInt(displaySpecific.trim()) || 0;
                        
                        if (captureCount > 0) {
                          detectedSharedDisplay = {
                            id: display.id,
                            name: display.name || `Display ${display.id}`,
                            bounds: display.bounds,
                            primary: display.primary
                          };
                          console.log(`[Detection Method 9] Detected shared display: ${detectedSharedDisplay.name} (${detectedSharedDisplay.id})`);
                          break;
                        }
                      } catch (e) {}
                    }
                    
                    // If no specific display detected, try to infer from Zoom UI
                    if (!detectedSharedDisplay && sharedDisplayInfo) {
                      // Parse display info from Zoom UI text
                      const displayMatch = sharedDisplayInfo.match(/(?:Screen|Display)\s*(\d+)/i);
                      if (displayMatch) {
                        const displayNumber = parseInt(displayMatch[1]);
                        const targetDisplay = displaysResult.displays.find(d => d.id === displayNumber.toString()) ||
                                            displaysResult.displays[displayNumber - 1]; // 1-based indexing
                        
                        if (targetDisplay) {
                          detectedSharedDisplay = {
                            id: targetDisplay.id,
                            name: targetDisplay.name || `Display ${displayNumber}`,
                            bounds: targetDisplay.bounds,
                            primary: targetDisplay.primary
                          };
                          console.log(`[Detection Method 9] Inferred shared display from UI: ${detectedSharedDisplay.name}`);
                        }
                      }
                    }
                  }
                }
                
                console.log(`[Detection Method 9] Shared display detection: ${detectedSharedDisplay ? detectedSharedDisplay.name : 'None detected'}`);
              } catch (e) {
                console.log(`[Detection Method 9] Display detection failed: ${e.message}`);
              }
              
              // ULTRA CONSERVATIVE DETECTION: Only trust the most reliable methods
              // Only accessibility detection is truly reliable
              const ultraHighConfidenceMethods = ['accessibility_detection'];
              const hasUltraHighConfidence = detectionScores.some(method => ultraHighConfidenceMethods.includes(method));
              
              // Window indicator must be very specific
              const hasSpecificWindowIndicator = detectionScores.includes('zoom_window_indicator');
              
              // Even if multiple methods trigger, be very conservative
              const multipleStrongIndicators = detectionScores.length >= 5 && 
                                               detectionScores.includes('display_streams') && 
                                               detectionScores.includes('iosurface_usage');
              
              // Final decision logic - EXTREMELY conservative to avoid false positives
              let isSharing = false;
              let confidenceReason = '';
              
              if (hasUltraHighConfidence) {
                isSharing = true;
                confidenceReason = 'Ultra high confidence: Accessibility API detected sharing UI elements';
              } else if (hasSpecificWindowIndicator && multipleStrongIndicators) {
                isSharing = true;
                confidenceReason = 'High confidence: Window indicators + multiple strong technical signals';
              } else {
                isSharing = false;
                if (detectionScores.length > 0) {
                  confidenceReason = `False positive protection: Only detected [${detectionScores.join(', ')}] but none are ultra-reliable`;
                } else {
                  confidenceReason = 'No screen sharing indicators detected';
                }
              }
              
              console.log(`[Zoom Screen Share Detection] Detection methods triggered: [${detectionScores.join(', ')}] (${detectionScores.length}/9)`);
              console.log(`[Zoom Screen Share Detection] Ultra high confidence: ${hasUltraHighConfidence}, Specific window: ${hasSpecificWindowIndicator}, Multiple strong: ${multipleStrongIndicators}`);
              console.log(`[Zoom Screen Share Detection] Final result: ${isSharing ? 'SCREEN SHARING DETECTED' : 'NO SCREEN SHARING'} (${confidenceReason})`);
              if (isSharing && detectedSharedDisplay) {
                console.log(`[Zoom Screen Share Detection] Shared display: ${detectedSharedDisplay.name} (${detectedSharedDisplay.id})`);
              }
              
              return {
                isSharing: isSharing,
                sharedDisplay: detectedSharedDisplay,
                confidenceReason: confidenceReason,
                detectionMethods: detectionScores
              };
            } catch (e) {
              console.log('[Zoom Screen Share Detection] Error in enhanced detection:', e.message);
              return false;
            }
          })();
          
          // Handle the enhanced sharing detection result
          const sharingResult = typeof isScreenSharing === 'object' ? isScreenSharing : { isSharing: isScreenSharing };
          const actuallySharing = sharingResult.isSharing;
          const sharedDisplay = sharingResult.sharedDisplay;
          
          // Decision logic - prioritize network connections as most reliable indicator
          if (networkConnections >= 10) {
            // High confidence: definitely in meeting
            if (actuallySharing) {
              let displaySuffix = '';
              if (sharedDisplay) {
                displaySuffix = ` (${sharedDisplay.name})`;
              }
              displayInfo = `Zoom Meeting - Screen Sharing${displaySuffix} (${networkConnections} connections, ${cpuUsage}% CPU)`;
              meetingStatus = 'In Meeting - Screen Sharing';
              screenSharingStatus = 'active';
              console.log(`[Zoom Detection] 🖥️ IN MEETING WITH SCREEN SHARING${displaySuffix}`);
            } else {
              displayInfo = `Zoom Meeting (${networkConnections} connections, ${cpuUsage}% CPU)`;
              meetingStatus = 'In Meeting';
              screenSharingStatus = 'inactive';
              console.log(`[Zoom Detection] ✅ IN MEETING (high network activity)`);
            }
          } else if (meetingProcesses.length >= 2 && cpuUsage > 20) {
            // Medium confidence: meeting processes + high CPU
            if (actuallySharing) {
              let displaySuffix = '';
              if (sharedDisplay) {
                displaySuffix = ` (${sharedDisplay.name})`;
              }
              displayInfo = `Zoom Meeting - Screen Sharing${displaySuffix} (${meetingProcesses.length} meeting processes, ${cpuUsage}% CPU)`;
              meetingStatus = 'In Meeting - Screen Sharing';
              screenSharingStatus = 'active';
              console.log(`[Zoom Detection] 🖥️ IN MEETING WITH SCREEN SHARING${displaySuffix}`);
            } else {
              displayInfo = `Zoom Meeting (${meetingProcesses.length} meeting processes, ${cpuUsage}% CPU)`;
              meetingStatus = 'In Meeting';
              screenSharingStatus = 'inactive';
              console.log(`[Zoom Detection] ✅ IN MEETING (meeting processes + high CPU)`);
            }
          } else {
            // Low network activity = homepage
            displayInfo = 'Zoom Workplace';
            meetingStatus = 'Not in meeting';
            screenSharingStatus = 'inactive';
            console.log(`[Zoom Detection] 🏠 ON HOMEPAGE (low network activity: ${networkConnections} connections)`);
          }
          
          console.log(`[Zoom Detection - Medium Permission] ========================================\n`);
          
        } catch (e) {
          console.error('[Zoom Detection] Error with enhanced detection:', e);
          // Fallback
          displayInfo = 'Zoom';
          meetingStatus = 'Status unknown';
          screenSharingStatus = 'unknown';
        }
      }
      
      // For browsers, extract tab information and detect Google Meet
      else if (activeAppName && (activeAppName.includes('Chrome') || activeAppName.includes('Safari') || activeAppName.includes('Firefox') || activeAppName.includes('Edge'))) {
        if (activeWindowTitle) {
          // Most browsers show: "Page Title - Site Name" or "Page Title — Site Name"
          const parts = activeWindowTitle.split(/\s[-—]\s/);
          if (parts.length >= 2) {
            tabInfo = {
              title: parts[0].trim(),
              site: parts[parts.length - 1].trim()
            };
            displayInfo = `${parts[0].trim()} (${parts[parts.length - 1].trim()})`;
            
            // Check for Google Meet
            const site = parts[parts.length - 1].trim().toLowerCase();
            const title = parts[0].trim().toLowerCase();
            
            if (site.includes('meet.google.com') || title.includes('meet') || 
                activeWindowTitle.toLowerCase().includes('google meet')) {
              console.log(`\n[Google Meet Detection] ========================================`);
              console.log(`[Google Meet] Window title: ${activeWindowTitle}`);
              console.log(`[Google Meet] Tab title: ${tabInfo.title}`);
              console.log(`[Google Meet] Site: ${tabInfo.site}`);
              
              // Detect Google Meet screen sharing using direct macOS system methods
              try {
                console.log(`[Google Meet] Checking for screen sharing using system-level detection...`);
                
                // Method 1: Check for Chrome Video Capture Service (most reliable indicator)
                let videoCaptureServiceDetected = false;
                try {
                  // Check for Chrome's video capture service which is used for screen sharing
                  const videoCaptureCheck = await execAsync('ps aux | grep -i chrome | grep -i "video.*capture\\|screen.*capture\\|media.*stream"');
                  if (videoCaptureCheck.stdout.includes('video_capture.mojom.VideoCaptureService')) {
                    videoCaptureServiceDetected = true;
                    console.log(`[Google Meet] Chrome Video Capture Service detected - strong screen sharing indicator`);
                  }
                } catch (e) {
                  console.log(`[Google Meet] Video capture service check failed: ${e.message}`);
                }
                
                // Method 2: Check active screen capture sessions using Core Graphics
                let activeCaptureSession = false;
                try {
                  // Use osascript to check for active screen capture via Core Graphics
                  const cgScript = `
                    tell application "System Events"
                      try
                        -- Check if there are active display capture sessions
                        set captureResult to (do shell script "ioreg -l | grep -i 'screencapture\\|displaycapture\\|CGDisplay' | head -5")
                        return captureResult
                      on error
                        return "no capture sessions"
                      end try
                    end tell
                  `;
                  
                  const cgResult = await execAsync(`osascript -e '${cgScript}'`);
                  if (cgResult.stdout && cgResult.stdout.includes('capture')) {
                    activeCaptureSession = true;
                    console.log(`[Google Meet] Active capture session detected via ioreg`);
                  }
                } catch (e) {
                  console.log(`[Google Meet] Core Graphics capture check failed: ${e.message}`);
                }
                
                // Method 3: Check for Screen Recording permission actively being used
                let activeScreenPermission = false;
                try {
                  // Check TCC database for recent screen recording grants to Chrome
                  const tccCheck = await execAsync('sqlite3 "/Library/Application Support/com.apple.TCC/TCC.db" "SELECT client, auth_value, last_modified FROM access WHERE service=\'kTCCServiceScreenCapture\' AND client LIKE \'%chrome%\' AND auth_value=2;" 2>/dev/null || echo "no access"');
                  if (tccCheck.stdout.includes('chrome') && !tccCheck.stdout.includes('no access')) {
                    activeScreenPermission = true;
                    console.log(`[Google Meet] Chrome has active screen recording permission`);
                  }
                } catch (e) {
                  console.log(`[Google Meet] TCC database check failed (expected on some systems): ${e.message}`);
                }
                
                // Method 4: Check WindowServer for active display connections
                let windowServerActivity = false;
                try {
                  // More specific WindowServer check for display connections
                  const wsConnections = await execAsync('lsof | grep "WindowServer.*Chrome" | grep -v grep');
                  if (wsConnections.stdout.trim()) {
                    windowServerActivity = true;
                    console.log(`[Google Meet] Chrome-WindowServer display connections detected`);
                  }
                } catch (e) {
                  console.log(`[Google Meet] WindowServer connection check failed: ${e.message}`);
                }
                
                // Method 5: Check for display stream usage via ioreg
                let displayStreamUsage = false;
                try {
                  // Check IORegistry for active display streams
                  const ioregResult = await execAsync('ioreg -l | grep -i -A 3 -B 3 "display.*stream\\|screen.*capture" | head -10');
                  if (ioregResult.stdout.trim() && ioregResult.stdout.length > 50) {
                    displayStreamUsage = true;
                    console.log(`[Google Meet] Display stream usage detected in IORegistry`);
                  }
                } catch (e) {
                  console.log(`[Google Meet] IORegistry display stream check failed: ${e.message}`);
                }
                
                // Combine all system-level detection methods
                const systemDetectionScores = [];
                
                if (videoCaptureServiceDetected) {
                  systemDetectionScores.push('video_capture_service'); // Strongest indicator
                }
                
                if (activeCaptureSession) {
                  systemDetectionScores.push('core_graphics');
                }
                
                if (activeScreenPermission) {
                  systemDetectionScores.push('tcc_permission');
                }
                
                if (windowServerActivity) {
                  systemDetectionScores.push('windowserver_connection');
                }
                
                if (displayStreamUsage) {
                  systemDetectionScores.push('display_stream');
                }
                
                // Final detection logic based on system-level indicators
                const hasScreenSharingIndicators = systemDetectionScores.length > 0;
                
                console.log(`[Google Meet] System-level detection analysis:`);
                console.log(`  - Video Capture Service: ${videoCaptureServiceDetected}`);
                console.log(`  - Active capture session: ${activeCaptureSession}`);
                console.log(`  - Screen permission active: ${activeScreenPermission}`);
                console.log(`  - WindowServer activity: ${windowServerActivity}`);
                console.log(`  - Display stream usage: ${displayStreamUsage}`);
                console.log(`  - Detection methods triggered: ${systemDetectionScores.join(', ')}`);
                console.log(`  - Screen sharing detected: ${hasScreenSharingIndicators}`);
                
                // Detect if in a meeting vs just on the landing page
                if (title.includes('meet') && !title.includes('meeting') && !title.includes('join')) {
                  // Likely in an active meeting (meeting titles don't usually contain "meet")
                  if (hasScreenSharingIndicators) {
                    meetingStatus = 'In Meeting - Screen Sharing';
                    screenSharingStatus = 'active';
                    displayInfo = `Google Meet - Screen Sharing (${tabInfo.title})`;
                    console.log(`[Google Meet] ✅ IN MEETING WITH SCREEN SHARING - Active meeting with sharing detected`);
                  } else {
                    meetingStatus = 'In Meeting';
                    screenSharingStatus = 'inactive';
                    displayInfo = `Google Meet (${tabInfo.title})`;
                    console.log(`[Google Meet] ✅ IN MEETING - Active meeting detected`);
                  }
                } else if (title.includes('join') || title.includes('meeting')) {
                  // On join page or meeting lobby
                  meetingStatus = 'Joining Meeting';
                  screenSharingStatus = 'inactive';
                  displayInfo = `Google Meet - Joining (${tabInfo.title})`;
                  console.log(`[Google Meet] 🚪 JOINING - On join/lobby page`);
                } else if (site.includes('meet.google.com')) {
                  // On Google Meet site but status unclear
                  meetingStatus = 'Google Meet Open';
                  screenSharingStatus = 'inactive';
                  displayInfo = `Google Meet (${tabInfo.title})`;
                  console.log(`[Google Meet] 📱 GOOGLE MEET OPEN - General Meet page`);
                }
              } catch (error) {
                console.log(`[Google Meet] Error detecting screen sharing: ${error.message}`);
                // Fallback to basic meeting detection
                if (title.includes('meet') && !title.includes('meeting') && !title.includes('join')) {
                  meetingStatus = 'In Meeting';
                  displayInfo = `Google Meet (${tabInfo.title})`;
                  console.log(`[Google Meet] ✅ IN MEETING - Active meeting detected (fallback)`);
                } else if (title.includes('join') || title.includes('meeting')) {
                  meetingStatus = 'Joining Meeting';
                  displayInfo = `Google Meet - Joining (${tabInfo.title})`;
                  console.log(`[Google Meet] 🚪 JOINING - On join/lobby page (fallback)`);
                } else if (site.includes('meet.google.com')) {
                  meetingStatus = 'Google Meet Open';
                  displayInfo = `Google Meet (${tabInfo.title})`;
                  console.log(`[Google Meet] 📱 GOOGLE MEET OPEN - General Meet page (fallback)`);
                }
                screenSharingStatus = 'unknown';
              }
            }
          } else {
            displayInfo = activeWindowTitle;
            
            // Check for Google Meet in single-part titles
            if (activeWindowTitle.toLowerCase().includes('google meet') || 
                activeWindowTitle.toLowerCase().includes('meet.google.com')) {
              meetingStatus = 'Google Meet Open';
              displayInfo = 'Google Meet';
              console.log(`[Google Meet] 📱 GOOGLE MEET OPEN - Basic detection`);
            }
          }
        }
      }
      
      // For other apps, use window title if available
      else if (activeWindowTitle && activeWindowTitle !== activeAppName) {
        displayInfo = activeWindowTitle;
      }
      
      // Always run independent Google Meet detection regardless of focused app
      let googleMeetResult = null;
      try {
        googleMeetResult = await detectGoogleMeetScreenSharing();
        console.log(`[get-focused-app] Independent Google Meet detection result:`, googleMeetResult);
      } catch (error) {
        console.log(`[get-focused-app] Independent Google Meet detection failed:`, error.message);
      }
      
      // Track shared display info (can come from Zoom or Google Meet detection)
      let sharedDisplay = null;
      
      // If independent Google Meet detection found screen sharing, prioritize it
      if (googleMeetResult && googleMeetResult.detected && googleMeetResult.isScreenSharing) {
        console.log(`[get-focused-app] ✅ Using independent Google Meet screen sharing result`);
        meetingStatus = googleMeetResult.meetingStatus;
        screenSharingStatus = googleMeetResult.screenSharingStatus;
        displayInfo = googleMeetResult.displayInfo;
        sharedDisplay = googleMeetResult.sharedDisplay;
        // Add additional info to indicate this was detected independently
        if (!tabInfo) {
          tabInfo = {
            title: googleMeetResult.windowTitle || 'Google Meet',
            site: 'meet.google.com'
          };
        }
      } else if (googleMeetResult && googleMeetResult.detected && !meetingStatus) {
        // Use Google Meet meeting status if no meeting was detected from focused app
        console.log(`[get-focused-app] ✅ Using independent Google Meet meeting result (no screen sharing)`);
        meetingStatus = googleMeetResult.meetingStatus;
        screenSharingStatus = googleMeetResult.screenSharingStatus;
        displayInfo = googleMeetResult.displayInfo;
        sharedDisplay = googleMeetResult.sharedDisplay;
        if (!tabInfo) {
          tabInfo = {
            title: googleMeetResult.windowTitle || 'Google Meet',
            site: 'meet.google.com'
          };
        }
      }
      
      return {
        success: true,
        appName: activeAppName,
        windowTitle: activeWindowTitle,
        displayInfo: displayInfo,
        tabInfo: tabInfo,
        meetingStatus: meetingStatus,
        screenSharingStatus: screenSharingStatus,
        sharedDisplay: sharedDisplay, // Include shared display info
        independentDetection: googleMeetResult // Include independent detection results
      };
    }
    
    return {
      success: false,
      error: 'Not implemented for this platform'
    };
  } catch (error) {
    safeLog.error('Failed to get focused app:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

// IPC handler for getting currently active/focused sources
ipcMain.handle('get-visible-sources', async (event, sourceIds) => {
  try {
    const { desktopCapturer } = require('electron');
    
    // Get the currently focused app first
    let focusedApp = null;
    if (process.platform === 'darwin') {
      try {
        const { exec } = require('child_process');
        const { promisify } = require('util');
        const execAsync = promisify(exec);
        
        const appScript = `osascript -e 'tell application "System Events" to get name of first application process whose frontmost is true'`;
        const appResult = await execAsync(appScript);
        
        if (appResult && appResult.stdout) {
          focusedApp = {
            appName: appResult.stdout.trim()
          };
        }
      } catch (error) {
        safeLog.warn('Could not get focused app:', error);
      }
    }
    
    if (sourceIds && sourceIds.length > 0) {
      safeLog.log('Checking active/focused apps for source IDs:', sourceIds);
      
      // Get currently active app and window title on macOS
      let activeAppName = null;
      let activeWindowTitle = null;
      if (process.platform === 'darwin') {
        try {
          const { exec } = require('child_process');
          const { promisify } = require('util');
          const execAsync = promisify(exec);
          
          // Get the frontmost application and its window title
          const appScript = `osascript -e 'tell application "System Events" to get name of first application process whose frontmost is true'`;
          const windowScript = `osascript -e '
            tell application "System Events"
              set frontApp to first application process whose frontmost is true
              try
                set windowTitle to title of first window of frontApp
                return windowTitle
              on error
                return ""
              end try
            end tell'`;
          
          const [appResult, windowResult] = await Promise.allSettled([
            execAsync(appScript),
            execAsync(windowScript)
          ]);
          
          activeAppName = appResult.status === 'fulfilled' ? appResult.value.stdout.trim().toLowerCase() : null;
          activeWindowTitle = windowResult.status === 'fulfilled' ? windowResult.value.stdout.trim() : null;
          
          safeLog.log(`Active app: "${activeAppName}", Window title: "${activeWindowTitle}"`);
        } catch (error) {
          safeLog.log('Could not get active app/window info:', error.message);
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
        
        if (id.startsWith('virtual-window:') || id.startsWith('virtual-browser-app:')) {
          let appNameMatch;
          if (id.startsWith('virtual-window:')) {
            appNameMatch = id.match(/virtual-window:\d+-(.+)$/);
            name = appNameMatch ? decodeURIComponent(appNameMatch[1]) : 'Unknown';
          } else if (id.startsWith('virtual-browser-app:')) {
            appNameMatch = id.match(/virtual-browser-app:(.+)$/);
            name = appNameMatch ? appNameMatch[1] : 'Unknown';
          }
          
          if (appNameMatch) {
            
            // Check if this app matches the active app
            if (activeAppName) {
              const appNameLower = name.toLowerCase();
              let isActive = false;
              
              // For virtual browser apps, we only check if browser is active and tab content matches
              if (id.startsWith('virtual-browser-app:')) {
                const service = name; // The service name from the ID
                
                // Check if a browser is currently active
                const isBrowserActive = activeAppName.includes('chrome') || 
                                       activeAppName.includes('safari') || 
                                       activeAppName.includes('firefox');
                
                if (isBrowserActive && activeWindowTitle) {
                  const windowTitleLower = activeWindowTitle.toLowerCase();
                  const servicePatterns = {
                    'zoom': ['zoom', 'meeting', 'webinar'],
                    'teams': ['teams', 'microsoft teams'],
                    'slack': ['slack'],
                    'notion': ['notion'],
                    'discord': ['discord'],
                    'figma': ['figma'],
                    'miro': ['miro'],
                    'github': ['github'],
                    'gmail': ['gmail', 'mail'],
                    'calendar': ['calendar', 'cal.com'],
                    'docs': ['docs', 'sheets', 'slides']
                  };
                  
                  // Check if the window title contains keywords for this service
                  const patterns = servicePatterns[service] || [service];
                  isActive = patterns.some(pattern => windowTitleLower.includes(pattern));
                  
                  if (isActive) {
                    safeLog.log(`Browser-based app is ACTIVE: ${service} found in "${activeWindowTitle}" (browser: ${activeAppName})`);
                  } else {
                    safeLog.log(`Browser-based app not active: ${service} not found in "${activeWindowTitle}" (browser: ${activeAppName})`);
                  }
                } else {
                  safeLog.log(`Browser-based app not active: no browser active or no window title (app: ${activeAppName})`);
                }
              } else {
                // For regular virtual windows, use the original logic
                const basicAppMatch = activeAppName.includes(appNameLower) || 
                                     appNameLower.includes(activeAppName) ||
                                     (appNameLower === 'msteams' && activeAppName.includes('teams')) ||
                                     (appNameLower === 'wechat' && (activeAppName.includes('wechat') || activeAppName.includes('weixin')));
                
                // For browsers or apps that support tabs/multiple windows, also check window title
                if (basicAppMatch) {
                  // If it's a browser, check if the tab content matches the selected app
                  if ((activeAppName.includes('chrome') || activeAppName.includes('safari') || activeAppName.includes('firefox')) && activeWindowTitle) {
                    // Check if the window title contains keywords that match our selected app
                    const windowTitleLower = activeWindowTitle.toLowerCase();
                    
                    // Define patterns for different services that might run in browser tabs
                    const servicePatterns = {
                      'zoom': ['zoom', 'meeting', 'webinar'],
                      'teams': ['teams', 'microsoft teams'],
                      'slack': ['slack'],
                      'notion': ['notion'],
                      'discord': ['discord'],
                      'figma': ['figma'],
                      'miro': ['miro'],
                      'github': ['github'],
                      'gmail': ['gmail', 'mail'],
                      'calendar': ['calendar', 'cal.com'],
                      'docs': ['docs', 'sheets', 'slides']
                    };
                    
                    // If the selected app name matches any service pattern, check the window title
                    let tabMatches = false;
                    for (const [service, patterns] of Object.entries(servicePatterns)) {
                      if (appNameLower.includes(service)) {
                        tabMatches = patterns.some(pattern => windowTitleLower.includes(pattern));
                        if (tabMatches) {
                          safeLog.log(`Tab content matches selected app: ${service} found in "${activeWindowTitle}"`);
                          break;
                        }
                      }
                    }
                    
                    // If we found a specific service match in the tab, that takes precedence
                    // If no specific match, fall back to general browser matching
                    isActive = tabMatches || (appNameLower.includes('chrome') && activeAppName.includes('chrome'));
                  } else {
                    // For non-browser apps, use the basic app matching
                    isActive = basicAppMatch;
                  }
                }
              } // end else block for regular virtual windows
              
              if (isActive) {
                isVisible = true;
                safeLog.log(`Virtual window is ACTIVE: ${id} -> ${name} (active app: ${activeAppName}, window: ${activeWindowTitle || 'N/A'})`);
              } else {
                safeLog.log(`Virtual window not active: ${id} -> ${name} (active app: ${activeAppName}, window: ${activeWindowTitle || 'N/A'})`);
              }
            } else {
              // Fallback: if we can't detect active app, assume visible (like before)
              isVisible = true;
              safeLog.log(`Virtual window assumed visible (no active app detection): ${id} -> ${name}`);
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
    
    // Handle virtual windows (minimized or on other spaces) and virtual browser apps
    if (sourceId.startsWith('virtual-window:') || sourceId.startsWith('virtual-browser-app:')) {
      // Extract app name from the source ID
      let appName = null;
      if (sourceId.startsWith('virtual-window:')) {
        const appNameMatch = sourceId.match(/virtual-window:\d+-(.+)$/);
        appName = appNameMatch ? decodeURIComponent(appNameMatch[1]) : null;
      } else if (sourceId.startsWith('virtual-browser-app:')) {
        const serviceMatch = sourceId.match(/virtual-browser-app:(.+)$/);
        appName = serviceMatch ? serviceMatch[1] : null;
      }
      
      // Declare matchingSource in the correct scope
      let matchingSource = null;
      
      // First, quickly check if the app might be visible on current desktop
      const quickSources = await desktopCapturer.getSources({
        types: ['window'],
        thumbnailSize: { width: 256, height: 144 }, // Small size for quick check
        fetchWindowIcons: false
      });
      
      // Debug: List ALL available windows when dealing with Zoom
      if (appName && appName.toLowerCase().includes('zoom')) {
        console.log(`\n[ZOOM DEBUG] ============ ALL AVAILABLE WINDOWS ============`);
        quickSources.forEach((source, index) => {
          console.log(`[ZOOM DEBUG] ${index + 1}. "${source.name}" (ID: ${source.id})`);
        });
        console.log(`[ZOOM DEBUG] ================================================\n`);
      }
      
      // Find all windows belonging to this app
      let appWindows = [];
      
      if (sourceId.startsWith('virtual-browser-app:')) {
        // For browser apps, find any active browser window
        appWindows = quickSources.filter(source => {
          const name = source.name.toLowerCase();
          return name.includes('chrome') || name.includes('safari') || name.includes('firefox');
        });
      } else {
        // For regular virtual windows, match by app name
        appWindows = quickSources.filter(source => {
          const name = source.name.toLowerCase();
          const appLower = appName.toLowerCase();
          
          // Special matching for different apps
          if (appLower.includes('zoom')) {
            // For Zoom, include ALL Zoom-related windows
            // Check both lowercase and original name for better matching
            const sourceName = source.name;
            const sourceNameLower = sourceName.toLowerCase();
            
            const isZoomWindow = sourceNameLower.includes('zoom') || 
                                 sourceName.includes('zoom.us') ||
                                 sourceName.includes('Zoom') ||
                                 sourceName.includes('Meeting') ||
                                 sourceName.includes('Workplace') ||
                                 // Additional patterns for meeting rooms
                                 sourceNameLower.includes('meeting') ||
                                 sourceNameLower.includes('webinar') ||
                                 sourceNameLower.includes('breakout') ||
                                 sourceNameLower.includes('share') ||
                                 sourceNameLower.includes('participants') ||
                                 sourceNameLower.includes('gallery') ||
                                 sourceNameLower.includes('speaker');
            
            if (isZoomWindow) {
              console.log(`[Dynamic Zoom] Found potential Zoom window: "${source.name}"`);
            }
            return isZoomWindow;
          } else if (appLower.includes('teams')) {
            return name.includes('teams') || name.includes('microsoft teams');
          } else if (appLower.includes('slack')) {
            return name.includes('slack');
          } else if (appLower.includes('powerpoint')) {
            return name.includes('powerpoint') || name.includes('ppt');
          } else if (appLower.includes('wechat')) {
            return name.includes('wechat') || name.includes('weixin');
          } else {
            return name.includes(appLower);
          }
        });
      }
      
      // Sort windows to prioritize the most relevant one (meetings, active windows)
      // IMPORTANT: This ensures we capture the meeting room instead of the homepage when both are available
      appWindows.sort((a, b) => {
        const aName = a.name.toLowerCase();
        const bName = b.name.toLowerCase();
        const appLower = appName ? appName.toLowerCase() : '';
        
        // For Zoom, prioritize meeting/webinar windows over the main window
        if (appLower.includes('zoom')) {
          // PRIORITY ORDER for Zoom:
          // 1. "Zoom Meeting" or any window with "meeting" in the name (highest priority - CAPTURES THE MEETING ROOM)
          // 2. "Zoom Webinar" or windows with "webinar"
          // 3. Windows with other meeting-related keywords (share, participants, gallery, speaker)
          // 4. "Zoom Workplace" (main app - CAPTURES THE HOMEPAGE)
          // 5. Other zoom windows
          
          // Check for exact matches and keywords
          const aExactMeeting = a.name === 'Zoom Meeting' || aName.includes('zoom meeting');
          const bExactMeeting = b.name === 'Zoom Meeting' || bName.includes('zoom meeting');
          const aExactWebinar = a.name === 'Zoom Webinar' || aName.includes('zoom webinar');
          const bExactWebinar = b.name === 'Zoom Webinar' || bName.includes('zoom webinar');
          
          // Broader meeting detection
          const aMeetingKeyword = aName.includes('meeting') || 
                                  aName.includes('webinar') ||
                                  aName.includes('share') ||
                                  aName.includes('participants') ||
                                  aName.includes('gallery') ||
                                  aName.includes('speaker') ||
                                  aName.includes('breakout');
          const bMeetingKeyword = bName.includes('meeting') || 
                                  bName.includes('webinar') ||
                                  bName.includes('share') ||
                                  bName.includes('participants') ||
                                  bName.includes('gallery') ||
                                  bName.includes('speaker') ||
                                  bName.includes('breakout');
          
          const aWorkplace = a.name === 'Zoom Workplace' || aName.includes('zoom workplace');
          const bWorkplace = b.name === 'Zoom Workplace' || bName.includes('zoom workplace');
          
          if (aExactMeeting && !bExactMeeting) return -1;
          if (!aExactMeeting && bExactMeeting) return 1;
          if (aExactWebinar && !bExactWebinar) return -1;
          if (!aExactWebinar && bExactWebinar) return 1;
          if (aMeetingKeyword && !bMeetingKeyword) return -1;
          if (!aMeetingKeyword && bMeetingKeyword) return 1;
          if (!aWorkplace && bWorkplace) return -1;
          if (aWorkplace && !bWorkplace) return 1;
          
          console.log(`[Dynamic Zoom] Priority check: "${a.name}" vs "${b.name}"`);
        }
        
        // For Teams, prioritize meeting windows
        if (appLower.includes('teams')) {
          if (aName.includes('meeting') && !bName.includes('meeting')) return -1;
          if (!aName.includes('meeting') && bName.includes('meeting')) return 1;
        }
        
        return 0;
      });
      
      // Use the top/most relevant window
      let quickMatch = appWindows.length > 0 ? appWindows[0] : null;
      
      if (appWindows.length > 0) {
        safeLog.log(`[Dynamic Zoom Selection] Found ${appWindows.length} window(s) for ${appName}:`);
        appWindows.forEach((w, i) => {
          safeLog.log(`  ${i === 0 ? '🎯 SELECTED' : '  -'} "${w.name}" (ID: ${w.id})`);
        });
        if (quickMatch) {
          const qName = quickMatch.name;
          const qNameLower = qName.toLowerCase();
          
          if (qName === 'Zoom Meeting' || qNameLower.includes('zoom meeting')) {
            safeLog.log(`[Dynamic Zoom Selection] 🎥 ✅ CAPTURING ZOOM MEETING ROOM - User is actively in a meeting!`);
            safeLog.log(`[Dynamic Zoom Selection]    Window: "${qName}"`);
            safeLog.log(`[Dynamic Zoom Selection]    This will capture the video conference window with participants`);
          } else if (qNameLower.includes('meeting') || qNameLower.includes('share') || 
                     qNameLower.includes('participants') || qNameLower.includes('gallery')) {
            safeLog.log(`[Dynamic Zoom Selection] 🎥 ✅ CAPTURING MEETING-RELATED WINDOW`);
            safeLog.log(`[Dynamic Zoom Selection]    Window: "${qName}"`);
            safeLog.log(`[Dynamic Zoom Selection]    This appears to be a meeting window based on its name`);
          } else if (qName === 'Zoom Workplace' || qNameLower.includes('zoom workplace')) {
            safeLog.log(`[Dynamic Zoom Selection] 🏠 CAPTURING ZOOM WORKPLACE - User is on the Zoom homepage`);
            safeLog.log(`[Dynamic Zoom Selection]    Window: "${qName}"`);
            safeLog.log(`[Dynamic Zoom Selection]    This will capture the main Zoom interface (not a meeting)`);
          } else if (qNameLower.includes('webinar')) {
            safeLog.log(`[Dynamic Zoom Selection] 🎤 CAPTURING ZOOM WEBINAR - User is in a webinar`);
            safeLog.log(`[Dynamic Zoom Selection]    Window: "${qName}"`);
          } else {
            safeLog.log(`[Dynamic Zoom Selection] 📷 Capturing Zoom window: "${qName}"`);
            safeLog.log(`[Dynamic Zoom Selection]    Note: Could not determine if this is a meeting or homepage`);
          }
        }
      } else {
        safeLog.log(`[Dynamic Zoom Selection] No windows found for ${appName}`);
      }
      
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
      
      let source = sources.find(s => s.id === sourceId);
      
      // If exact source not found, try to find the top window for the app
      if (!source) {
        // Extract app name from the sourceId if available
        const appNameMatch = sourceId.match(/window:\d+:(.+)/);
        if (appNameMatch) {
          const targetAppName = appNameMatch[1].toLowerCase();
          
          // Find all windows for this app
          const appWindows = sources.filter(s => {
            if (!s.name) return false;
            const sourceName = s.name.toLowerCase();
            
            // Special matching for different apps
            if (targetAppName.includes('zoom')) {
              // For Zoom, prioritize meeting/webinar windows
              return sourceName.includes('zoom meeting') || 
                     sourceName.includes('zoom webinar') ||
                     sourceName.includes('zoom');
            } else if (targetAppName.includes('teams')) {
              return sourceName.includes('teams') || sourceName.includes('microsoft teams');
            } else if (targetAppName.includes('slack')) {
              return sourceName.includes('slack');
            } else {
              return sourceName.includes(targetAppName);
            }
          });
          
          // Sort to prioritize the most relevant window
          appWindows.sort((a, b) => {
            const aName = a.name.toLowerCase();
            const bName = b.name.toLowerCase();
            
            // For Zoom, prioritize meeting/webinar windows
            if (targetAppName.includes('zoom')) {
              if (aName.includes('meeting') && !bName.includes('meeting')) return -1;
              if (!aName.includes('meeting') && bName.includes('meeting')) return 1;
              if (aName.includes('webinar') && !bName.includes('webinar')) return -1;
              if (!aName.includes('webinar') && bName.includes('webinar')) return 1;
            }
            
            // For Teams, prioritize meeting windows
            if (targetAppName.includes('teams')) {
              if (aName.includes('meeting') && !bName.includes('meeting')) return -1;
              if (!aName.includes('meeting') && bName.includes('meeting')) return 1;
            }
            
            return 0;
          });
          
          if (appWindows.length > 0) {
            source = appWindows[0];
            safeLog.log(`[Screenshot] Using top window for ${targetAppName}: ${source.name}`);
          }
        }
      }
      
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

// IPC handler for getting the currently focused window with detailed tab information
ipcMain.handle('get-focused-window-info', async () => {
  try {
    if (process.platform === 'darwin') {
      const { exec } = require('child_process');
      const { promisify } = require('util');
      const execAsync = promisify(exec);
      
      // Get the frontmost application and its window title
      const appScript = `osascript -e 'tell application "System Events" to get name of first application process whose frontmost is true'`;
      const windowScript = `osascript -e '
        tell application "System Events"
          set frontApp to first application process whose frontmost is true
          try
            set windowTitle to title of first window of frontApp
            return windowTitle
          on error
            return ""
          end try
        end tell'`;
      
      const [appResult, windowResult] = await Promise.allSettled([
        execAsync(appScript),
        execAsync(windowScript)
      ]);
      
      const activeApp = appResult.status === 'fulfilled' ? appResult.value.stdout.trim() : '';
      const activeWindowTitle = windowResult.status === 'fulfilled' ? windowResult.value.stdout.trim() : '';
      
      safeLog.log(`Focused window: App="${activeApp}", Title="${activeWindowTitle}"`);
      
      return {
        success: true,
        activeApp: activeApp,
        activeWindowTitle: activeWindowTitle
      };
    } else {
      // For non-macOS platforms, return basic info
      return {
        success: true,
        activeApp: '',
        activeWindowTitle: ''
      };
    }
  } catch (error) {
    safeLog.error('Error getting focused window info:', error);
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