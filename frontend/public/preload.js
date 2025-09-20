const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  selectFiles: () => ipcRenderer.invoke('select-files'),
  selectSavePath: (options) => ipcRenderer.invoke('select-save-path', options),
  

  
  // Screenshot functions
  takeScreenshot: (options) => ipcRenderer.invoke('take-screenshot', options),
  takeScreenshotDisplay: (displayId) => ipcRenderer.invoke('take-screenshot-display', displayId),
  listDisplays: () => ipcRenderer.invoke('list-displays'),
  cleanupScreenshots: (maxAge) => ipcRenderer.invoke('cleanup-screenshots', maxAge),
  checkScreenPermission: () => ipcRenderer.invoke('check-screen-permission'),
  openScreenRecordingPrefs: () => ipcRenderer.invoke('open-screen-recording-prefs'),
  
  // Window/App capture functions
  getCaptureSources: () => ipcRenderer.invoke('get-capture-sources'),
  getVisibleSources: (sourceIds) => ipcRenderer.invoke('get-visible-sources', sourceIds),
  takeSourceScreenshot: (sourceId) => ipcRenderer.invoke('take-source-screenshot', sourceId),
  
  // Image reading function for similarity comparison
  readImageAsBase64: (filepath) => ipcRenderer.invoke('read-image-base64', filepath),
  
  // Delete screenshot function (for removing similar screenshots)
  deleteScreenshot: (filepath) => ipcRenderer.invoke('delete-screenshot', filepath),
  
  // Debug comparison image saving
  saveDebugComparisonImage: (imageData, filename) => ipcRenderer.invoke('save-debug-comparison-image', imageData, filename),
  
  // Image saving functions
  saveImageToTmp: (sourcePath, filename) => ipcRenderer.invoke('save-image-to-tmp', sourcePath, filename),
  saveImageBufferToTmp: (arrayBuffer, filename) => ipcRenderer.invoke('save-image-buffer-to-tmp', arrayBuffer, filename),
  cleanupTmpImages: (maxAge) => ipcRenderer.invoke('cleanup-tmp-images', maxAge),
  
  // Menu event listeners - wrap callbacks to prevent passing non-serializable event objects
  onMenuNewChat: (callback) => {
    const wrappedCallback = (event, ...args) => callback(...args);
    ipcRenderer.on('menu-new-chat', wrappedCallback);
    return () => ipcRenderer.removeListener('menu-new-chat', wrappedCallback);
  },
  onMenuOpenTerminal: (callback) => {
    const wrappedCallback = (event, ...args) => callback(...args);
    ipcRenderer.on('menu-open-terminal', wrappedCallback);
    return () => ipcRenderer.removeListener('menu-open-terminal', wrappedCallback);
  },
  
  // Window event listeners
  onWindowShow: (callback) => {
    const wrappedCallback = (event, ...args) => callback(...args);
    ipcRenderer.on('window-show', wrappedCallback);
    return () => ipcRenderer.removeListener('window-show', wrappedCallback);
  },
  onAppActivate: (callback) => {
    const wrappedCallback = (event, ...args) => callback(...args);
    ipcRenderer.on('app-activate', wrappedCallback);
    return () => ipcRenderer.removeListener('app-activate', wrappedCallback);
  },
  onMenuTakeScreenshot: (callback) => {
    const wrappedCallback = (event, ...args) => callback(...args);
    ipcRenderer.on('menu-take-screenshot', wrappedCallback);
    return () => ipcRenderer.removeListener('menu-take-screenshot', wrappedCallback);
  },
  
  // Remove listeners
  removeAllListeners: (channel) => ipcRenderer.removeAllListeners(channel),
  
  // Platform info
  platform: process.platform
}); 