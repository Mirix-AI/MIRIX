const fs = require('fs');
const path = require('path');
const os = require('os');

class NativeCaptureHelper {
  constructor() {
    this.isRunning = false;
    this.screenshots = null;
  }

  async initialize() {
    console.log('Initializing Native Capture Helper (Pure JavaScript)...');
    
    try {
      // Import node-screenshots dynamically
      this.screenshots = require('node-screenshots');
      
      // Test that the module works
      const monitors = this.screenshots.Monitor.all();
      console.log(`[Native Helper] Found ${monitors.length} monitor(s)`);
      
      this.isRunning = true;
      console.log('✅ Native Capture Helper initialized successfully (Python-free!)');
    } catch (error) {
      throw new Error(`Native capture helper failed to initialize: ${error.message}`);
    }
  }

    // High-level methods using pure JavaScript

  async getAllWindows() {
    if (!this.isRunning) {
      throw new Error('Helper not initialized');
    }

    try {
      // Use Electron's desktopCapturer to get available windows (visible ones)
      // For now, we'll focus on screen capture rather than individual windows
      // since node-screenshots doesn't have window enumeration capabilities
      
      // This is a simplified implementation - for full window management,
      // we'd need a native addon or different approach
      console.log('[Native Helper] Using simplified window detection (visible windows only)');
      
      // Return empty for now - the main focus is screen capture
      // Individual window capture will fall back to desktopCapturer in electron.js
      return [];
    } catch (error) {
      console.error('Failed to get windows from native helper:', error);
      return [];
    }
  }

  async captureWindow(windowId) {
    if (!this.isRunning) {
      throw new Error('Helper not initialized');
    }

    try {
      console.log(`[Native Helper] Pure JS window capture not supported for windowId ${windowId}`);
      
      // For pure JavaScript solution, we can't capture specific windows by ID
      // This will gracefully fail and let the caller fall back to other methods
      return {
        success: false,
        error: 'Pure JavaScript helper does not support individual window capture by ID'
      };
    } catch (error) {
      console.error(`Failed to capture window ${windowId}:`, error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  async captureScreen(monitorIndex = 0) {
    if (!this.isRunning) {
      throw new Error('Helper not initialized');
    }

    try {
      console.log(`[Native Helper] Capturing screen ${monitorIndex} using node-screenshots`);
      
      const monitors = this.screenshots.Monitor.all();
      if (monitorIndex >= monitors.length) {
        return {
          success: false,
          error: `Monitor ${monitorIndex} not found. Available monitors: ${monitors.length}`
        };
      }

      const monitor = monitors[monitorIndex];
      const image = monitor.captureImageSync();
      const pngBuffer = image.toPngSync();
      
      console.log(`[Native Helper] Screen capture successful, size: ${pngBuffer.length} bytes`);
      
      return {
        success: true,
        data: pngBuffer,
        size: pngBuffer.length
      };
    } catch (error) {
      console.error(`Failed to capture screen ${monitorIndex}:`, error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  async captureApp(appName) {
    if (!this.isRunning) {
      throw new Error('Helper not initialized');
    }

    try {
      console.log(`[Native Helper] Pure JS app capture for: ${appName}`);
      console.log(`[Native Helper] Note: Individual app capture not supported, falling back to screen capture`);
      
      // Since we can't capture individual apps with node-screenshots,
      // we'll capture the primary screen as a fallback
      // The main Electron code will handle specific window capture via desktopCapturer
      
      return await this.captureScreen(0); // Capture primary monitor
    } catch (error) {
      console.error(`Failed to capture app ${appName}:`, error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  async shutdown() {
    if (this.isRunning) {
      console.log('Shutting down native capture helper');
      this.isRunning = false;
    }
  }
}

module.exports = NativeCaptureHelper;