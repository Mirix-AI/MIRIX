const { exec } = require('child_process');

// Function to get all running applications using System Events
async function getRunningApplications() {
  return new Promise((resolve, reject) => {
    const script = 'tell application "System Events" to get name of every application process whose visible is true';
    
    exec(`osascript -e '${script}'`, (error, stdout, stderr) => {
      if (error) {
        reject(error);
        return;
      }
      
      try {
        const apps = stdout.trim().split(', ').map(app => app.trim());
        resolve(apps);
      } catch (parseError) {
        reject(parseError);
      }
    });
  });
}

// Function to get windows for a specific application
async function getWindowsForApp(appName) {
  return new Promise((resolve) => {
    const script = `tell application "System Events" to tell application process "${appName}" to get title of every window`;
    
    exec(`osascript -e '${script}'`, (error, stdout, stderr) => {
      if (error) {
        resolve([]);
        return;
      }
      
      try {
        if (!stdout.trim()) {
          resolve([]);
          return;
        }
        
        const windows = stdout.trim().split(', ').map(title => title.trim()).filter(title => title !== '');
        resolve(windows);
      } catch (parseError) {
        resolve([]);
      }
    });
  });
}

// Function to get actual window information with real IDs using Python script
async function getWindowsWithRealIds() {
  return new Promise((resolve, reject) => {
    // Create a temporary Python script that uses Quartz to get window information
    const pythonScript = `
import sys
try:
    from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionAll, kCGNullWindowID
    import json
    
    # Get all windows including off-screen ones and windows on other spaces
    window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionAll, kCGNullWindowID)
    
    windows = []
    important_apps = ['zoom.us', 'Zoom', 'Microsoft PowerPoint', 'Notion', 'Slack', 
                     'Microsoft Teams', 'MSTeams', 'Teams', 'Discord', 'Google Chrome',
                     'Microsoft Word', 'Microsoft Excel', 'Keynote', 'Figma',
                     'Sketch', 'Adobe Photoshop', 'Visual Studio Code', 'Cursor',
                     'Safari', 'Firefox', 'WeChat', 'Obsidian', 'Chrome']
    
    # Group windows by app to get the best representative window
    app_windows = {}
    
    for window in window_list:
        if window.get('kCGWindowOwnerName') and window.get('kCGWindowNumber'):
            app_name = window['kCGWindowOwnerName']
            
            # Skip system apps but be less restrictive
            if app_name in ['SystemUIServer', 'Dock', 'ControlCenter', 'WindowManager', 'MIRIX', 'Electron', 'Finder']:
                continue
            
            # Get bounds - be more permissive with size for cross-space windows
            bounds = window.get('kCGWindowBounds', {})
            width = bounds.get('Width', 0)
            height = bounds.get('Height', 0)
            
            # Very small windows are likely not main windows
            if width < 50 or height < 50:
                continue
            
            # Be more permissive with layers - apps on other spaces might have higher layers
            layer = window.get('kCGWindowLayer', 0)
            if layer > 200:  # Only exclude very high system layers
                continue
            
            # Check if this is an important app or has meaningful content
            is_important = any(app.lower() in app_name.lower() or app_name.lower() in app.lower() for app in important_apps)
            has_content = window.get('kCGWindowName', '').strip() != ''
            window_title = window.get('kCGWindowName', '')
            
            # Include if important app OR has substantial content OR reasonable size
            should_include = (
                is_important or 
                has_content or 
                (width > 300 and height > 200)  # Reasonable size suggests main window
            )
            
            if not should_include:
                continue
            
            window_info = {
                'windowId': window['kCGWindowNumber'],
                'appName': app_name,
                'windowTitle': window_title,
                'bounds': bounds,
                'isOnScreen': window.get('kCGWindowIsOnscreen', False),
                'layer': layer,
                'isImportant': is_important,
                'area': width * height
            }
            
            # Group by app name to find best representative window
            if app_name not in app_windows:
                app_windows[app_name] = []
            app_windows[app_name].append(window_info)
    
    # Select best window for each app
    for app_name, app_window_list in app_windows.items():
        if not app_window_list:
            continue
            
        # Sort windows by preference: important apps first, then by area, then by layer
        def window_score(w):
            return (
                w['isImportant'],           # Important apps first
                w['area'],                  # Larger windows preferred
                -w['layer'],                # Lower layers preferred (negative for descending)
                bool(w['windowTitle'])      # Windows with titles preferred
            )
        
        best_window = max(app_window_list, key=window_score)
        windows.append(best_window)
    
    # Sort final list by importance and then by app name
    windows.sort(key=lambda x: (not x['isImportant'], x['appName']))
    
    print(json.dumps(windows))
    
except ImportError:
    # Fallback if Quartz is not available
    print("[]")
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    print("[]")
`;

    // Write the Python script to a temporary file
    const fs = require('fs');
    const os = require('os');
    const path = require('path');
    
    const tempFile = path.join(os.tmpdir(), 'get_windows.py');
    fs.writeFileSync(tempFile, pythonScript);
    
    exec(`python3 "${tempFile}"`, (error, stdout, stderr) => {
      // Clean up temp file
      try {
        fs.unlinkSync(tempFile);
      } catch (e) {}
      
      if (error) {
        resolve([]);
        return;
      }
      
      try {
        const windows = JSON.parse(stdout);
        resolve(windows);
      } catch (parseError) {
        resolve([]);
      }
    });
  });
}

// Function to get all windows using the best available method
async function getAllWindows() {
  try {
    // First try to get windows with real IDs using Python/Quartz
    let windowsWithIds = await getWindowsWithRealIds();
    
    if (windowsWithIds.length > 0) {
      
      // Filter and process the windows
      const allWindows = [];
      const importantApps = [
        'zoom.us', 'Zoom', 'Microsoft PowerPoint', 'Notion', 'Slack', 
        'Microsoft Teams', 'MSTeams', 'Teams', 'Discord', 'Google Chrome',
        'Microsoft Word', 'Microsoft Excel', 'Keynote', 'Figma',
        'Sketch', 'Adobe Photoshop', 'Visual Studio Code', 'Cursor',
        'Safari', 'Firefox', 'WeChat', 'Obsidian', 'Roam Research'
      ];
      
      for (const window of windowsWithIds) {
        const appName = window.appName;
        
        // Skip system apps and our own app
        if (appName === 'MIRIX' || appName === 'Electron' || 
            appName === 'SystemUIServer' || appName === 'Dock' ||
            appName === 'ControlCenter' || appName === 'WindowManager' ||
            appName === 'NotificationCenter' || appName === 'Spotlight') {
          continue;
        }
        
        const isImportant = importantApps.some(app => 
          appName.toLowerCase().includes(app.toLowerCase()) ||
          app.toLowerCase().includes(appName.toLowerCase())
        );
        
        // Include windows that have titles or are from important apps
        if (window.windowTitle || isImportant) {
          let finalTitle = window.windowTitle;
          
          if (!finalTitle || finalTitle.trim() === '') {
            if (appName.includes('zoom')) finalTitle = 'Zoom Meeting';
            else if (appName.includes('PowerPoint')) finalTitle = 'PowerPoint Presentation';
            else if (appName.includes('Notion')) finalTitle = 'Notion Workspace';
            else if (appName.includes('Slack')) finalTitle = 'Slack Workspace';
            else if (appName.includes('Teams')) finalTitle = 'Teams Meeting';
            else finalTitle = appName + ' Window';
          }
          
          // Special handling for Cursor window titles
          if (appName === 'Cursor' && finalTitle && finalTitle.includes(' - ')) {
            // Cursor window titles are in format "filename - ProjectName"
            // We want to extract just the project name
            const parts = finalTitle.split(' - ');
            if (parts.length >= 2) {
              // Take the last part as the project name
              finalTitle = parts[parts.length - 1];
            }
          }
          
          allWindows.push({
            windowId: window.windowId, // Real window ID from Core Graphics
            appName: appName,
            windowTitle: finalTitle,
            isOnScreen: window.isOnScreen,
            bounds: window.bounds,
            isImportantApp: isImportant,
            layer: window.layer
          });
        }
      }
      
      // Sort to put important apps first
      allWindows.sort((a, b) => {
        if (a.isImportantApp && !b.isImportantApp) return -1;
        if (!a.isImportantApp && b.isImportantApp) return 1;
        return a.appName.localeCompare(b.appName);
      });
      
      return allWindows;
    }
    
    // Fallback to the original method if Python approach fails
    const runningApps = await getRunningApplications();
    const allWindows = [];
    
    const importantApps = [
      'zoom.us', 'Zoom', 'Microsoft PowerPoint', 'Notion', 'Slack', 
      'Microsoft Teams', 'MSTeams', 'Teams', 'Discord', 'Google Chrome',
      'Microsoft Word', 'Microsoft Excel', 'Keynote', 'Figma',
      'Sketch', 'Adobe Photoshop', 'Visual Studio Code', 'Cursor',
      'Safari', 'Firefox', 'WeChat', 'Obsidian', 'Roam Research'
    ];
    
    for (const appName of runningApps) {
      if (appName === 'MIRIX' || appName === 'Electron' || 
          appName === 'SystemUIServer' || appName === 'Dock' ||
          appName === 'ControlCenter' || appName === 'WindowManager' ||
          appName === 'NotificationCenter' || appName === 'Spotlight') {
        continue;
      }
      
      const isImportant = importantApps.some(app => 
        appName.toLowerCase().includes(app.toLowerCase()) ||
        app.toLowerCase().includes(appName.toLowerCase())
      );
      
      if (isImportant) {
        let defaultTitle;
        if (appName.includes('zoom')) defaultTitle = 'Zoom Meeting';
        else if (appName.includes('PowerPoint')) defaultTitle = 'PowerPoint Presentation';
        else if (appName.includes('Notion')) defaultTitle = 'Notion Workspace';
        else if (appName.includes('Slack')) defaultTitle = 'Slack Workspace';
        else if (appName.includes('Teams')) defaultTitle = 'Teams Meeting';
        else defaultTitle = appName + ' Window';
        
        allWindows.push({
          windowId: Math.floor(Math.random() * 1000000),
          appName: appName,
          windowTitle: defaultTitle,
          isOnScreen: false,
          isImportantApp: true
        });
      }
    }
    
    return allWindows;
    
  } catch (error) {
    console.error('Error getting windows:', error);
    return [];
  }
}

// Function to capture window using different methods - DEPRECATED
// This function is no longer used as we've switched to desktopCapturer
async function captureWindowById(windowId, appName = null) {
  // Always reject since we no longer use this method
  throw new Error('captureWindowById is deprecated - use desktopCapturer instead');
}

// Function to get app icon (simplified)
async function getAppIcon(appName, bundleId) {
  return null;
}

module.exports = {
  getAllWindows,
  captureWindowById,
  getAppIcon
};