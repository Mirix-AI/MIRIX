# MIRIX Desktop Application

A modern React + Electron desktop application for the MMIRIXirix AI assistant, replacing the PyQt interface with a web-based UI that can be packaged as a native desktop app.

## Features

- 🚀 Modern React-based UI with real-time streaming
- 💬 Chat interface with markdown and code syntax highlighting
- 📎 File attachment support (images)
- ⚙️ Settings panel for model, persona, and timezone configuration
- 🖥️ Cross-platform desktop app (Windows, macOS, Linux)
- 🔄 Real-time streaming responses from the backend
- �� Responsive design
- 🔧 **Bundled backend** - No separate Python server needed in production

## Prerequisites

- Node.js 16+ and npm
- Python 3.8+ (for building the backend executable)
- PyInstaller (automatically installed during build)

## Quick Start

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Development Mode

For web development:
```bash
npm start
```

For Electron development (recommended):
```bash
npm run electron-dev
```

**Note**: In development mode, you still need to run the Python backend separately:
```bash
# In the main project directory
python server.py
```

### 3. Building for Production

The production build automatically bundles the Python backend:

```bash
npm run electron-pack
```

This will:
1. Build the React frontend
2. Package the Python backend into an executable
3. Create platform-specific installers with everything bundled

## Database Storage

The application stores its database in the user's home directory:
- **Windows**: `C:\Users\{username}\.mirix\sqlite.db`
- **macOS**: `/Users/{username}/.mirix/sqlite.db`
- **Linux**: `/home/{username}/.mirix/sqlite.db`

This location is consistent whether running in development or as a packaged app.

## Distribution Options

### Option 1: Bundled App (Recommended)
- **Pros**: Single installer, no Python dependencies for users
- **Cons**: Larger file size (~100-200MB)
- **Use case**: End-user distribution

### Option 2: Separate Backend
- **Pros**: Smaller frontend, easier development
- **Cons**: Users need Python environment
- **Use case**: Developer/power user distribution

## Building for Different Platforms

### Windows
```bash
npm run electron-pack
# Creates: dist/MIRIX Setup 0.1.0.exe (~150MB)
```

### macOS
```bash
npm run electron-pack
# Creates: dist/MIRIX-0.1.0.dmg (~120MB)
```

### Linux
```bash
npm run electron-pack
# Creates: dist/MIRIX-0.1.0.AppImage (~130MB)
```

## Project Structure

```
frontend/
├── public/
│   ├── electron.js          # Main Electron process (with backend management)
│   ├── preload.js          # Secure IPC bridge
│   └── index.html          # HTML template
├── src/
│   ├── components/
│   │   ├── ChatWindow.js    # Main chat interface
│   │   ├── ChatBubble.js    # Message bubbles
│   │   ├── MessageInput.js  # Input with file support
│   │   └── SettingsPanel.js # Configuration panel
│   ├── App.js              # Main app component
│   └── index.js            # React entry point
├── scripts/
│   └── build-backend.js    # Backend packaging script
├── backend/                # Generated during build
│   ├── server.exe          # Bundled Python backend (Windows)
│   └── ...                 # Other backend files
└── package.json            # Dependencies and scripts
```

## Configuration

The app can be configured through the Settings panel:

- **AI Model**: Choose between different language models
- **Persona**: Set the assistant's personality
- **Timezone**: Configure local timezone for timestamps
- **Server URL**: Backend API endpoint (auto-configured in production)

## Backend Integration

### Development Mode
- Frontend connects to `http://localhost:47283`
- Python backend runs separately
- Hot reload for both frontend and backend

### Production Mode
- Electron automatically starts bundled backend
- Backend runs on random available port
- Frontend auto-connects to bundled backend
- Backend stops when app closes

## Development

### Available Scripts

- `npm start` - Start React development server
- `npm run electron-dev` - Start Electron in development mode
- `npm run build` - Build React app for production
- `npm run build-backend` - Package Python backend into executable
- `npm run electron` - Run Electron with built app
- `npm run electron-pack` - Package complete desktop app

### Backend Building Process

The build process automatically:
1. Copies Python source files to `frontend/backend/`
2. Installs PyInstaller if not available
3. Creates a single executable with all dependencies
4. Includes the executable in the Electron package
5. Cleans up temporary build files

### Troubleshooting Backend Build

If the backend build fails:

1. **Install PyInstaller manually**:
   ```bash
   pip install pyinstaller
   ```

2. **Build backend manually**:
   ```bash
   cd frontend/backend
   pyinstaller --onefile server.py
   ```

3. **Check Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Customization

### Adding Python Dependencies
1. Add to main project's `requirements.txt`
2. Update `frontend/scripts/build-backend.js` hidden imports if needed
3. Rebuild with `npm run build-backend`

### Modifying Backend Startup
Edit `frontend/public/electron.js` in the `startBackendServer()` function.

## Security

The app follows Electron security best practices:
- Context isolation enabled
- Node integration disabled in renderer
- Secure IPC communication via preload script
- External links open in default browser
- Backend process managed securely by main process

## Performance

### File Sizes (Approximate)
- **Development**: Frontend only (~50MB)
- **Production Windows**: ~150MB (includes Python runtime)
- **Production macOS**: ~120MB
- **Production Linux**: ~130MB

### Startup Time
- **Development**: ~2-3 seconds
- **Production**: ~3-5 seconds (includes backend startup)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test in both development and production modes
5. Submit a pull request

## License

This project is part of the MIRIX AI assistant suite. 