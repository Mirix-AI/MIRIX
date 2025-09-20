const fs = require('fs');
const path = require('path');

console.log('📦 Copying pre-built backend executable...');

// Setup paths
const backendDir = path.join(__dirname, '..', 'backend');
const sourceDir = path.join(__dirname, '..', '..');
const executableName = 'main';
const sourceExecutablePath = path.join(sourceDir, 'dist', executableName);
const destExecutablePath = path.join(backendDir, executableName);

// Ensure backend directory exists
if (!fs.existsSync(backendDir)) {
  fs.mkdirSync(backendDir, { recursive: true });
}

// Check if pre-built executable exists
if (!fs.existsSync(sourceExecutablePath)) {
  console.error(`❌ Pre-built executable not found at: ${sourceExecutablePath}`);
  console.log('');
  console.log('🔧 Please build the backend first:');
  console.log('1. Activate your environment: source mirix_env/bin/activate');
  console.log('2. Build executable: pyinstaller main.spec --clean');
  console.log('');
  process.exit(1);
}

// Copy the executable
fs.copyFileSync(sourceExecutablePath, destExecutablePath);

// Make executable on Unix systems
if (process.platform !== 'win32') {
  const { execSync } = require('child_process');
  execSync(`chmod +x "${destExecutablePath}"`);
}

// Get file size
const stats = fs.statSync(destExecutablePath);
const fileSizeInMB = (stats.size / (1024 * 1024)).toFixed(2);
console.log(`✅ Copied executable (${fileSizeInMB} MB)`);

// Copy config files
const configDir = path.join(backendDir, 'configs');
const sourceConfigDir = path.join(sourceDir, 'configs');

if (fs.existsSync(sourceConfigDir)) {
  // Create config directory
  if (!fs.existsSync(configDir)) {
    fs.mkdirSync(configDir, { recursive: true });
  }
  
  // Copy all config files
  const configFiles = fs.readdirSync(sourceConfigDir);
  configFiles.forEach(file => {
    const src = path.join(sourceConfigDir, file);
    const dest = path.join(configDir, file);
    fs.copyFileSync(src, dest);
  });
  console.log(`✅ Copied ${configFiles.length} config files`);
} else {
  console.log('⚠️  Warning: No config files found');
}

console.log('🎉 Backend ready for packaging'); 