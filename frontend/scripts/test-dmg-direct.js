#!/usr/bin/env node

const { execSync, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

const DMG_PATH = path.join(__dirname, '..', 'dist', 'MIRIX-0.1.0-arm64.dmg');
const MOUNT_POINT = '/Volumes/MIRIX 0.1.0-arm64';
const DEBUG_LOG_DIR = path.join(os.homedir(), '.mirix', 'debug');

// Create debug log directory
if (!fs.existsSync(DEBUG_LOG_DIR)) {
    fs.mkdirSync(DEBUG_LOG_DIR, { recursive: true });
}

const DEBUG_LOG_FILE = path.join(DEBUG_LOG_DIR, `dmg-test-${Date.now()}.log`);

function log(message) {
    const timestamp = new Date().toISOString();
    const logMessage = `${timestamp}: ${message}`;
    console.log(logMessage);
    try {
        fs.appendFileSync(DEBUG_LOG_FILE, logMessage + '\n');
    } catch (error) {
        console.error('Failed to write to debug log:', error);
    }
}

function cleanup() {
    try {
        log('🧹 Cleaning up - unmounting DMG...');
        execSync(`hdiutil detach "${MOUNT_POINT}"`, { stdio: 'ignore' });
        log('✅ DMG unmounted successfully');
    } catch (error) {
        log(`⚠️  Failed to unmount DMG: ${error.message}`);
    }
}

// Handle cleanup on exit
process.on('exit', cleanup);
process.on('SIGINT', () => {
    cleanup();
    process.exit(0);
});
process.on('SIGTERM', () => {
    cleanup();
    process.exit(0);
});

async function main() {
    try {
        log('🔍 DMG Direct Test Started');
        log(`📁 DMG Path: ${DMG_PATH}`);
        log(`🗂️  Debug Log: ${DEBUG_LOG_FILE}`);
        
        // Check if DMG exists
        if (!fs.existsSync(DMG_PATH)) {
            throw new Error(`DMG file not found: ${DMG_PATH}`);
        }
        
        const dmgStats = fs.statSync(DMG_PATH);
        log(`📦 DMG Size: ${(dmgStats.size / 1024 / 1024).toFixed(2)} MB`);
        
        // Mount the DMG
        log('🔧 Mounting DMG...');
        try {
            execSync(`hdiutil attach "${DMG_PATH}" -nobrowse -quiet`, { stdio: 'pipe' });
            log('✅ DMG mounted successfully');
        } catch (error) {
            throw new Error(`Failed to mount DMG: ${error.message}`);
        }
        
        // Wait a moment for mount to complete
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Check if mount point exists
        if (!fs.existsSync(MOUNT_POINT)) {
            throw new Error(`Mount point not found: ${MOUNT_POINT}`);
        }
        
        // List contents of mounted DMG
        log('📂 DMG Contents:');
        const contents = fs.readdirSync(MOUNT_POINT);
        contents.forEach(item => {
            const itemPath = path.join(MOUNT_POINT, item);
            const stats = fs.statSync(itemPath);
            log(`  - ${item} (${stats.isDirectory() ? 'directory' : 'file'})`);
        });
        
        // Find the .app bundle
        const appName = 'MIRIX.app';
        const appPath = path.join(MOUNT_POINT, appName);
        
        if (!fs.existsSync(appPath)) {
            throw new Error(`App bundle not found: ${appPath}`);
        }
        
        log(`🎯 Found app bundle: ${appPath}`);
        
        // Find the backend executable
        const possibleBackendPaths = [
            path.join(appPath, 'Contents', 'Resources', 'backend', 'main'),
            path.join(appPath, 'Contents', 'Resources', 'app', 'backend', 'main'),
            path.join(appPath, 'Contents', 'Resources', 'app.asar.unpacked', 'backend', 'main')
        ];
        
        let backendPath = null;
        for (const candidatePath of possibleBackendPaths) {
            log(`🔍 Checking backend path: ${candidatePath}`);
            if (fs.existsSync(candidatePath)) {
                const stats = fs.statSync(candidatePath);
                log(`  ✅ Found! Size: ${(stats.size / 1024 / 1024).toFixed(2)} MB`);
                log(`  File mode: ${stats.mode.toString(8)}`);
                log(`  Executable: ${(stats.mode & parseInt('111', 8)) !== 0}`);
                backendPath = candidatePath;
                break;
            } else {
                log(`  ❌ Not found`);
            }
        }
        
        if (!backendPath) {
            throw new Error('Backend executable not found in app bundle');
        }
        
        // Test backend execution
        log('🚀 Testing backend execution...');
        
                 // Create test working directory
         const testWorkingDir = path.join(os.homedir(), '.mirix-test');
         if (!fs.existsSync(testWorkingDir)) {
             fs.mkdirSync(testWorkingDir, { recursive: true });
         }
         
         // Copy config files to test working directory
         const configsDir = path.join(testWorkingDir, 'configs');
         if (!fs.existsSync(configsDir)) {
             fs.mkdirSync(configsDir, { recursive: true });
         }
         
         const sourceConfigsDir = path.join(appPath, 'Contents', 'Resources', 'backend', 'configs');
         if (fs.existsSync(sourceConfigsDir)) {
             log('📋 Copying config files...');
             const configFiles = fs.readdirSync(sourceConfigsDir);
             for (const configFile of configFiles) {
                 const sourcePath = path.join(sourceConfigsDir, configFile);
                 const targetPath = path.join(configsDir, configFile);
                 fs.copyFileSync(sourcePath, targetPath);
                 log(`  ✅ Copied: ${configFile}`);
             }
         } else {
             log('⚠️  Config directory not found, backend may fail');
         }
        
        // Test backend startup
        const testBackend = () => {
            return new Promise((resolve, reject) => {
                log('🔄 Starting backend process...');
                
                const env = {
                    ...process.env,
                    PORT: '8001', // Use different port for testing
                    PYTHONPATH: testWorkingDir,
                    MIRIX_PG_URI: '', // Force SQLite
                    DEBUG: 'true',
                    MIRIX_DEBUG: 'true'
                };
                
                const backend = spawn(backendPath, ['--host', '0.0.0.0', '--port', '8001'], {
                    cwd: testWorkingDir,
                    env: env,
                    stdio: 'pipe'
                });
                
                let backendOutput = '';
                let backendErrors = '';
                let healthCheckStarted = false;
                
                backend.stdout.on('data', (data) => {
                    const output = data.toString();
                    backendOutput += output;
                    log(`Backend STDOUT: ${output.trim()}`);
                    
                    if (output.includes('Uvicorn running on') || 
                        output.includes('Application startup complete') ||
                        output.includes('Started server process')) {
                        
                        if (!healthCheckStarted) {
                            healthCheckStarted = true;
                            log('✅ Backend startup detected!');
                            
                            // Give it a moment then test health endpoint
                            setTimeout(async () => {
                                try {
                                    const http = require('http');
                                    const healthReq = http.get('http://127.0.0.1:8001/health', { timeout: 5000 }, (res) => {
                                        let healthData = '';
                                        res.on('data', chunk => healthData += chunk);
                                        res.on('end', () => {
                                            if (res.statusCode === 200) {
                                                log('✅ Health check passed!');
                                                log(`Health response: ${healthData}`);
                                                backend.kill();
                                                resolve({
                                                    success: true,
                                                    output: backendOutput,
                                                    errors: backendErrors
                                                });
                                            } else {
                                                log(`❌ Health check failed: ${res.statusCode}`);
                                                backend.kill();
                                                reject(new Error(`Health check failed: ${res.statusCode}`));
                                            }
                                        });
                                    });
                                    
                                    healthReq.on('error', (error) => {
                                        log(`❌ Health check error: ${error.message}`);
                                        backend.kill();
                                        reject(error);
                                    });
                                    
                                    healthReq.on('timeout', () => {
                                        log('❌ Health check timeout');
                                        backend.kill();
                                        reject(new Error('Health check timeout'));
                                    });
                                    
                                } catch (error) {
                                    log(`❌ Health check exception: ${error.message}`);
                                    backend.kill();
                                    reject(error);
                                }
                            }, 3000);
                        }
                    }
                });
                
                backend.stderr.on('data', (data) => {
                    const output = data.toString();
                    backendErrors += output;
                    log(`Backend STDERR: ${output.trim()}`);
                    
                    // Check stderr for startup messages too
                    if (output.includes('Uvicorn running on') || 
                        output.includes('Application startup complete') ||
                        output.includes('Started server process')) {
                        
                        if (!healthCheckStarted) {
                            healthCheckStarted = true;
                            log('✅ Backend startup detected in stderr!');
                            
                            // Give it a moment then test health endpoint
                            setTimeout(async () => {
                                try {
                                    const http = require('http');
                                    const healthReq = http.get('http://127.0.0.1:8001/health', { timeout: 5000 }, (res) => {
                                        let healthData = '';
                                        res.on('data', chunk => healthData += chunk);
                                        res.on('end', () => {
                                            if (res.statusCode === 200) {
                                                log('✅ Health check passed!');
                                                log(`Health response: ${healthData}`);
                                                backend.kill();
                                                resolve({
                                                    success: true,
                                                    output: backendOutput,
                                                    errors: backendErrors
                                                });
                                            } else {
                                                log(`❌ Health check failed: ${res.statusCode}`);
                                                backend.kill();
                                                reject(new Error(`Health check failed: ${res.statusCode}`));
                                            }
                                        });
                                    });
                                    
                                    healthReq.on('error', (error) => {
                                        log(`❌ Health check error: ${error.message}`);
                                        backend.kill();
                                        reject(error);
                                    });
                                    
                                    healthReq.on('timeout', () => {
                                        log('❌ Health check timeout');
                                        backend.kill();
                                        reject(new Error('Health check timeout'));
                                    });
                                    
                                } catch (error) {
                                    log(`❌ Health check exception: ${error.message}`);
                                    backend.kill();
                                    reject(error);
                                }
                            }, 3000);
                        }
                    }
                });
                
                backend.on('close', (code) => {
                    log(`Backend process exited with code ${code}`);
                    if (code !== 0 && !healthCheckStarted) {
                        reject(new Error(`Backend exited with code ${code}\nOutput: ${backendOutput}\nErrors: ${backendErrors}`));
                    }
                });
                
                backend.on('error', (error) => {
                    log(`Backend process error: ${error.message}`);
                    reject(error);
                });
                
                // Timeout after 30 seconds
                setTimeout(() => {
                    if (backend.exitCode === null && !healthCheckStarted) {
                        log('❌ Backend startup timeout (30s)');
                        backend.kill();
                        reject(new Error('Backend startup timeout'));
                    }
                }, 30000);
            });
        };
        
        // Run the test
        const result = await testBackend();
        
        if (result.success) {
            log('🎉 DMG TEST PASSED! Backend started successfully and responded to health check');
            log('✅ The DMG appears to be working correctly');
        } else {
            log('❌ DMG TEST FAILED!');
        }
        
        // Cleanup test directory
        try {
            fs.rmSync(testWorkingDir, { recursive: true, force: true });
            log('🧹 Cleaned up test directory');
        } catch (error) {
            log(`⚠️  Failed to cleanup test directory: ${error.message}`);
        }
        
    } catch (error) {
        log(`❌ DMG test failed: ${error.message}`);
        if (error.stack) {
            log(`Stack trace: ${error.stack}`);
        }
        process.exit(1);
    } finally {
        cleanup();
    }
}

// Run the test
main().catch(error => {
    log(`❌ Unhandled error: ${error.message}`);
    process.exit(1);
}); 