#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const os = require('os');

console.log('🔍 Testing MIRIX Backend Logging System...');
console.log('');

// Check if debug directory exists
const debugDir = path.join(os.homedir(), '.mirix', 'debug');
console.log(`📁 Debug directory: ${debugDir}`);

if (!fs.existsSync(debugDir)) {
    console.log('❌ Debug directory does not exist yet');
    console.log('   This is normal if the app hasn\'t been run yet');
} else {
    console.log('✅ Debug directory exists');
    
    // List all backend log files
    const files = fs.readdirSync(debugDir);
    const backendLogFiles = files.filter(file => file.startsWith('backend-') && file.endsWith('.log'));
    const electronLogFiles = files.filter(file => file.startsWith('electron-startup-') && file.endsWith('.log'));
    
    console.log(`📄 Found ${backendLogFiles.length} backend log files:`);
    backendLogFiles.forEach(file => {
        const filePath = path.join(debugDir, file);
        const stats = fs.statSync(filePath);
        console.log(`   - ${file} (${(stats.size / 1024).toFixed(2)} KB, modified: ${stats.mtime.toLocaleString()})`);
    });
    
    console.log(`📄 Found ${electronLogFiles.length} electron startup log files:`);
    electronLogFiles.forEach(file => {
        const filePath = path.join(debugDir, file);
        const stats = fs.statSync(filePath);
        console.log(`   - ${file} (${(stats.size / 1024).toFixed(2)} KB, modified: ${stats.mtime.toLocaleString()})`);
    });
    
    // Show content of the most recent backend log file
    if (backendLogFiles.length > 0) {
        const mostRecentBackendLog = backendLogFiles
            .map(file => ({ file, mtime: fs.statSync(path.join(debugDir, file)).mtime }))
            .sort((a, b) => b.mtime - a.mtime)[0];
        
        console.log('');
        console.log(`📋 Content of most recent backend log (${mostRecentBackendLog.file}):`);
        console.log('─'.repeat(60));
        
        const logContent = fs.readFileSync(path.join(debugDir, mostRecentBackendLog.file), 'utf8');
        const lines = logContent.split('\n');
        
        // Show first 20 lines and last 20 lines
        if (lines.length <= 40) {
            console.log(logContent);
        } else {
            console.log(lines.slice(0, 20).join('\n'));
            console.log(`\n... (${lines.length - 40} lines omitted) ...\n`);
            console.log(lines.slice(-20).join('\n'));
        }
        
        console.log('─'.repeat(60));
    }
}

console.log('');
console.log('🎯 Next steps:');
console.log('1. Run the MIRIX app to generate backend logs');
console.log('2. Close and reopen the app to test backend reconnection');
console.log('3. Check the debug directory for new log files');
console.log('4. Backend logs will show detailed startup, health check, and error information'); 