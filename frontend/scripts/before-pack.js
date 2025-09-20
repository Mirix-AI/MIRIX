const fs = require('fs');
const path = require('path');

console.log('🔧 Running before-pack script...');

// Check if PGlite files exist and are in the correct format
const pglitePath = path.join(__dirname, '..', 'node_modules', '@electric-sql', 'pglite');
if (!fs.existsSync(pglitePath)) {
  console.error('❌ PGlite package not found!');
  console.log('Install PGlite: npm install @electric-sql/pglite');
  process.exit(1);
}

const distPath = path.join(pglitePath, 'dist');
if (!fs.existsSync(distPath)) {
  console.error('❌ PGlite dist directory not found!');
  process.exit(1);
}

const pgliteDataPath = path.join(distPath, 'pglite.data');
if (!fs.existsSync(pgliteDataPath)) {
  console.error('❌ pglite.data not found!');
  process.exit(1);
}

// Check if pglite.data is a file (it should be in v0.3.x)
const stats = fs.statSync(pgliteDataPath);
if (!stats.isFile()) {
  console.error('❌ pglite.data is not a file!');
  console.log('In PGlite v0.3.x, pglite.data should be a file, not a directory.');
  process.exit(1);
}

// Check file size
const fileSizeInMB = (stats.size / (1024 * 1024)).toFixed(2);
console.log(`✅ pglite.data is a file with size: ${fileSizeInMB} MB`);

// Check for pglite.wasm
const pgliteWasmPath = path.join(distPath, 'pglite.wasm');
if (!fs.existsSync(pgliteWasmPath)) {
  console.error('❌ pglite.wasm not found!');
  process.exit(1);
}

const wasmStats = fs.statSync(pgliteWasmPath);
const wasmSizeInMB = (wasmStats.size / (1024 * 1024)).toFixed(2);
console.log(`✅ pglite.wasm is a file with size: ${wasmSizeInMB} MB`);

console.log('✅ PGlite pre-pack validation completed successfully'); 