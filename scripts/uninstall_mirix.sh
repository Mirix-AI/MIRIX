#!/bin/bash

echo "🗑️  Completely removing MIRIX from macOS..."

# Remove the main application
echo "Removing application bundle..."
rm -rf /Applications/MIRIX.app
rm -rf ~/Applications/MIRIX.app

# Remove MIRIX-specific data directory
echo "Removing MIRIX data directory..."
rm -rf ~/.mirix

# Remove preferences
echo "Removing preferences..."
rm -rf ~/Library/Preferences/com.electron.mirix.*
rm -rf ~/Library/Preferences/com.mirix.*

# Remove caches
echo "Removing caches..."
rm -rf ~/Library/Caches/com.electron.mirix.*
rm -rf ~/Library/Caches/com.mirix.*

# Remove application support
echo "Removing application support files..."
rm -rf ~/Library/Application\ Support/MIRIX
rm -rf ~/Library/Application\ Support/mirix

# Remove logs
echo "Removing logs..."
rm -rf ~/Library/Logs/MIRIX
rm -rf ~/Library/Logs/mirix

# Remove saved application state
echo "Removing saved application state..."
rm -rf ~/Library/Saved\ Application\ State/com.electron.mirix.*
rm -rf ~/Library/Saved\ Application\ State/com.mirix.*

# Remove any remaining Electron caches
echo "Removing Electron caches..."
rm -rf ~/Library/Caches/Electron
rm -rf ~/Library/Application\ Support/Electron

echo "✅ MIRIX has been completely removed from your system!"
echo "You may want to restart your Mac to clear any remaining system caches." 