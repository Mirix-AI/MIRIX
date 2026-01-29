# Formatting Setup Instructions

This document explains how to set up consistent code formatting for the MIRIX project.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -e ".[dev]"
   # OR if using poetry:
   poetry install --extras dev
   ```

2. **Install VS Code/Cursor extensions:**
   - Open VS Code/Cursor
   - Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
   - Type "Extensions: Show Recommended Extensions"
   - Install the recommended extensions (Python, Black Formatter, mypy)

3. **Restart your editor** to pick up the new settings

4. **Format all files (one-time):**
   ```bash
   isort .
   black .
   ```

## How Workspace Settings Work

**VS Code/Cursor automatically prioritizes workspace settings over user settings.** The `.vscode/settings.json` file in this repository will override your personal editor settings when you're working in this project.

### To Verify Your Settings:

1. Open any Python file
2. Make a formatting change (add extra spaces, etc.)
3. Save the file (`Cmd+S` / `Ctrl+S`)
4. The file should automatically format using Black with 120 character line length

### If Formatting Doesn't Work:

1. **Check that Black Formatter extension is installed:**
   - Open Extensions (`Cmd+Shift+X` / `Ctrl+Shift+X`)
   - Search for "Black Formatter"
   - Install if not present

2. **Check your Python interpreter:**
   - Press `Cmd+Shift+P` / `Ctrl+Shift+P`
   - Type "Python: Select Interpreter"
   - Choose the project's virtual environment

3. **Verify format on save is enabled:**
   - The workspace settings should have `"editor.formatOnSave": true`
   - If your personal settings have this disabled, the workspace setting should override it

## Formatting Commands

### Format All Files at Once

```bash
isort .
black .
```

### Format Specific Files

```bash
# Single file
black path/to/file.py
isort path/to/file.py

# Directory
black mirix/server/
isort mirix/server/
```

### Check Formatting Without Changing Files

```bash
# Check with Black
black --check .

# Check with isort
isort --check-only .

# Check with mypy
mypy mirix/
```

## Type Checking with mypy

```bash
# Run mypy on the codebase
mypy mirix/

# Check specific file
mypy mirix/server/rest_api.py
```

## Troubleshooting

### "Black not found" error

Make sure you've installed dev dependencies:
```bash
pip install -e ".[dev]"
```

### Formatting conflicts persist

1. **Check for conflicting user settings:**
   - Open Settings (`Cmd+,` / `Ctrl+,`)
   - Search for "format on save"
   - Workspace settings should show `✓` indicating they override user settings

2. **Reload the window:**
   - Press `Cmd+Shift+P` / `Ctrl+Shift+P`
   - Type "Developer: Reload Window"

3. **Check `.editorconfig` is being respected:**
   - Install the EditorConfig extension if needed
   - Most editors respect `.editorconfig` automatically

### Different line lengths

The project uses **120 characters** (not the PEP 8 default of 79). This is configured in:
- `pyproject.toml` (`[tool.black]` section)
- `.editorconfig`
- `.vscode/settings.json`

All three should match. If they don't, update them to use 120.


## Summary

- **Formatter**: Black (120 char line length)
- **Import sorter**: isort (configured for Black)
- **Type checker**: mypy
- **Settings location**: `.vscode/settings.json` (workspace settings override user settings)
- **Format all files**: `make format-black` or `isort . && black .`
