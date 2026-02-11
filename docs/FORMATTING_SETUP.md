# Formatting Setup Instructions

This document explains how to set up consistent code formatting for the MIRIX project.

## Quick Start

1. **Install dependencies:**
   ```bash
   poetry install --extras dev
   ```

2. **Install VS Code/Cursor extensions:**
   - Open VS Code/Cursor
   - Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
   - Type "Extensions: Show Recommended Extensions"
   - Install the recommended extensions (Python, Black Formatter, mypy)

3. **Restart your editor** to pick up the new settings

5. **Format all files (one-time):**
   ```bash
   poetry run isort .
   poetry run black .
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

**Important:** This project uses Poetry for dependency management. Always use `poetry run` to ensure you're using the correct versions.

### Format All Files at Once

```bash
poetry run isort .
poetry run black .
```

### Format Specific Files

```bash
# Single file
poetry run black path/to/file.py
poetry run isort path/to/file.py

# Directory
poetry run black mirix/server/
poetry run isort mirix/server/
```

### Check Formatting Without Changing Files

```bash
# Check with Black
poetry run black --check .

# Check with isort
poetry run isort --check-only .

# Check with mypy
poetry run mypy mirix/
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
poetry install --extras dev
```

If using Poetry, ensure you're running commands with `poetry run`:
```bash
poetry run black .
```

### Formatting conflicts persist

1. **Verify Python interpreter is set to Poetry environment:**
   - Check bottom-right corner of VS Code/Cursor
   - Should show Poetry virtual environment (`.venv` or `poetry`)
   - If not, select it: `Cmd+Shift+P` -> "Python: Select Interpreter"
   - The Black Formatter extension uses `black-formatter.importStrategy: "fromEnvironment"` to use Poetry's Black

2. **Check for conflicting user settings:**
   - Open Settings (`Cmd+,` / `Ctrl+,`)
   - Search for "format on save"
   - Workspace settings should show `✓` indicating they override user settings

3. **Reload the window:**
   - Press `Cmd+Shift+P` / `Ctrl+Shift+P`
   - Type "Developer: Reload Window"

4. **Check `.editorconfig` is being respected:**
   - Install the EditorConfig extension if needed
   - Most editors respect `.editorconfig` automatically

5. **Verify Black version matches:**
   - Command line: `poetry run black --version` (should show 25.11.0)
   - Editor uses the same version from Poetry's virtual environment

### Different line lengths

The project uses **120 characters** (not the PEP 8 default of 79). This is configured in:
- `pyproject.toml` (`[tool.black]` section)
- `.editorconfig`
- `.vscode/settings.json`

All three should match. If they don't, update them to use 120.


## Summary

- **Formatter**: Black 25.11.0 (120 char line length, pinned version)
- **Import sorter**: isort (configured for Black)
- **Type checker**: mypy
- **Dependency management**: Poetry
- **Settings location**: `.vscode/settings.json` (workspace settings override user settings)
- **Format all files**: `poetry run isort . && poetry run black .`
