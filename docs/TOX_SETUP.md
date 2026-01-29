# Tox Setup Guide

## What is Tox?

**Tox** is a tool for running tests and code quality checks in isolated, reproducible environments. Think of it as a standardized way to run your project's checks that works the same way locally and in CI/CD.

### Key Benefits:

1. **Isolated Environments**: Each check runs in its own virtual environment, preventing dependency conflicts
2. **Reproducible**: Same commands work locally and in CI/CD
3. **Multiple Environments**: Can test against multiple Python versions
4. **Standardized**: One command (`tox`) runs all checks consistently

### How Tox Works:

1. Creates a virtual environment for each "testenv" (format, lint, typecheck, etc.)
2. Installs dependencies specified for that environment
3. Runs the commands you've configured
4. Reports success/failure

## Tox Environments in This Project

Our `tox.ini` defines several environments:

### `format`
Checks code formatting with Black and isort:
```bash
tox -e format
```
- **What it does**: Verifies that code is properly formatted
- **Fails if**: Files need reformatting (shows diff of what needs to change)
- **Use case**: CI/CD checks, pre-commit validation

### `format-fix`
Automatically fixes formatting:
```bash
tox -e format-fix
```
- **What it does**: Runs `isort .` and `black .` to fix formatting
- **Use case**: Local development when you want to auto-fix formatting

### `lint`
Runs linting with ruff:
```bash
tox -e lint
```
- **What it does**: Checks code quality and style issues
- **Fails if**: Linting errors are found

### `typecheck`
Runs type checking with mypy:
```bash
tox -e typecheck
```
- **What it does**: Validates type hints and catches type errors
- **Fails if**: Type errors are found

### `test`
Runs pytest tests:
```bash
tox -e test
```
- **What it does**: Runs all tests
- **Fails if**: Tests fail

### `all`
Runs all checks (format, lint, typecheck, test):
```bash
tox -e all
```
- **What it does**: Runs everything in sequence
- **Fails if**: Any check fails

## Installation

### Install Tox:
```bash
pip install tox
# OR
poetry add --group dev tox
```

### First Run:
```bash
tox
```
This will create virtual environments and run all default environments (`format`, `lint`, `typecheck`).

## Usage

### Run All Default Checks:
```bash
tox
```

### Run Specific Environment:
```bash
tox -e format      # Check formatting
tox -e lint        # Run linting
tox -e typecheck   # Type checking
tox -e test        # Run tests
tox -e all         # Run everything
```

### Auto-fix Formatting:
```bash
tox -e format-fix
```

### Run Multiple Environments:
```bash
tox -e format,lint,typecheck
```

### Recreate Environments (if dependencies change):
```bash
tox --recreate -e format
```

### Show What Would Run (dry run):
```bash
tox --list
```

## GitHub Actions Integration

Tox is integrated into our GitHub Actions workflow (`.github/workflows/tests.yml`). The workflow includes:

1. **`format` job**: Runs `tox -e format` - **BLOCKS MERGES** if formatting is incorrect
2. **`lint` job**: Runs `tox -e lint` - **BLOCKS MERGES** if linting fails
3. **`typecheck` job**: Runs `tox -e typecheck` - **BLOCKS MERGES** if type checking fails

### How It Blocks Merges:

1. When you create a Pull Request, GitHub Actions runs these jobs
2. If any job fails (red X), the PR cannot be merged
3. You'll see the failure in the PR checks section
4. Fix the issues and push again - the checks will re-run

### Required Status Checks:

To ensure these checks block merges:

1. Go to your repository settings
2. Navigate to **Branches** → **Branch protection rules**
3. Edit the rule for your main branch (e.g., `main`, `re-org`)
4. Under **Require status checks to pass before merging**, enable:
   - ✅ `format` (Code Formatting Check)
   - ✅ `lint` (Linting checks)
   - ✅ `typecheck` (Type Checking)
5. Save the changes

Now PRs cannot be merged until all these checks pass!

## Local Development Workflow

### Before Committing:
```bash
# Check formatting
tox -e format

# If formatting fails, fix it:
tox -e format-fix

# Check linting
tox -e lint

# Check types
tox -e typecheck

# Or run everything:
tox -e all
```

### Quick Fix Workflow:
```bash
# 1. Make your changes
# 2. Auto-fix formatting
tox -e format-fix

# 3. Run all checks
tox -e all

# 4. If everything passes, commit and push
```

## Troubleshooting

### "Command not found: tox"
Install tox:
```bash
pip install tox
```

### "Environment creation failed"
Try recreating the environment:
```bash
tox --recreate -e format
```

### "Dependencies not found"
Tox creates isolated environments. Make sure your `tox.ini` lists all required dependencies in the `deps` section for each environment.

### "Formatting check fails but code looks fine"
Run `tox -e format-fix` to see what Black/isort would change, then review the diff.

### "Tox is slow"
- Use `tox --parallel` to run environments in parallel (if you have multiple)
- Environments are cached - first run is slower
- Use `tox -e format` instead of `tox -e all` if you only need formatting checks

## Configuration File

The `tox.ini` file contains all configuration. Key sections:

- `[tox]`: Global settings (envlist, minversion)
- `[testenv]`: Base configuration for all environments
- `[testenv:format]`: Specific configuration for the format environment
- `[testenv:lint]`: Specific configuration for the lint environment
- etc.

To modify checks, edit the `commands` section in the relevant `[testenv:*]` block.

## Summary

- **Tox** = Standardized way to run checks in isolated environments
- **`tox -e format`** = Check if code is properly formatted (used in CI)
- **`tox -e format-fix`** = Auto-fix formatting issues
- **GitHub Actions** = Runs tox automatically on PRs
- **Branch Protection** = Ensures tox checks must pass before merging

This ensures consistent code quality across all contributions! 🎉
