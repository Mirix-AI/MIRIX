# Tox Quick Start

## What is Tox?

**Tox** is a tool that runs your code quality checks (formatting, linting, type checking) in isolated environments. It ensures the same checks work locally and in CI/CD.

## Quick Commands

```bash
# Install tox
pip install tox
# OR
poetry install --extras dev  # (tox is now in dev dependencies)

# Check formatting (what CI runs)
tox -e format

# Auto-fix formatting issues
tox -e format-fix

# Run all checks
tox -e all
```

## GitHub Actions Integration

✅ **Already configured!** The workflow (`.github/workflows/tests.yml`) now includes:

1. **`format` job** - Runs `tox -e format` (checks Black + isort)
2. **`lint` job** - Runs `tox -e lint` (checks ruff)
3. **`typecheck` job** - Runs `tox -e typecheck` (checks mypy)

**These jobs will BLOCK merges if they fail!**

## Setting Up Branch Protection (One-Time Setup)

To ensure PRs can't be merged without passing checks:

1. Go to: **Repository Settings** → **Branches** → **Branch protection rules**
2. Click **Add rule** or edit existing rule for your main branch
3. Enable: **Require status checks to pass before merging**
4. Check these required status checks:
   - ✅ `format` (Code Formatting Check)
   - ✅ `lint` (Linting checks)  
   - ✅ `typecheck` (Type Checking)
5. Save

Now PRs **cannot be merged** until all checks pass! 🎯

## Workflow

### Before Committing:
```bash
tox -e format-fix  # Auto-fix formatting
tox -e all          # Run all checks
```

### If CI Fails:
1. Check the failed job in GitHub Actions
2. Run the same command locally: `tox -e format` (or whatever failed)
3. Fix the issues
4. Push again - checks will re-run automatically

## What Each Check Does

- **`format`**: Verifies code is formatted with Black (120 char lines) and imports sorted with isort
- **`lint`**: Checks code quality with ruff
- **`typecheck`**: Validates type hints with mypy

## Files Created

- ✅ `tox.ini` - Tox configuration
- ✅ `.github/workflows/tests.yml` - Updated with format/lint/typecheck jobs
- ✅ `pyproject.toml` - Added tox to dev dependencies

See `docs/TOX_SETUP.md` for detailed documentation.
