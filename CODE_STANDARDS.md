# MIRIX Code Standards

## Code Formatting and Style

MIRIX follows PEP 8 style guidelines with some project-specific modifications. We use automated tools to ensure consistency across the codebase.

### Formatting Tools

- **[Black](https://black.readthedocs.io/)**: Code formatter (configured in `pyproject.toml`)
- **[isort](https://pycqa.github.io/isort/)**: Import sorter (configured to work with Black)
- **[mypy](https://mypy.readthedocs.io/)**: Static type checker
- **[Ruff](https://docs.astral.sh/ruff/)**: Fast Python linter
- **[Pyright](https://microsoft.github.io/pyright/#/)**: Type checker

### Style Guidelines

- **Indentation**: Use 4 spaces per indentation level. Never use tabs.
- **Line Length**: Maximum 120 characters (configured in Black and isort)
- **Blank Lines**: 
  - Two blank lines to separate top-level functions and classes
  - One blank line to separate methods within a class
- **Imports**: 
  - Place all import statements at the top of the file
  - Group imports: standard library, third-party, local application (with blank lines between groups)
  - Use `isort` to automatically organize imports (configured with Black profile)
- **Naming Conventions**:
  - **Variables and Functions**: `snake_case` (lowercase with underscores)
  - **Classes**: `CamelCase` (each word capitalized)
  - **Constants**: `ALL_CAPS_WITH_UNDERSCORES`
- **Whitespace**: Use spaces around operators and after commas, but not directly inside parentheses or brackets
- **Comments**: Use inline comments sparingly. They should explain *why*, not *what*. Separate inline comments from statements by at least two spaces
- **Docstrings**: Use triple double quotes (`"""`) for docstrings. Clearly describe the purpose of modules, classes, and functions

### Editor Configuration

The repository includes configuration files to ensure consistent formatting:

- **`.editorconfig`**: Defines basic editor settings (indentation, line endings, etc.)
- **`.vscode/settings.json`**: VS Code/Cursor settings for automatic formatting with Black
- **`pyproject.toml`**: Contains Black, isort, and mypy configuration

### Running Formatting Tools

```bash
# Format code with Black
black .

# Sort imports with isort
isort .

# Type check with mypy
mypy mirix/

# Lint with Ruff
ruff check .
```

### IDE Setup

For VS Code/Cursor users, the `.vscode/settings.json` file configures:
- Black as the default formatter
- Format on save enabled
- Automatic import organization
- mypy type checking

Make sure you have the following extensions installed:
- Python extension (ms-python.python)
- Black Formatter extension (ms-python.black-formatter)

The editor will automatically format your code on save using the project's Black configuration.