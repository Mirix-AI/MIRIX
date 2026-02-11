#!/usr/bin/env python3
"""
Mirix Python Cache Cleanup Script

This script removes all Python cache files and directories from the project.
Works on Windows, macOS, and Linux.

Usage:
    python scripts/clean_cache.py
    or
    ./scripts/clean_cache.py
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple


# ANSI color codes for terminal output
class Colors:
    """Terminal color codes."""

    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    NC = "\033[0m"  # No Color


def print_header():
    """Print script header."""
    print(f"{Colors.GREEN}={'=' * 40}{Colors.NC}")
    print(f"{Colors.GREEN}Mirix Python Cache Cleanup{Colors.NC}")
    print(f"{Colors.GREEN}={'=' * 40}{Colors.NC}")
    print()


def get_project_root() -> Path:
    """Get the project root directory."""
    # Script is in scripts/ directory, so parent is project root
    return Path(__file__).parent.parent.resolve()


def remove_directories(root: Path, pattern: str) -> int:
    """
    Remove all directories matching the pattern.

    Args:
        root: Root directory to search from
        pattern: Directory name pattern to match

    Returns:
        Number of directories removed
    """
    count = 0
    for dirpath in root.rglob(pattern):
        if dirpath.is_dir():
            try:
                shutil.rmtree(dirpath)
                count += 1
            except (OSError, PermissionError) as e:
                print(f"{Colors.RED}Warning: Could not remove {dirpath}: {e}{Colors.NC}")
    return count


def remove_files(root: Path, pattern: str) -> int:
    """
    Remove all files matching the pattern.

    Args:
        root: Root directory to search from
        pattern: File name pattern to match

    Returns:
        Number of files removed
    """
    count = 0
    for filepath in root.rglob(pattern):
        if filepath.is_file():
            try:
                filepath.unlink()
                count += 1
            except (OSError, PermissionError) as e:
                print(f"{Colors.RED}Warning: Could not remove {filepath}: {e}{Colors.NC}")
    return count


def clean_cache() -> List[Tuple[str, int]]:
    """
    Clean all Python cache files and directories.

    Returns:
        List of (description, count) tuples for each type cleaned
    """
    project_root = get_project_root()
    print(f"{Colors.YELLOW}Project root: {project_root}{Colors.NC}")
    print()

    results = []

    # 1. Remove __pycache__ directories
    print(f"{Colors.YELLOW}Cleaning __pycache__ directories...{Colors.NC}")
    count = remove_directories(project_root, "__pycache__")
    results.append(("__pycache__ directories", count))
    print(f"{Colors.GREEN}✓ Removed {count} __pycache__ directories{Colors.NC}")

    # 2. Remove .pyc files
    print(f"{Colors.YELLOW}Cleaning .pyc files...{Colors.NC}")
    count = remove_files(project_root, "*.pyc")
    results.append((".pyc files", count))
    print(f"{Colors.GREEN}✓ Removed {count} .pyc files{Colors.NC}")

    # 3. Remove .pyo files
    print(f"{Colors.YELLOW}Cleaning .pyo files...{Colors.NC}")
    count = remove_files(project_root, "*.pyo")
    results.append((".pyo files", count))
    print(f"{Colors.GREEN}✓ Removed {count} .pyo files{Colors.NC}")

    # 4. Remove .pytest_cache directories
    print(f"{Colors.YELLOW}Cleaning .pytest_cache directories...{Colors.NC}")
    count = remove_directories(project_root, ".pytest_cache")
    results.append((".pytest_cache directories", count))
    print(f"{Colors.GREEN}✓ Removed {count} .pytest_cache directories{Colors.NC}")

    # 5. Remove .mypy_cache directories
    print(f"{Colors.YELLOW}Cleaning .mypy_cache directories...{Colors.NC}")
    count = remove_directories(project_root, ".mypy_cache")
    results.append((".mypy_cache directories", count))
    print(f"{Colors.GREEN}✓ Removed {count} .mypy_cache directories{Colors.NC}")

    # 6. Remove .coverage files
    print(f"{Colors.YELLOW}Cleaning .coverage files...{Colors.NC}")
    count = remove_files(project_root, ".coverage")
    results.append((".coverage files", count))
    print(f"{Colors.GREEN}✓ Removed {count} .coverage files{Colors.NC}")

    # 7. Remove .coverage.* files
    print(f"{Colors.YELLOW}Cleaning .coverage.* files...{Colors.NC}")
    count = remove_files(project_root, ".coverage.*")
    results.append((".coverage.* files", count))
    print(f"{Colors.GREEN}✓ Removed {count} .coverage.* files{Colors.NC}")

    # 8. Remove htmlcov directories
    print(f"{Colors.YELLOW}Cleaning htmlcov directories...{Colors.NC}")
    count = remove_directories(project_root, "htmlcov")
    results.append(("htmlcov directories", count))
    print(f"{Colors.GREEN}✓ Removed {count} htmlcov directories{Colors.NC}")

    # 9. Remove .eggs directories
    print(f"{Colors.YELLOW}Cleaning .eggs directories...{Colors.NC}")
    count = remove_directories(project_root, ".eggs")
    results.append((".eggs directories", count))
    print(f"{Colors.GREEN}✓ Removed {count} .eggs directories{Colors.NC}")

    # 10. Remove *.egg-info directories
    print(f"{Colors.YELLOW}Cleaning *.egg-info directories...{Colors.NC}")
    count = remove_directories(project_root, "*.egg-info")
    results.append(("*.egg-info directories", count))
    print(f"{Colors.GREEN}✓ Removed {count} *.egg-info directories{Colors.NC}")

    # 11. Remove build directories
    print(f"{Colors.YELLOW}Cleaning build directories...{Colors.NC}")
    count = remove_directories(project_root, "build")
    results.append(("build directories", count))
    print(f"{Colors.GREEN}✓ Removed {count} build directories{Colors.NC}")

    # 12. Remove dist directories
    print(f"{Colors.YELLOW}Cleaning dist directories...{Colors.NC}")
    count = remove_directories(project_root, "dist")
    results.append(("dist directories", count))
    print(f"{Colors.GREEN}✓ Removed {count} dist directories{Colors.NC}")

    # 13. Remove .ruff_cache directories
    print(f"{Colors.YELLOW}Cleaning .ruff_cache directories...{Colors.NC}")
    count = remove_directories(project_root, ".ruff_cache")
    results.append((".ruff_cache directories", count))
    print(f"{Colors.GREEN}✓ Removed {count} .ruff_cache directories{Colors.NC}")

    # 14. Remove .ipynb_checkpoints directories
    print(f"{Colors.YELLOW}Cleaning .ipynb_checkpoints directories...{Colors.NC}")
    count = remove_directories(project_root, ".ipynb_checkpoints")
    results.append((".ipynb_checkpoints directories", count))
    print(f"{Colors.GREEN}✓ Removed {count} .ipynb_checkpoints directories{Colors.NC}")

    return results


def print_summary(results: List[Tuple[str, int]]):
    """Print cleanup summary."""
    print()
    print(f"{Colors.GREEN}={'=' * 40}{Colors.NC}")
    print(f"{Colors.GREEN}✓ Python cache cleanup complete!{Colors.NC}")
    print(f"{Colors.GREEN}={'=' * 40}{Colors.NC}")
    print()

    total_items = sum(count for _, count in results)
    print(f"Total items removed: {Colors.BLUE}{total_items}{Colors.NC}")
    print()
    print("Cache types cleaned:")
    for description, count in results:
        if count > 0:
            print(f"  {Colors.GREEN}✓{Colors.NC} {description}: {count}")
        else:
            print(f"  • {description}: 0")
    print()


def main():
    """Main function."""
    try:
        print_header()
        results = clean_cache()
        print_summary(results)
        return 0
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Cleanup interrupted by user{Colors.NC}")
        return 1
    except Exception as e:
        print(f"\n{Colors.RED}Error during cleanup: {e}{Colors.NC}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
