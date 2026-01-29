#!/usr/bin/env python3
"""
Mirix Client Package Setup
===========================

This setup script packages ONLY the client-side components of Mirix.
The client package is lightweight and contains only what's needed to
communicate with a Mirix server.

Package Name: mirix-client (default, configurable via --package-name)
Purpose: Remote client library for Mirix server
"""

import os
import sys

from setuptools import find_packages, setup

# Parse command line arguments for package name and version
package_name = "mirix-client"  # Default value
version = None

if "--package-name" in sys.argv:
    try:
        idx = sys.argv.index("--package-name")
        package_name = sys.argv[idx + 1]
        # Remove the --package-name argument and its value from sys.argv
        # so setuptools doesn't see it
        sys.argv.pop(idx)  # Remove --package-name
        sys.argv.pop(idx)  # Remove the value
    except (IndexError, ValueError):
        print("Error: --package-name requires a value")
        sys.exit(1)

if "--version" in sys.argv:
    try:
        idx = sys.argv.index("--version")
        version = sys.argv[idx + 1]
        # Remove the --version argument and its value from sys.argv
        # so setuptools doesn't see it
        sys.argv.pop(idx)  # Remove --version
        sys.argv.pop(idx)  # Remove the value
    except (IndexError, ValueError):
        print("Error: --version requires a value")
        sys.exit(1)

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
# Go up two levels: packaging/ -> scripts/ -> project_root/
project_root = os.path.dirname(os.path.dirname(this_directory))
CLIENT_VERSION = "0.1.0"

with open(os.path.join(project_root, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def load_requirements(filename: str) -> list[str]:
    """Load dependency list from a requirements file, ignoring comments/blanks."""
    req_path = os.path.join(this_directory, filename)
    with open(req_path, encoding="utf-8") as req_file:
        return [line.strip() for line in req_file.readlines() if line.strip() and not line.startswith("#")]


# Get version from command line or __init__.py
def get_version():
    # If version was provided via command line, use that
    if version is not None:
        return version

    # Otherwise, fall back to reading from __init__.py
    import re

    version_file = os.path.join(project_root, "mirix", "__init__.py")
    with open(version_file, "r", encoding="utf-8") as version_f:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_f.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


# Client-specific dependencies (minimal)
client_dependencies = load_requirements("requirements_client.txt")

# Change to project root directory so we can use relative paths
os.chdir(project_root)

setup(
    name=package_name,
    version=get_version(),
    author="Mirix AI",
    author_email="yuwang@mirix.io",
    description="Mirix Client - Lightweight Python client for Mirix server",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mirix-AI/MIRIX",
    project_urls={
        "Documentation": "https://docs.mirix.io",
        "Website": "https://mirix.io",
        "Source Code": "https://github.com/Mirix-AI/MIRIX",
        "Bug Reports": "https://github.com/Mirix-AI/MIRIX/issues",
    },
    # Only include client-related packages (explicit to avoid pulling server code)
    packages=[
        "mirix.client",
        "mirix.schemas",
        "mirix.schemas.openai",
        "mirix.helpers",
    ],
    py_modules=[
        "mirix.system",
        "mirix.settings",
        "mirix.log",
        "mirix.constants",
    ],
    package_dir={"": "."},
    include_package_data=False,
    install_requires=client_dependencies,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-asyncio",
            "black",
            "isort",
            "flake8",
            "mypy",
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Communications :: Chat",
    ],
    keywords="ai, memory, agent, llm, assistant, client, api",
    license="Apache License 2.0",
    zip_safe=False,
)
