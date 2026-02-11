#!/usr/bin/env python3
"""
Mirix Server Package Setup
===========================

This setup script packages the full Mirix server including:
- FastAPI REST API server
- PostgreSQL database integration
- Redis caching and search
- All agent implementations (Meta, Memory agents)
- LLM integrations (OpenAI, Gemini, Claude, etc.)
- Queue management
- Function tools
- System prompts

Package Name: mirix-server (default, configurable via --package-name)
Purpose: Complete Mirix AI server with multi-agent memory system
"""

import os
import sys

from setuptools import find_packages, setup

# Parse command line arguments for package name and version
package_name = "mirix-server"  # Default value
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

with open(os.path.join(project_root, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


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


# Server dependencies (comprehensive)
server_dependencies = [
    # Core dependencies
    "pytz>=2024.1",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "openpyxl>=3.1.0",
    "Markdown>=3.5.0",
    "Pillow>=10.2.0,<11.0.0",
    "scikit-image>=0.22.0",
    # LLM APIs
    "openai>=1.108.1,<2.0.0",
    "tiktoken>=0.5.0",
    "google-genai>=0.4.0",
    "anthropic>=0.23.0",
    "cohere>=4.0.0",
    # FastAPI and server
    "fastapi>=0.104.1",
    "uvicorn[standard]>=0.31.1",
    "python-multipart>=0.0.6",
    "httpx>=0.25.0",
    "httpx_sse>=0.3.0",
    # Database
    "sqlalchemy>=2.0.0",
    "psycopg2-binary>=2.9.0",
    "pg8000>=1.30.0",
    "pgvector>=0.2.0",
    "redis>=5.0.0",  # Redis client
    # Pydantic and validation
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    # Utilities
    "python-dotenv>=1.0.0",
    "demjson3>=3.0.0",
    "pathvalidate>=3.0.0",
    "docstring_parser>=0.15",
    "jinja2>=3.1.0",
    "humps>=0.2.0",
    "colorama>=0.4.0",
    "rapidfuzz>=3.0.0",
    "rank-bm25>=0.2.0",
    "psutil>=5.9.0",
    "json_repair>=0.12.0",
    "rich>=13.7.1,<14.0.0",
    "anyio>=4.7.0",
    "pyyaml>=6.0.0",
    "requests>=2.31.0",
    # LlamaIndex
    "llama_index>=0.9.0",
    "llama-index-embeddings-google-genai>=0.1.0",
    # Composio and MCP
    "composio>=0.3.0",
    "mcp>=0.1.0",
    # Google APIs
    "google-auth>=2.0.0",
    "google-auth-oauthlib>=1.0.0",
    "google-auth-httplib2>=0.1.0",
    "google-api-python-client>=2.0.0",
    # Observability
    "opentelemetry-api>=1.20.0",
    "opentelemetry-sdk>=1.20.0",
    "opentelemetry-exporter-otlp>=1.20.0",
    "opentelemetry-instrumentation-requests>=0.41b0",
    "langfuse>=3.11.0,<4.0.0",
    # Voice processing
    "SpeechRecognition>=3.10.0",
    "pydub>=0.25.0",
    # Protobuf (for queue message serialization) - compatible with pynumaflow
    "protobuf>=5.0.0,<6.0.0",
]

# Optional dependencies
extras_require = {
    "dev": [
        "pytest>=6.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
        "mypy>=1.5.0",
        "ruff>=0.1.0",
        "pyright>=1.1.0",
    ],
}

# Change to project root directory so we can use relative paths
os.chdir(project_root)

setup(
    name=package_name,
    version=get_version(),
    author="Mirix AI",
    author_email="yuwang@mirix.io",
    description="MIRIX Server - Multi-Agent Personal Assistant with Advanced Memory System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mirix-AI/MIRIX",
    project_urls={
        "Documentation": "https://docs.mirix.io",
        "Website": "https://mirix.io",
        "Source Code": "https://github.com/Mirix-AI/MIRIX",
        "Bug Reports": "https://github.com/Mirix-AI/MIRIX/issues",
    },
    # Include all packages
    packages=find_packages(
        where=".",
        exclude=["tests*", "scripts*", "frontend*", "public_evaluations*", "samples*"],
    ),
    package_dir={"": "."},
    include_package_data=True,
    package_data={
        "mirix": [
            "*.yaml",
            "*.yml",
            "*.txt",
            "*.proto",
            "configs/**/*.yaml",
            "configs/**/*.yml",
            "prompts/**/*.txt",
            "prompts/**/*.yaml",
            "prompts/**/*.yml",
            "server/*.sh",
        ],
    },
    install_requires=server_dependencies,
    extras_require=extras_require,
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
        "Framework :: FastAPI",
    ],
    keywords="ai, memory, agent, llm, assistant, chatbot, multimodal, server, fastapi",
    entry_points={
        "console_scripts": [
            "mirix-server=mirix.server.rest_api:main",
        ],
    },
    license="Apache License 2.0",
    zip_safe=False,
)
