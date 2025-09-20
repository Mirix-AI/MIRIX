#!/usr/bin/env python3

import os
from setuptools import setup, find_packages

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

# Filter requirements - exclude some development/system specific packages
def filter_requirements(requirements):
    """Filter out packages that shouldn't be in install_requires."""
    exclude_patterns = [
        'ffmpeg',  # System dependency, not a pip package
    ]
    
    filtered = []
    for req in requirements:
        # Skip empty lines and comments
        if not req or req.startswith('#'):
            continue
        
        # Skip packages that match exclude patterns
        if any(pattern in req.lower() for pattern in exclude_patterns):
            continue
            
        filtered.append(req)
    
    return filtered

# Get version from mirix/__init__.py
def get_version():
    import re
    version_file = os.path.join(this_directory, 'mirix', '__init__.py')
    with open(version_file, 'r', encoding='utf-8') as f:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")

requirements = parse_requirements('requirements.txt')
install_requires = filter_requirements(requirements)

setup(
    name='mirix',
    version=get_version(),
    author='Mirix AI',
    author_email='yuwang@mirix.io',
    description='Multi-Agent Personal Assistant with an Advanced Memory System',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Mirix-AI/MIRIX',
    project_urls={
        'Documentation': 'https://docs.mirix.io',
        'Website': 'https://mirix.io',
        'Source Code': 'https://github.com/Mirix-AI/MIRIX',
        'Bug Reports': 'https://github.com/Mirix-AI/MIRIX/issues',
    },
    packages=find_packages(exclude=['tests*', 'scripts*', 'frontend*', 'public_evaluations*']),
    include_package_data=True,
    package_data={
        'mirix': [
            '*.yaml',
            '*.yml', 
            '*.txt',
            'configs/*.yaml',
            'configs/*.yml',
            'prompts/**/*.txt',
            'prompts/**/*.yaml',
            'prompts/**/*.yml',
        ],
    },
    install_requires=install_requires,
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-asyncio',
            'black',
            'isort',
            'flake8',
        ],
        'voice': [
            'SpeechRecognition',
            'pydub',
        ],
        'full': [
            'SpeechRecognition',
            'pydub',
        ]
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Communications :: Chat',
    ],
    keywords='ai, memory, agent, llm, assistant, chatbot, multimodal',
    entry_points={
        'console_scripts': [
            'mirix=mirix.__main__:main',
        ],
    },
    license='Apache License 2.0',
    zip_safe=False,
)
