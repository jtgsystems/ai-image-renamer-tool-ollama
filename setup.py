"""Setup script for Optimal Image Renamer."""

from pathlib import Path

from setuptools import setup

# Read the README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="optimal-image-renamer",
    version="1.0.0",
    author="Optimal Image Renamer Contributors",
    description="AI-powered bulk image renaming using Ollama's LLaVA model with multi-GPU acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jtgsystems/ai-image-renamer-tool-ollama",
    py_modules=["OPTIMALIMAGERENAMER"],
    python_requires=">=3.8",
    install_requires=[
        "ollama>=0.1.0",
        "pillow>=10.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "mypy>=1.0.0",
            "ruff>=0.1.0",
            "pre-commit>=3.0.0",
            "bandit>=1.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "optimal-image-renamer=OPTIMALIMAGERENAMER:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Utilities",
        "License :: Public Domain",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    keywords="ai image renamer gpu ollama llava photo organizer seo automation vision ml",
)
