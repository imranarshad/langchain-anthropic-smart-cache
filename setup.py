"""Setup configuration for langchain-anthropic-smart-cache package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="langchain-anthropic-smart-cache",
    version="0.1.0",
    author="Imran Arshad",
    author_email="imran.arshad01@gmail.com",
    description="Intelligent cache management for LangChain Anthropic models with advanced optimization strategies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/imranarshad/langchain-anthropic-smart-cache",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "langchain-core>=0.1.0",
        "tiktoken>=0.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "anthropic": ["langchain-anthropic>=0.1.0"],
    },
    keywords="langchain, cache, anthropic, claude, optimization, ai, llm",
    project_urls={
        "Bug Reports": "https://github.com/imranarshad/langchain-anthropic-smart-cache/issues",
        "Source": "https://github.com/imranarshad/langchain-anthropic-smart-cache",
        "Documentation": "https://github.com/imranarshad/langchain-anthropic-smart-cache#readme",
    },
)