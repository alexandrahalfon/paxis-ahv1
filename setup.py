from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="exueed",
    version="0.1.0",
    author="Your Name",
    description="Knowledge base system for PDF document processing and RAG",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
    ],
    extras_require={
        "api": ["fastapi>=0.104.0", "uvicorn[standard]>=0.24.0"],
        "processing": ["pdf2image>=1.16.0", "mistralai>=1.0.0", "openai>=1.0.0"],
        "ingestion": ["qdrant-client>=1.7.0", "tiktoken>=0.5.0"],
        "dev": ["pytest>=7.4.0", "black>=23.0.0", "isort>=5.12.0"],
    },
    entry_points={
        "console_scripts": [
            "exueed=src.cli.main:main",
        ],
    },
)
