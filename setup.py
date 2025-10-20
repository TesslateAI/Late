from setuptools import setup, find_packages
from pathlib import Path

# Read README with explicit UTF-8 encoding (fixes Windows cp1252 issue)
this_directory = Path(__file__).parent
try:
    long_description = (this_directory / "README.md").read_text(encoding='utf-8')
except Exception:
    long_description = "Late: A CLI and web UI for scheduling and running training jobs on ROCm servers."

# Core dependencies for the lightweight CLI tool
core_deps = [
    "click",
    "pyyaml",
    "flask",
    "waitress",
    "requests",
    "Jinja2",
    "matplotlib",
    "openpyxl",
    "pandas"
]

# Optional dependencies for a full development/training environment
extras = {
    "training": [
        "torch",
        "transformers>=4.56.2",
        "datasets",
        "accelerate",
        "trl",
        "peft",
        "wandb",
        "scipy",
        "numpy",
        "ninja",
        "cmake",
        "pybind11"
    ],
    # Note: Unsloth for AMD GPUs must be installed separately using:
    # pip install --no-deps unsloth unsloth-zoo
    # pip install --no-deps git+https://github.com/unslothai/unsloth-zoo.git
    # pip install "unsloth[amd] @ git+https://github.com/unslothai/unsloth"
    "test": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-mock>=3.10.0",
        "pytest-asyncio>=0.21.0",
        "pytest-xdist>=3.0.0",
        "pytest-timeout>=2.1.0",
        "responses>=0.23.0",
        "hypothesis>=6.0.0",
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-mock>=3.10.0",
        "pytest-xdist>=3.0.0",
        "black>=23.0.0",
        "ruff>=0.1.0",
        "mypy>=1.0.0",
        "pre-commit>=3.0.0",
    ],
}

setup(
    name="late",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=core_deps,
    extras_require=extras,
    entry_points={
        "console_scripts": [
            "late = late.cli:cli",
        ],
    },
    package_data={
        "late": [
            "server/templates/*.html",
            "server/static/*.css",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A CLI and web UI for scheduling and running training jobs on ROCm servers.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/TesslateAI/Late",
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)