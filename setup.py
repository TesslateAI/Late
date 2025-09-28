from setuptools import setup, find_packages

# Core dependencies for the lightweight CLI tool
core_deps = [
    "click",
    "pyyaml",
    "flask",
    "waitress",
    "requests",
    "Jinja2"
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
    ]
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
    author="Your Name",
    author_email="your.email@example.com",
    description="A CLI and web UI for scheduling and running training jobs on ROCm servers.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/TesslateAI/Late",
)