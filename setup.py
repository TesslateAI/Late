from setuptools import setup, find_packages

setup(
    name="late",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "click",
        "pyyaml",
        "flask",
        "waitress",
        "transformers>=4.40.0",
        "datasets",
        "accelerate",
        "trl",
        "peft",
        "wandb",
        "torch",
        "Jinja2"
    ],
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
    url="https://github.com/your-username/late",
)