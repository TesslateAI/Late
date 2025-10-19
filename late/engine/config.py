import os
import yaml
from pathlib import Path

# Path to store global settings like tokens
CONFIG_DIR = Path.home() / ".late"
TOKEN_FILE = CONFIG_DIR / "tokens.yml"

def ensure_config_dir():
    """Ensures the ~/.late directory exists."""
    CONFIG_DIR.mkdir(exist_ok=True)

def load_training_config(path: str) -> dict:
    """Loads a training YAML file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_token(key: str, value: str):
    """Saves a token (e.g., wandb, hf_token) to the global config."""
    ensure_config_dir()
    tokens = {}
    if TOKEN_FILE.exists():
        with open(TOKEN_FILE, 'r', encoding='utf-8') as f:
            tokens = yaml.safe_load(f) or {}

    tokens[key.upper()] = value

    with open(TOKEN_FILE, 'w', encoding='utf-8') as f:
        yaml.dump(tokens, f)
    print(f"[OK] Saved {key.upper()} to {TOKEN_FILE}")

def load_tokens() -> dict:
    """Loads all saved tokens into environment variables."""
    if not TOKEN_FILE.exists():
        return {}

    with open(TOKEN_FILE, 'r', encoding='utf-8') as f:
        tokens = yaml.safe_load(f)

    for key, value in tokens.items():
        os.environ[key] = value
    return tokens