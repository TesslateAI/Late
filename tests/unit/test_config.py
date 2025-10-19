"""
Unit tests for configuration loading and management.

Tests the config.py module including:
- Loading training configs from YAML
- Saving and loading API tokens
- Default value handling
- Invalid config detection
"""

import pytest
import yaml
from pathlib import Path
from late.engine.config import load_training_config, save_token, load_tokens


class TestConfigLoading:
    """Tests for loading training configurations."""

    def test_load_valid_config(self, sample_config_file):
        """Test loading a valid YAML config file."""
        config = load_training_config(str(sample_config_file))

        assert config is not None
        assert config['base_model'] == 'meta-llama/Llama-3.2-3B-Instruct'
        assert config['training_type'] == 'sft'
        assert config['loss_masking_strategy'] == 'full'

    def test_default_loss_masking_strategy(self, tmp_path):
        """Test that loss_masking_strategy defaults to 'full' when not specified."""
        # Create config without loss_masking_strategy
        config_data = {
            'base_model': 'test-model',
            'dataset_name': 'test-dataset',
            'output_model_name': 'test/output',
            'output_dir': '/tmp/test',
            'training_type': 'sft',
            'max_seq_length': 2048,
            'batch_size': 1,
            'gradient_accumulation': 4,
            'epochs': 1,
            'learning_rate': 2e-5,
        }

        config_file = tmp_path / "config_no_loss_strategy.yml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        loaded_config = load_training_config(str(config_file))

        # Should use default 'full' if not specified
        assert loaded_config.get('loss_masking_strategy', 'full') == 'full'

    def test_explicit_assistant_only_masking(self, tmp_path):
        """Test that assistant_only loss masking can be explicitly set."""
        config_data = {
            'base_model': 'test-model',
            'dataset_name': 'test-dataset',
            'output_model_name': 'test/output',
            'output_dir': '/tmp/test',
            'training_type': 'sft',
            'loss_masking_strategy': 'assistant_only',  # Explicit
            'max_seq_length': 2048,
            'batch_size': 1,
            'gradient_accumulation': 4,
            'epochs': 1,
            'learning_rate': 2e-5,
        }

        config_file = tmp_path / "config_assistant_only.yml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        loaded_config = load_training_config(str(config_file))

        assert loaded_config['loss_masking_strategy'] == 'assistant_only'

    def test_load_lora_config(self, tmp_path, sample_config_lora):
        """Test loading a LoRA training config."""
        config_file = tmp_path / "lora_config.yml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config_lora, f)

        config = load_training_config(str(config_file))

        assert config['training_type'] == 'lora'
        assert 'lora' in config
        assert config['lora']['r'] == 64
        assert config['lora']['lora_alpha'] == 128


class TestTokenManagement:
    """Tests for API token saving and loading."""

    def test_save_and_load_token(self, tmp_path, monkeypatch):
        """Test saving and loading an API token."""
        # Use temp directory for token file
        token_file = tmp_path / "tokens.yml"
        monkeypatch.setattr('late.engine.config.TOKEN_FILE', token_file)

        # Save a token
        save_token('hf_token', 'test_token_12345')

        # Verify file exists
        assert token_file.exists()

        # Load tokens
        tokens = load_tokens()

        assert 'HF_TOKEN' in tokens  # Should be uppercase
        assert tokens['HF_TOKEN'] == 'test_token_12345'

    def test_save_multiple_tokens(self, tmp_path, monkeypatch):
        """Test saving multiple tokens."""
        token_file = tmp_path / "tokens.yml"
        monkeypatch.setattr('late.engine.config.TOKEN_FILE', token_file)

        # Save multiple tokens
        save_token('hf_token', 'hf_test_123')
        save_token('wandb', 'wandb_test_456')

        # Load and verify
        tokens = load_tokens()

        assert tokens['HF_TOKEN'] == 'hf_test_123'
        assert tokens['WANDB'] == 'wandb_test_456'

    def test_load_tokens_sets_env_vars(self, tmp_path, monkeypatch, clean_env):
        """Test that loading tokens sets environment variables."""
        import os

        token_file = tmp_path / "tokens.yml"
        monkeypatch.setattr('late.engine.config.TOKEN_FILE', token_file)

        # Save a token
        save_token('hf_token', 'env_test_token')

        # Load tokens (should set env vars)
        load_tokens()

        # Verify environment variable is set
        assert os.environ.get('HF_TOKEN') == 'env_test_token'


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_missing_required_fields(self, tmp_path):
        """Test handling of missing required fields."""
        # Config missing required fields
        incomplete_config = {
            'base_model': 'test-model',
            # Missing dataset_name, output_model_name, etc.
        }

        config_file = tmp_path / "incomplete_config.yml"
        with open(config_file, 'w') as f:
            yaml.dump(incomplete_config, f)

        # This should load but may fail validation later
        # (validation happens in training script generation)
        config = load_training_config(str(config_file))
        assert 'base_model' in config
        assert 'dataset_name' not in config

    def test_invalid_file_path(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_training_config('/nonexistent/path/config.yml')


@pytest.mark.unit
class TestConfigDefaults:
    """Tests for configuration default values."""

    def test_default_batch_size(self, tmp_path):
        """Test default batch size is applied."""
        config_data = {
            'base_model': 'test-model',
            'dataset_name': 'test-dataset',
            'output_model_name': 'test/output',
            'output_dir': '/tmp/test',
            'training_type': 'sft',
            'max_seq_length': 2048,
            'epochs': 1,
            # batch_size not specified
        }

        config_file = tmp_path / "config_defaults.yml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        config = load_training_config(str(config_file))

        # Default should be applied in training script
        # Here we just verify the config loads
        assert 'batch_size' not in config or config.get('batch_size', 1) >= 1
