"""
Pytest configuration and fixtures for Late testing.

This module provides shared fixtures for testing the Late library,
including mock models, tokenizers, configs, and datasets.
"""

import pytest
import os
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, MagicMock


# ============================================================================
# Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for test runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_model_cache():
    """Create a temporary model cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Sample Configuration Fixtures
# ============================================================================

@pytest.fixture
def sample_config_full_masking() -> Dict[str, Any]:
    """Sample config with full loss masking (default)."""
    return {
        'base_model': 'meta-llama/Llama-3.2-3B-Instruct',
        'dataset_name': 'yahma/alpaca-cleaned',
        'output_model_name': 'test-user/test-model-full',
        'output_dir': '/tmp/test_output',
        'training_type': 'sft',
        'loss_masking_strategy': 'full',  # DEFAULT
        'max_seq_length': 2048,
        'batch_size': 1,
        'gradient_accumulation': 4,
        'epochs': 1,
        'learning_rate': 2e-5,
        'cache_dir': '/tmp/model_cache',
        'report_to_wandb': False,
        'upload_to_hub': False,
    }


@pytest.fixture
def sample_config_assistant_masking() -> Dict[str, Any]:
    """Sample config with assistant-only loss masking."""
    return {
        'base_model': 'meta-llama/Llama-3.2-3B-Instruct',
        'dataset_name': 'yahma/alpaca-cleaned',
        'output_model_name': 'test-user/test-model-assistant',
        'output_dir': '/tmp/test_output',
        'training_type': 'sft',
        'loss_masking_strategy': 'assistant_only',  # EXPLICIT
        'max_seq_length': 2048,
        'batch_size': 1,
        'gradient_accumulation': 4,
        'epochs': 1,
        'learning_rate': 2e-5,
        'cache_dir': '/tmp/model_cache',
        'report_to_wandb': False,
        'upload_to_hub': False,
    }


@pytest.fixture
def sample_config_lora() -> Dict[str, Any]:
    """Sample LoRA training config."""
    return {
        'base_model': 'meta-llama/Llama-3.2-3B-Instruct',
        'dataset_name': 'yahma/alpaca-cleaned',
        'output_model_name': 'test-user/test-lora',
        'output_dir': '/tmp/test_output_lora',
        'training_type': 'lora',
        'loss_masking_strategy': 'full',
        'max_seq_length': 2048,
        'batch_size': 1,
        'gradient_accumulation': 4,
        'epochs': 1,
        'learning_rate': 2e-4,
        'lora': {
            'r': 64,
            'lora_alpha': 128,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        },
        'cache_dir': '/tmp/model_cache',
        'report_to_wandb': False,
        'upload_to_hub': False,
    }


@pytest.fixture
def sample_config_with_ntfy() -> Dict[str, Any]:
    """Sample config with ntfy notifications enabled."""
    return {
        'base_model': 'meta-llama/Llama-3.2-3B-Instruct',
        'dataset_name': 'yahma/alpaca-cleaned',
        'output_model_name': 'test-user/test-model-ntfy',
        'output_dir': '/tmp/test_output',
        'training_type': 'sft',
        'loss_masking_strategy': 'full',
        'max_seq_length': 2048,
        'batch_size': 1,
        'gradient_accumulation': 4,
        'epochs': 1,
        'learning_rate': 2e-5,
        'ntfy_topic': 'test-topic',
        'cache_dir': '/tmp/model_cache',
        'report_to_wandb': False,
        'upload_to_hub': False,
    }


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_messages() -> list:
    """Sample chat messages for testing."""
    return [
        {'role': 'user', 'content': 'Hello, how are you?'},
        {'role': 'assistant', 'content': 'I am doing well, thank you for asking!'},
        {'role': 'user', 'content': 'Can you help me with Python?'},
        {'role': 'assistant', 'content': 'Of course! I would be happy to help you with Python programming.'}
    ]


@pytest.fixture
def sample_dataset_dict(sample_messages) -> Dict[str, Any]:
    """Sample dataset in HuggingFace format."""
    return {
        'messages': [
            sample_messages,
            [
                {'role': 'user', 'content': 'What is machine learning?'},
                {'role': 'assistant', 'content': 'Machine learning is a subset of artificial intelligence...'}
            ]
        ]
    }


# ============================================================================
# Mock Model and Tokenizer Fixtures
# ============================================================================

@pytest.fixture
def mock_tokenizer():
    """Mock HuggingFace tokenizer."""
    tokenizer = Mock()
    tokenizer.pad_token = '<pad>'
    tokenizer.eos_token = '</s>'
    tokenizer.bos_token = '<s>'
    tokenizer.padding_side = 'right'
    tokenizer.model_max_length = 2048

    # Mock encode method
    def mock_encode(text, add_special_tokens=False):
        # Simple mock: return list of integers based on text length
        return list(range(len(text.split())))

    tokenizer.encode = Mock(side_effect=mock_encode)

    # Mock apply_chat_template
    def mock_apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
        # Simple mock: join role and content
        if isinstance(messages, list) and len(messages) > 0:
            msg = messages[0] if len(messages) == 1 else messages
            if isinstance(msg, dict):
                return f"<|{msg['role']}|>{msg['content']}<|end|>"
            return " ".join([f"<|{m['role']}|>{m['content']}<|end|>" for m in messages])
        return ""

    tokenizer.apply_chat_template = Mock(side_effect=mock_apply_chat_template)

    # Mock __call__ for tokenization
    def mock_call(text, add_special_tokens=False, **kwargs):
        tokens = text.split()
        return {
            'input_ids': list(range(len(tokens))),
            'attention_mask': [1] * len(tokens)
        }

    tokenizer.__call__ = Mock(side_effect=mock_call)
    tokenizer.save_pretrained = Mock()

    return tokenizer


@pytest.fixture
def mock_model():
    """Mock HuggingFace model."""
    model = Mock()
    model.config = Mock()
    model.config.vocab_size = 32000
    model.config.hidden_size = 4096
    model.gradient_checkpointing_enable = Mock()
    model.save_pretrained = Mock()
    model.print_trainable_parameters = Mock()
    return model


@pytest.fixture
def mock_peft_model():
    """Mock PEFT model for LoRA testing."""
    model = Mock()
    model.merge_and_unload = Mock(return_value=model)
    model.save_pretrained = Mock()
    model.print_trainable_parameters = Mock()
    return model


# ============================================================================
# Sweep Configuration Fixtures
# ============================================================================

@pytest.fixture
def sample_sweep_config() -> Dict[str, Any]:
    """Sample sweep configuration."""
    return {
        'sweep_id': 'test_sweep_001',
        'sweep_parameters': {
            'learning_rate': [1e-4, 2e-4, 3e-4]
        },
        'overrides': {
            'max_seq_length': 1024
        },
        'early_stop': {
            'percent_epoch': 25
        }
    }


@pytest.fixture
def sample_multi_param_sweep() -> Dict[str, Any]:
    """Sample sweep with multiple parameters."""
    return {
        'sweep_id': 'test_sweep_multi',
        'sweep_parameters': {
            'learning_rate': [1e-4, 2e-4],
            'lora.r': [64, 128]
        },
        'early_stop': {
            'percent_epoch': 30
        }
    }


# ============================================================================
# File-based Fixtures
# ============================================================================

@pytest.fixture
def sample_config_file(tmp_path, sample_config_full_masking) -> Path:
    """Create a temporary YAML config file."""
    config_file = tmp_path / "test_config.yml"
    with open(config_file, 'w') as f:
        yaml.dump(sample_config_full_masking, f)
    return config_file


@pytest.fixture
def sample_sweep_file(tmp_path, sample_sweep_config) -> Path:
    """Create a temporary sweep config file."""
    sweep_file = tmp_path / "test_sweep.sweep"
    with open(sweep_file, 'w') as f:
        yaml.dump(sample_sweep_config, f)
    return sweep_file


# ============================================================================
# Mock HuggingFace API Fixtures
# ============================================================================

@pytest.fixture
def mock_hf_api():
    """Mock HuggingFace API."""
    api = Mock()
    api.create_repo = Mock()
    api.upload_folder = Mock()
    return api


# ============================================================================
# Mock Requests Fixtures (for ntfy testing)
# ============================================================================

@pytest.fixture
def mock_requests(monkeypatch):
    """Mock requests library for HTTP calls."""
    mock_post = Mock(return_value=Mock(status_code=200))
    monkeypatch.setattr('requests.post', mock_post)
    return mock_post


# ============================================================================
# Environment Fixtures
# ============================================================================

@pytest.fixture
def clean_env(monkeypatch):
    """Clean environment variables for testing."""
    # Remove any existing tokens
    monkeypatch.delenv('HF_TOKEN', raising=False)
    monkeypatch.delenv('WANDB_API_KEY', raising=False)
    monkeypatch.delenv('WANDB_PROJECT', raising=False)
    yield
    # Cleanup happens automatically with monkeypatch


@pytest.fixture
def hf_token_env(monkeypatch):
    """Set HF_TOKEN environment variable."""
    monkeypatch.setenv('HF_TOKEN', 'test_hf_token_12345')
    yield 'test_hf_token_12345'
