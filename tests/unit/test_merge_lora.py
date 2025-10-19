"""
Unit tests for LoRA adapter merging functionality.

Tests the merge_lora.py module including:
- Merging LoRA adapters with base models
- Saving merged models locally
- Uploading to HuggingFace Hub
- Error handling for invalid paths
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from late.engine.merge_lora import merge_lora_adapter, merge_lora_from_config


class TestLoRAMerging:
    """Tests for basic LoRA merging functionality."""

    @patch('late.engine.merge_lora.AutoModelForCausalLM')
    @patch('late.engine.merge_lora.AutoTokenizer')
    @patch('late.engine.merge_lora.PeftModel')
    def test_merge_lora_adapter_basic(self, mock_peft, mock_tokenizer, mock_model, tmp_path):
        """Test basic LoRA adapter merging."""
        # Setup mocks
        base_model_mock = Mock()
        mock_model.from_pretrained.return_value = base_model_mock

        merged_model_mock = Mock()
        merged_model_mock.merge_and_unload.return_value = merged_model_mock
        mock_peft.from_pretrained.return_value = merged_model_mock

        tokenizer_mock = Mock()
        mock_tokenizer.from_pretrained.return_value = tokenizer_mock

        # Create fake adapter directory
        adapter_path = tmp_path / "adapter"
        adapter_path.mkdir()
        (adapter_path / "adapter_config.json").write_text("{}")

        local_save_path = tmp_path / "merged"

        # Run merge
        result = merge_lora_adapter(
            adapter_path=str(adapter_path),
            base_model_name="test/base-model",
            output_repo_id="test/merged-model",
            local_save_path=str(local_save_path),
            upload_to_hub=False
        )

        # Verify model loading
        mock_model.from_pretrained.assert_called_once()
        mock_peft.from_pretrained.assert_called_once_with(base_model_mock, str(adapter_path))

        # Verify merge
        merged_model_mock.merge_and_unload.assert_called_once()

        # Verify save
        merged_model_mock.save_pretrained.assert_called_once()
        tokenizer_mock.save_pretrained.assert_called_once()

        # Verify result
        assert result == str(local_save_path)

    def test_merge_invalid_adapter_path(self):
        """Test merging with non-existent adapter path."""
        with pytest.raises(FileNotFoundError):
            merge_lora_adapter(
                adapter_path="/nonexistent/adapter",
                base_model_name="test/model",
                output_repo_id="test/output",
                upload_to_hub=False
            )

    @patch('late.engine.merge_lora.AutoModelForCausalLM')
    @patch('late.engine.merge_lora.AutoTokenizer')
    @patch('late.engine.merge_lora.PeftModel')
    @patch('late.engine.merge_lora.HfApi')
    def test_merge_with_hub_upload(self, mock_api, mock_peft, mock_tokenizer, mock_model, tmp_path):
        """Test merging with HuggingFace Hub upload."""
        # Setup mocks
        base_model_mock = Mock()
        mock_model.from_pretrained.return_value = base_model_mock

        merged_model_mock = Mock()
        merged_model_mock.merge_and_unload.return_value = merged_model_mock
        mock_peft.from_pretrained.return_value = merged_model_mock

        tokenizer_mock = Mock()
        mock_tokenizer.from_pretrained.return_value = tokenizer_mock

        api_mock = Mock()
        mock_api.return_value = api_mock

        # Create fake adapter directory
        adapter_path = tmp_path / "adapter"
        adapter_path.mkdir()
        (adapter_path / "adapter_config.json").write_text("{}")

        local_save_path = tmp_path / "merged"

        # Run merge with upload
        result = merge_lora_adapter(
            adapter_path=str(adapter_path),
            base_model_name="test/base-model",
            output_repo_id="test/merged-model",
            local_save_path=str(local_save_path),
            upload_to_hub=True,
            private_repo=False
        )

        # Verify Hub upload was called
        api_mock.create_repo.assert_called_once()
        api_mock.upload_folder.assert_called_once()

        # Verify result is Hub URL
        assert "huggingface.co/test/merged-model" in result

    @patch('late.engine.merge_lora.AutoModelForCausalLM')
    @patch('late.engine.merge_lora.AutoTokenizer')
    @patch('late.engine.merge_lora.PeftModel')
    @patch('late.engine.merge_lora.HfApi')
    def test_merge_with_private_repo(self, mock_api, mock_peft, mock_tokenizer, mock_model, tmp_path):
        """Test merging with private repository."""
        # Setup mocks
        base_model_mock = Mock()
        mock_model.from_pretrained.return_value = base_model_mock

        merged_model_mock = Mock()
        merged_model_mock.merge_and_unload.return_value = merged_model_mock
        mock_peft.from_pretrained.return_value = merged_model_mock

        tokenizer_mock = Mock()
        mock_tokenizer.from_pretrained.return_value = tokenizer_mock

        api_mock = Mock()
        mock_api.return_value = api_mock

        # Create fake adapter directory
        adapter_path = tmp_path / "adapter"
        adapter_path.mkdir()
        (adapter_path / "adapter_config.json").write_text("{}")

        # Run merge with private repo
        merge_lora_adapter(
            adapter_path=str(adapter_path),
            base_model_name="test/base-model",
            output_repo_id="test/private-model",
            local_save_path=str(tmp_path / "merged"),
            upload_to_hub=True,
            private_repo=True
        )

        # Verify private flag was passed
        call_args = api_mock.create_repo.call_args
        assert call_args[1]['private'] == True


class TestLoRAMergeFromConfig:
    """Tests for merging using configuration files."""

    @patch('late.engine.merge_lora.merge_lora_adapter')
    def test_merge_from_config_file(self, mock_merge, tmp_path):
        """Test merging using a YAML config file."""
        import yaml

        # Create config file
        config_data = {
            'adapter_path': '/path/to/adapter',
            'base_model': 'test/base-model',
            'output_repo_id': 'test/merged-model',
            'local_save_path': '/tmp/merged',
            'upload_to_hub': True,
            'private_repo': False
        }

        config_file = tmp_path / "merge_config.yml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        # Run merge from config
        merge_lora_from_config(str(config_file))

        # Verify merge_lora_adapter was called with correct args
        mock_merge.assert_called_once_with(
            adapter_path='/path/to/adapter',
            base_model_name='test/base-model',
            output_repo_id='test/merged-model',
            local_save_path='/tmp/merged',
            upload_to_hub=True,
            private_repo=False,
            hf_token=None
        )

    def test_merge_from_config_missing_required_fields(self, tmp_path):
        """Test error handling for missing required fields in config."""
        import yaml

        # Config missing required field
        config_data = {
            'adapter_path': '/path/to/adapter',
            'base_model': 'test/base-model',
            # Missing output_repo_id
        }

        config_file = tmp_path / "incomplete_config.yml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        # Should raise ValueError
        with pytest.raises(ValueError, match="Missing required field"):
            merge_lora_from_config(str(config_file))

    @patch('late.engine.merge_lora.merge_lora_adapter')
    def test_merge_from_config_with_defaults(self, mock_merge, tmp_path):
        """Test that config uses default values when not specified."""
        import yaml

        # Minimal config (only required fields)
        config_data = {
            'adapter_path': '/path/to/adapter',
            'base_model': 'test/base-model',
            'output_repo_id': 'test/merged-model',
        }

        config_file = tmp_path / "minimal_config.yml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        # Run merge from config
        merge_lora_from_config(str(config_file))

        # Verify defaults were used
        call_args = mock_merge.call_args[1]
        assert call_args['upload_to_hub'] == True  # Default
        assert call_args['private_repo'] == False  # Default


@pytest.mark.unit
class TestLoRAMergeErrorHandling:
    """Tests for error handling in LoRA merging."""

    @patch('late.engine.merge_lora.AutoModelForCausalLM')
    @patch('late.engine.merge_lora.AutoTokenizer')
    @patch('late.engine.merge_lora.PeftModel')
    @patch('late.engine.merge_lora.HfApi')
    def test_merge_handles_upload_failure(self, mock_api, mock_peft, mock_tokenizer, mock_model, tmp_path):
        """Test that merge handles upload failures gracefully."""
        # Setup mocks
        base_model_mock = Mock()
        mock_model.from_pretrained.return_value = base_model_mock

        merged_model_mock = Mock()
        merged_model_mock.merge_and_unload.return_value = merged_model_mock
        mock_peft.from_pretrained.return_value = merged_model_mock

        tokenizer_mock = Mock()
        mock_tokenizer.from_pretrained.return_value = tokenizer_mock

        # Make upload fail
        api_mock = Mock()
        api_mock.upload_folder.side_effect = Exception("Upload failed")
        mock_api.return_value = api_mock

        # Create fake adapter directory
        adapter_path = tmp_path / "adapter"
        adapter_path.mkdir()
        (adapter_path / "adapter_config.json").write_text("{}")

        local_save_path = tmp_path / "merged"

        # Run merge (should not raise, but return local path)
        result = merge_lora_adapter(
            adapter_path=str(adapter_path),
            base_model_name="test/base-model",
            output_repo_id="test/merged-model",
            local_save_path=str(local_save_path),
            upload_to_hub=True
        )

        # Should still save locally even if upload fails
        merged_model_mock.save_pretrained.assert_called_once()

        # Should return local path, not Hub URL
        assert result == str(local_save_path)

    @patch('late.engine.merge_lora.HfFolder')
    def test_merge_with_hf_token(self, mock_folder, tmp_path):
        """Test that HF token is properly set."""
        adapter_path = tmp_path / "adapter"
        adapter_path.mkdir()
        (adapter_path / "adapter_config.json").write_text("{}")

        with patch('late.engine.merge_lora.AutoModelForCausalLM'), \
             patch('late.engine.merge_lora.AutoTokenizer'), \
             patch('late.engine.merge_lora.PeftModel'):

            # Test with explicit token
            merge_lora_adapter(
                adapter_path=str(adapter_path),
                base_model_name="test/model",
                output_repo_id="test/output",
                local_save_path=str(tmp_path / "merged"),
                upload_to_hub=False,
                hf_token="test_token_123"
            )

            # Verify token was saved
            mock_folder.save_token.assert_called_with("test_token_123")


@pytest.mark.unit
class TestLoRAMergeIntegration:
    """Integration-style tests for LoRA merging (with mocking)."""

    @patch('late.engine.merge_lora.AutoModelForCausalLM')
    @patch('late.engine.merge_lora.AutoTokenizer')
    @patch('late.engine.merge_lora.PeftModel')
    def test_full_merge_workflow(self, mock_peft, mock_tokenizer, mock_model, tmp_path):
        """Test complete merge workflow from adapter to saved model."""
        # Setup mocks
        base_model_mock = Mock()
        mock_model.from_pretrained.return_value = base_model_mock

        peft_model_mock = Mock()
        merged_model_mock = Mock()
        peft_model_mock.merge_and_unload.return_value = merged_model_mock
        mock_peft.from_pretrained.return_value = peft_model_mock

        tokenizer_mock = Mock()
        mock_tokenizer.from_pretrained.return_value = tokenizer_mock

        # Create adapter directory with config
        adapter_path = tmp_path / "adapter"
        adapter_path.mkdir()
        (adapter_path / "adapter_config.json").write_text('{"r": 64, "lora_alpha": 128}')
        (adapter_path / "adapter_model.safetensors").write_bytes(b"fake model weights")

        local_save_path = tmp_path / "merged_model"

        # Run merge
        result = merge_lora_adapter(
            adapter_path=str(adapter_path),
            base_model_name="meta-llama/Llama-3.2-3B-Instruct",
            output_repo_id="test-user/merged-llama",
            local_save_path=str(local_save_path),
            upload_to_hub=False
        )

        # Verify complete workflow
        assert mock_model.from_pretrained.called
        assert mock_peft.from_pretrained.called
        assert peft_model_mock.merge_and_unload.called
        assert merged_model_mock.save_pretrained.called
        assert tokenizer_mock.save_pretrained.called

        # Verify result
        assert result == str(local_save_path)
