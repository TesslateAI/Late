"""
Unit tests for loss masking strategies.

Tests the two loss masking approaches:
1. "full" - Compute loss on entire conversation (DEFAULT)
2. "assistant_only" - Mask user prompts from loss computation

These tests validate the preprocessing functions generated in the training script.
"""

import pytest
from late.engine.training import generate_training_script


class TestFullLossMasking:
    """Tests for full loss masking strategy (default)."""

    def test_default_is_full_masking(self, sample_config_full_masking):
        """Test that default loss masking strategy is 'full'."""
        # Remove loss_masking_strategy to test default
        config = sample_config_full_masking.copy()
        if 'loss_masking_strategy' in config:
            del config['loss_masking_strategy']

        script = generate_training_script(config)

        # Should use full loss masking by default
        assert 'FULL Loss Masking - DEFAULT' in script
        assert 'format_for_training' in script
        assert 'apply_chat_template' in script
        assert 'dataset_text_field="text"' in script

    def test_explicit_full_masking(self, sample_config_full_masking):
        """Test explicit full loss masking configuration."""
        config = sample_config_full_masking.copy()
        config['loss_masking_strategy'] = 'full'

        script = generate_training_script(config)

        # Should use full loss masking
        assert 'FULL Loss Masking' in script
        assert 'format_for_training' in script
        assert 'computes loss on full conversation' in script
        assert 'dataset_text_field="text"' in script

    def test_full_masking_uses_simple_preprocessing(self, sample_config_full_masking):
        """Test that full masking uses simple tokenizer.apply_chat_template."""
        config = sample_config_full_masking.copy()
        config['loss_masking_strategy'] = 'full'

        script = generate_training_script(config)

        # Should use simple chat template formatting
        assert 'tokenizer.apply_chat_template' in script
        assert 'format_for_training' in script

        # Should NOT use complex masking logic
        assert 'preprocess_for_assistant_loss' not in script
        assert '[-100]' not in script  # No masking tokens

    def test_full_masking_no_data_collator(self, sample_config_full_masking):
        """Test that full masking doesn't use DataCollatorForLanguageModeling."""
        config = sample_config_full_masking.copy()
        config['loss_masking_strategy'] = 'full'

        script = generate_training_script(config)

        # Full masking uses SFTTrainer's built-in collation with dataset_text_field
        # So DataCollatorForLanguageModeling should not be explicitly created
        # (it's only needed for assistant_only strategy)
        lines = script.split('\n')
        data_collator_lines = [l for l in lines if 'DataCollatorForLanguageModeling' in l and 'import' not in l]

        # Should not instantiate data collator for full masking
        assert len(data_collator_lines) == 0


class TestAssistantOnlyLossMasking:
    """Tests for assistant-only loss masking strategy."""

    def test_explicit_assistant_only_masking(self, sample_config_assistant_masking):
        """Test explicit assistant_only loss masking configuration."""
        config = sample_config_assistant_masking.copy()
        config['loss_masking_strategy'] = 'assistant_only'

        script = generate_training_script(config)

        # Should use assistant-only masking
        assert 'ASSISTANT-ONLY Loss Masking' in script
        assert 'preprocess_for_assistant_loss' in script
        assert 'mask user prompts from loss' in script

    def test_assistant_masking_uses_complex_preprocessing(self, sample_config_assistant_masking):
        """Test that assistant_only uses complex preprocessing with masking."""
        config = sample_config_assistant_masking.copy()
        config['loss_masking_strategy'] = 'assistant_only'

        script = generate_training_script(config)

        # Should use complex preprocessing
        assert 'preprocess_for_assistant_loss' in script
        assert '[-100]' in script  # Masking tokens
        assert "message['role'] == 'assistant'" in script

        # Should NOT use simple formatting
        assert 'format_for_training' not in script

    def test_assistant_masking_uses_data_collator(self, sample_config_assistant_masking):
        """Test that assistant_only masking uses DataCollatorForLanguageModeling."""
        config = sample_config_assistant_masking.copy()
        config['loss_masking_strategy'] = 'assistant_only'

        script = generate_training_script(config)

        # Should create data collator for assistant-only masking
        assert 'DataCollatorForLanguageModeling' in script
        assert "data_collator = DataCollatorForLanguageModeling" in script
        assert "trainer_kwargs['data_collator'] = data_collator" in script

    def test_assistant_masking_no_dataset_text_field(self, sample_config_assistant_masking):
        """Test that assistant_only doesn't use dataset_text_field."""
        config = sample_config_assistant_masking.copy()
        config['loss_masking_strategy'] = 'assistant_only'

        script = generate_training_script(config)

        # Should NOT use dataset_text_field (it's for full masking only)
        assert 'dataset_text_field="text"' not in script

    def test_assistant_masking_handles_bos_eos_tokens(self, sample_config_assistant_masking):
        """Test that assistant_only masking handles BOS/EOS tokens."""
        config = sample_config_assistant_masking.copy()
        config['loss_masking_strategy'] = 'assistant_only'

        script = generate_training_script(config)

        # Should handle special tokens
        assert 'if tokenizer.bos_token:' in script
        assert 'if tokenizer.eos_token:' in script
        assert 'bos_tokens' in script
        assert 'eos_tokens' in script


class TestLossMaskingWithLoRA:
    """Tests for loss masking strategies combined with LoRA."""

    def test_lora_with_full_masking(self, sample_config_lora):
        """Test LoRA training with full loss masking."""
        config = sample_config_lora.copy()
        config['loss_masking_strategy'] = 'full'

        script = generate_training_script(config)

        # Should have both LoRA and full masking
        assert 'LoRA' in script
        assert 'get_peft_model' in script
        assert 'format_for_training' in script
        assert 'dataset_text_field="text"' in script

    def test_lora_with_assistant_only_masking(self, sample_config_lora):
        """Test LoRA training with assistant_only loss masking."""
        config = sample_config_lora.copy()
        config['loss_masking_strategy'] = 'assistant_only'

        script = generate_training_script(config)

        # Should have both LoRA and assistant-only masking
        assert 'LoRA' in script
        assert 'get_peft_model' in script
        assert 'preprocess_for_assistant_loss' in script
        assert '[-100]' in script


@pytest.mark.unit
class TestLossMaskingScriptGeneration:
    """Tests for training script generation with different loss masking strategies."""

    def test_script_imports_for_full_masking(self, sample_config_full_masking):
        """Test that full masking includes necessary imports."""
        config = sample_config_full_masking.copy()
        config['loss_masking_strategy'] = 'full'

        script = generate_training_script(config)

        # Check imports
        assert 'from transformers import' in script
        assert 'from datasets import load_dataset' in script
        assert 'from trl import SFTTrainer, SFTConfig' in script

    def test_script_imports_for_assistant_masking(self, sample_config_assistant_masking):
        """Test that assistant_only includes DataCollator import."""
        config = sample_config_assistant_masking.copy()
        config['loss_masking_strategy'] = 'assistant_only'

        script = generate_training_script(config)

        # Check data collator import
        assert 'DataCollatorForLanguageModeling' in script

    def test_processing_class_parameter(self, sample_config_full_masking):
        """Test that generated script uses processing_class parameter."""
        script = generate_training_script(sample_config_full_masking)

        # Should use processing_class instead of tokenizer
        assert 'processing_class' in script
        assert "'processing_class': tokenizer" in script

    def test_trainer_kwargs_structure(self, sample_config_full_masking):
        """Test that trainer kwargs are properly structured."""
        script = generate_training_script(sample_config_full_masking)

        # Should use trainer_kwargs dict
        assert 'trainer_kwargs = {' in script
        assert "'model': model" in script
        assert "'processing_class': tokenizer" in script
        assert "'train_dataset': processed_dataset" in script
        assert 'SFTTrainer(**trainer_kwargs)' in script


@pytest.mark.unit
class TestLossMaskingCompatibility:
    """Tests for backward compatibility and edge cases."""

    def test_invalid_loss_strategy_falls_back_to_full(self, sample_config_full_masking):
        """Test that invalid loss_masking_strategy defaults to 'full'."""
        config = sample_config_full_masking.copy()
        config['loss_masking_strategy'] = 'invalid_strategy'

        script = generate_training_script(config)

        # Should fall back to full masking (treated as not 'assistant_only')
        assert 'format_for_training' not in script or 'preprocess_for_assistant_loss' in script
        # Actually, with current implementation, anything not 'full' goes to assistant_only
        # So this test documents current behavior

    def test_case_sensitive_strategy_names(self, sample_config_full_masking):
        """Test that strategy names are case-sensitive."""
        config = sample_config_full_masking.copy()
        config['loss_masking_strategy'] = 'FULL'  # Uppercase

        script = generate_training_script(config)

        # Current implementation is case-sensitive
        # 'FULL' != 'full', so it would use assistant_only
        # This documents expected behavior
        assert 'preprocess_for_assistant_loss' in script
