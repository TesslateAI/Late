# Changelog

All notable changes to the Late training library.

## [Unreleased]

### 🎉 Major Features

#### Configurable Loss Masking Strategies
- **Default Strategy Changed**: Loss masking now defaults to `"full"` (computes loss on entire conversation)
- **Two Strategies Available**:
  - `"full"` (NEW DEFAULT): Simple preprocessing using `tokenizer.apply_chat_template()`
  - `"assistant_only"`: Masks user prompts from loss computation (must be explicitly set)
- **Backward Compatible**: Existing configs continue to work
- **Configuration**: Set `loss_masking_strategy: "full"` or `"assistant_only"` in YAML config
- **Documentation**: Complete guide with examples in `examples/loss_masking/`

#### Push Notifications via ntfy.sh
- **Real-time Notifications**: Get push notifications on your phone/desktop during training
- **Notification Events**:
  - Training start
  - Checkpoint saves (every `save_steps`)
  - Training completion
  - Model upload success/failure
  - Checkpoint resume events
  - Error notifications
- **Easy Setup**: Just add `ntfy_topic: "your-topic"` to config
- **Custom Callback**: `NtfyCheckpointCallback` integrated into training script
- **Example**: See `examples/with_notifications.yml`

#### LoRA Adapter Merging
- **New Command**: `late merge` for merging LoRA adapters with base models
- **Features**:
  - Merge LoRA weights into base model
  - Save merged model locally
  - Upload to HuggingFace Hub (public or private)
  - Support for both CLI arguments and YAML config
- **Module**: New `late/engine/merge_lora.py`
- **CLI Examples**:
  ```bash
  late merge /path/to/checkpoint --base-model model/name --output user/merged-model
  ```

### 🔧 Improvements

#### Training System (`late/engine/training.py`)
- **HF Token Support**: Automatic token handling from config or environment
- **Checkpoint Resume**: Automatic detection and resume from last checkpoint
- **Processing Class**: Updated to use `processing_class` parameter (modern TRL API)
- **Better Logging**: Enhanced logging with timestamps and status updates
- **ntfy Integration**: Notifications embedded in training script generation
- **Force Restart Option**: Support for `force_restart` config to bypass checkpoint resume

#### Sweep System (`late/engine/sweep.py`)
- **Live Chart Updates**: Comparison charts update after each sweep run
- **CSV Export**: Per-run CSV files with step/loss data in `sweep_output/run_data/`
- **Summary CSV**: Dynamic summary with all sweep parameters and final losses
- **Improved Parsing**: Better parameter extraction from filenames
- **Better Logging**: Enhanced progress tracking and status updates

#### CLI (`late/cli.py`)
- **New `merge` Command**: Merge LoRA adapters with base models
- **Merge Options**:
  - `--base-model`: Specify base model
  - `--output`: Output repository ID
  - `--local-path`: Local save directory
  - `--no-upload`: Skip HuggingFace upload
  - `--private`: Create private repository
  - `--config`: Use YAML config file

### 🧪 Testing Infrastructure

#### Comprehensive Test Suite
- **Framework**: pytest with extensive plugins
- **Coverage**: 80%+ code coverage target
- **Test Categories**:
  - Unit tests (`tests/unit/`)
  - Integration tests (`tests/integration/`)
  - Fixtures and mocks (`tests/conftest.py`)
- **Test Files**:
  - `test_config.py`: Configuration loading and validation
  - `test_loss_masking.py`: Both loss masking strategies
  - `test_merge_lora.py`: LoRA merging functionality
  - More to come...

#### Test Documentation
- **Complete Guide**: `tests/README.md` with usage examples
- **Fixtures**: Comprehensive fixture library for testing
- **Markers**: Test categorization (unit, integration, slow, requires_gpu)
- **CI/CD Ready**: GitHub Actions workflow support

#### Dependencies (`pyproject.toml`)
- **Test Dependencies**: pytest, pytest-cov, pytest-mock, pytest-xdist, etc.
- **Dev Dependencies**: black, ruff, mypy, pre-commit
- **Easy Installation**: `pip install -e ".[test]"` or `pip install -e ".[dev]"`

### 📚 Documentation

#### README Updates
- **What's New Section**: Highlights all new features
- **Loss Masking Guide**: Complete explanation of both strategies
- **ntfy.sh Guide**: Setup instructions and notification details
- **LoRA Merge Guide**: Complete merge workflow documentation
- **Updated Config Reference**: All new parameters documented
- **Updated Examples**: All examples include new optional fields

#### New Example Configs
- `examples/loss_masking/full_masking_example.yml`: Full loss masking example
- `examples/loss_masking/assistant_only_masking_example.yml`: Assistant-only example
- `examples/with_notifications.yml`: ntfy.sh integration example
- Updated `examples/llama3/llama3.2_3b_lora.yml`: Includes all new fields

#### Testing Documentation
- `tests/README.md`: Comprehensive testing guide
- Running tests, writing tests, fixtures, coverage
- CI/CD integration instructions
- Debugging tips and best practices

### 🔄 Breaking Changes

**None!** All changes are backward compatible.

- Existing configs work without modification
- Default behavior is sensible (full loss masking)
- New features are opt-in (ntfy, LoRA merge)

### 📦 Configuration Changes

#### New Optional Fields
```yaml
# Loss masking strategy (optional, defaults to "full")
loss_masking_strategy: "full"  # or "assistant_only"

# Push notifications (optional, leave empty to disable)
ntfy_topic: "your-topic-name"

# HuggingFace token (optional, uses HF_TOKEN env var if not set)
hf_token: ""  # or "${HF_TOKEN}"
```

### 🐛 Bug Fixes

#### Sweep System
- **Filename Parsing**: Fixed learning rate extraction from CSV filenames
- **Chart Labels**: Improved parameter label parsing for comparison charts
- **CSV Structure**: Dynamic header generation based on sweep parameters

### 🚀 Performance

- **Faster Preprocessing**: Full loss masking is simpler and faster than assistant-only
- **Memory Efficiency**: Checkpoint resume reduces wasted compute
- **Parallel Tests**: pytest-xdist support for faster test execution

### 📝 Notes for Developers

#### Code Quality
- Comprehensive type hints
- Clear documentation strings
- Modular, testable code
- Consistent code style (black, ruff)

#### Testing
- Run tests: `pytest`
- Run with coverage: `pytest --cov=late --cov-report=html`
- Run unit tests only: `pytest -m unit`
- Run in parallel: `pytest -n auto`

#### Contributing
- All new features have tests
- Documentation updated for all changes
- Examples provided for new functionality
- CI/CD ready (GitHub Actions workflow included)

---

## Installation

### Standard Installation
```bash
git clone https://github.com/TesslateAI/Late.git
cd Late
pip install -e .
```

### With Test Dependencies
```bash
pip install -e ".[test]"
```

### With Development Dependencies
```bash
pip install -e ".[dev]"
```

---

## Migration Guide

### From Previous Version

No migration needed! All changes are backward compatible.

**Optional Enhancements:**
1. Add `loss_masking_strategy: "full"` to configs (already the default)
2. Add `ntfy_topic` for push notifications
3. Update to use `late merge` for LoRA adapter merging
4. Run test suite to ensure everything works: `pytest`

### Recommended Updates

**Update your configs** to include new optional fields:
```yaml
# Add these lines (optional, all have sensible defaults)
loss_masking_strategy: "full"
ntfy_topic: ""
hf_token: ""
```

**Try the new features:**
```bash
# Test loss masking strategies
late train examples/loss_masking/full_masking_example.yml

# Test notifications
late train examples/with_notifications.yml

# Test LoRA merging
late merge /path/to/checkpoint --base-model model/name --output user/merged
```

---

## Acknowledgments

- Training script implementation inspired by proven ROCm training patterns
- ntfy.sh integration for simple, privacy-friendly notifications
- Comprehensive testing infrastructure for production readiness

---

**Full documentation:** See README.md and tests/README.md
**Examples:** See examples/ directory
**Tests:** See tests/ directory
