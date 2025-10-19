# Late Testing Guide

Comprehensive testing documentation for the Late training library.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Running Tests](#running-tests)
- [Test Structure](#test-structure)
- [Writing Tests](#writing-tests)
- [Fixtures](#fixtures)
- [Coverage](#coverage)
- [CI/CD Integration](#cicd-integration)

---

## 🚀 Quick Start

### Installation

Install test dependencies:

```bash
# Install Late with test dependencies
pip install -e ".[test]"

# Or install test packages separately
pip install pytest pytest-cov pytest-mock responses
```

### Run All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=late --cov-report=html
```

---

## 🏃 Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_config.py

# Run specific test class
pytest tests/unit/test_loss_masking.py::TestFullLossMasking

# Run specific test function
pytest tests/unit/test_loss_masking.py::TestFullLossMasking::test_default_is_full_masking

# Run tests matching pattern
pytest -k "test_loss"
```

### Test Categories

Tests are organized with markers for selective running:

```bash
# Run only unit tests (fast)
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run tests that require GPU
pytest -m requires_gpu

# Run tests that need network
pytest -m requires_network
```

### Parallel Execution

Speed up tests by running in parallel:

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel (auto-detect CPUs)
pytest -n auto

# Run on 4 CPUs
pytest -n 4
```

### Watch Mode

Auto-run tests on file changes:

```bash
# Install pytest-watch
pip install pytest-watch

# Watch and re-run tests
ptw
```

---

## 📁 Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── pytest.ini               # Pytest settings
├── README.md                # This file
├── fixtures/                # Test data and mocks
│   ├── configs/             # Sample YAML configs
│   └── datasets/            # Sample datasets
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_config.py       # Configuration loading tests
│   ├── test_loss_masking.py # Loss masking strategy tests
│   ├── test_merge_lora.py   # LoRA merging tests
│   ├── test_sweep.py        # Sweep generation tests
│   ├── test_training.py     # Training script generation tests
│   └── test_callbacks.py    # Callback tests (ntfy, etc.)
├── integration/             # Integration tests (slower)
│   └── test_end_to_end.py   # Full workflow tests
└── utils/                   # Test utilities
    └── test_helpers.py      # Helper functions for tests
```

### Test Categories

**Unit Tests** (`tests/unit/`)
- Fast, isolated tests
- Mock external dependencies
- Test individual functions/classes
- Run frequently during development

**Integration Tests** (`tests/integration/`)
- Test component interactions
- May use real (small) models
- Slower execution
- Run before commits/PRs

---

## ✍️ Writing Tests

### Test File Naming

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Example Test Structure

```python
import pytest
from late.engine.training import generate_training_script

class TestTrainingScript:
    """Tests for training script generation."""

    def test_default_loss_masking(self, sample_config_full_masking):
        """Test that default loss masking is 'full'."""
        script = generate_training_script(sample_config_full_masking)

        assert 'FULL Loss Masking' in script
        assert 'format_for_training' in script

    def test_assistant_only_masking(self, sample_config_assistant_masking):
        """Test assistant_only loss masking."""
        script = generate_training_script(sample_config_assistant_masking)

        assert 'ASSISTANT-ONLY Loss Masking' in script
        assert 'preprocess_for_assistant_loss' in script

    @pytest.mark.slow
    def test_full_training_run(self, sample_config_full_masking):
        """Slow integration test."""
        # This test takes longer
        pass
```

### Using Fixtures

Fixtures provide reusable test data and mocks:

```python
def test_with_fixtures(sample_config_full_masking, mock_tokenizer, tmp_path):
    """Test using multiple fixtures."""
    # sample_config_full_masking: Sample config dict
    # mock_tokenizer: Mock HuggingFace tokenizer
    # tmp_path: Temporary directory (pytest built-in)

    config = sample_config_full_masking.copy()
    config['cache_dir'] = str(tmp_path)

    # Test code here...
```

### Mocking External Dependencies

```python
from unittest.mock import Mock, patch

def test_with_mocking():
    """Test using mocks."""
    with patch('late.engine.training.AutoModelForCausalLM') as mock_model:
        # Configure mock
        mock_model.from_pretrained.return_value = Mock()

        # Run test
        # ...

        # Verify mock was called
        mock_model.from_pretrained.assert_called_once()
```

### Test Markers

Mark tests for categorization:

```python
import pytest

@pytest.mark.unit
def test_fast_unit():
    """Fast unit test."""
    pass

@pytest.mark.integration
def test_integration():
    """Integration test."""
    pass

@pytest.mark.slow
def test_slow_operation():
    """Slow test (e.g., full training run)."""
    pass

@pytest.mark.requires_gpu
def test_needs_gpu():
    """Test that needs GPU."""
    pass

@pytest.mark.requires_network
def test_needs_network():
    """Test that needs network access."""
    pass
```

### Parametrized Tests

Test multiple inputs efficiently:

```python
@pytest.mark.parametrize("loss_strategy,expected", [
    ("full", "format_for_training"),
    ("assistant_only", "preprocess_for_assistant_loss"),
])
def test_loss_strategies(loss_strategy, expected, sample_config_full_masking):
    """Test different loss masking strategies."""
    config = sample_config_full_masking.copy()
    config['loss_masking_strategy'] = loss_strategy

    script = generate_training_script(config)

    assert expected in script
```

---

## 🎯 Fixtures

### Available Fixtures (from conftest.py)

#### Configuration Fixtures

- `sample_config_full_masking` - Config with full loss masking (default)
- `sample_config_assistant_masking` - Config with assistant-only masking
- `sample_config_lora` - LoRA training config
- `sample_config_with_ntfy` - Config with ntfy notifications
- `sample_config_file` - Temporary YAML config file
- `sample_sweep_file` - Temporary sweep config file

#### Data Fixtures

- `sample_messages` - Sample chat messages
- `sample_dataset_dict` - Sample HuggingFace dataset

#### Mock Fixtures

- `mock_tokenizer` - Mock HuggingFace tokenizer
- `mock_model` - Mock HuggingFace model
- `mock_peft_model` - Mock PEFT model
- `mock_hf_api` - Mock HuggingFace API
- `mock_requests` - Mock requests library

#### Utility Fixtures

- `temp_output_dir` - Temporary output directory
- `temp_model_cache` - Temporary model cache
- `tmp_path` - Pytest built-in temp directory
- `clean_env` - Clean environment variables
- `hf_token_env` - Set HF_TOKEN environment variable

### Creating Custom Fixtures

Add fixtures to `conftest.py`:

```python
import pytest

@pytest.fixture
def my_custom_fixture():
    """Custom fixture for my tests."""
    # Setup
    data = {"key": "value"}

    yield data

    # Teardown (optional)
    # Cleanup code here
```

---

## 📊 Coverage

### Running Coverage

```bash
# Run tests with coverage
pytest --cov=late

# Generate HTML report
pytest --cov=late --cov-report=html

# Open HTML report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows

# Generate terminal report with missing lines
pytest --cov=late --cov-report=term-missing

# Fail if coverage is below threshold
pytest --cov=late --cov-fail-under=80
```

### Coverage Goals

- **Minimum**: 80% overall coverage
- **Target**: 90% for core modules
- **Focus areas**:
  - `late/engine/training.py` - 90%+
  - `late/engine/sweep.py` - 90%+
  - `late/engine/merge_lora.py` - 90%+
  - `late/engine/config.py` - 95%+

### Viewing Coverage

Coverage report shows:
- Lines executed vs. total lines
- Missing line numbers
- Branch coverage (if enabled)

Example output:
```
Name                           Stmts   Miss  Cover   Missing
------------------------------------------------------------
late/engine/config.py            42      2    95%   45-46
late/engine/training.py         156      8    95%   234, 456-462
late/engine/sweep.py            98      5    95%   123, 234-237
late/engine/merge_lora.py       87      3    97%   145-147
------------------------------------------------------------
TOTAL                           383     18    95%
```

---

## 🔄 CI/CD Integration

### GitHub Actions

Tests run automatically on:
- Push to main branch
- Pull requests
- Manual workflow dispatch

Example workflow (`.github/workflows/test.yml`):

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.8", "3.9", "3.10", "3.11"]

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -e ".[test]"

      - name: Run tests
        run: |
          pytest --cov=late --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

### Pre-commit Hooks

Run tests before commits:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## 🐛 Debugging Tests

### Verbose Output

```bash
# Show print statements
pytest -s

# Show full error tracebacks
pytest --tb=long

# Show all test output
pytest -vv

# Stop on first failure
pytest -x

# Drop into debugger on failure
pytest --pdb
```

### Logging

Enable logging during tests:

```python
import pytest
import logging

@pytest.fixture(autouse=True)
def configure_logging():
    """Enable logging for tests."""
    logging.basicConfig(level=logging.DEBUG)
```

### Common Issues

**Import Errors**
```bash
# Install Late in editable mode
pip install -e .
```

**Missing Fixtures**
- Check that fixtures are defined in `conftest.py`
- Verify fixture names match usage

**Flaky Tests**
- Use `@pytest.mark.flaky(reruns=3)` (requires pytest-rerunfailures)
- Check for race conditions
- Ensure proper cleanup in fixtures

---

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [pytest-mock Documentation](https://pytest-mock.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)

---

## 🤝 Contributing Tests

When contributing:

1. **Write tests for new features**
   - Add unit tests for new functions
   - Add integration tests for workflows

2. **Maintain coverage**
   - Aim for 80%+ coverage
   - Test edge cases and error handling

3. **Follow conventions**
   - Use descriptive test names
   - Add docstrings to test classes/functions
   - Use appropriate markers

4. **Run tests before PR**
   ```bash
   pytest --cov=late --cov-fail-under=80
   ```

5. **Update documentation**
   - Add new fixtures to this README
   - Document special test requirements

---

## 📞 Support

For questions about testing:
- Check existing tests for examples
- Review this documentation
- Ask in GitHub Discussions
- Open an issue for test infrastructure problems

---

**Happy Testing! 🎉**
