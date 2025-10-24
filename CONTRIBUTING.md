# Contributing to Optimal Image Renamer

Thank you for your interest in contributing to Optimal Image Renamer! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

Please be respectful and constructive in all interactions. We're here to build something useful together.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/ai-image-renamer-tool-ollama.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`

## Development Setup

### Prerequisites

- Python 3.8 or higher (3.9+ recommended for development)
- Ollama with the LLaVA model installed
- NVIDIA GPU (optional, but recommended for testing GPU features)

### Install Development Dependencies

```bash
# Install the package in editable mode with dev dependencies
pip install -e ".[dev]"

# Or install dev dependencies separately
pip install pytest pytest-cov pytest-mock mypy ruff pre-commit
```

### Set Up Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
pre-commit install
```

This will automatically run linters and formatters before each commit.

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-new-model-support` - for new features
- `fix/handle-corrupted-images` - for bug fixes
- `docs/update-readme` - for documentation
- `refactor/improve-performance` - for refactoring

### Coding Standards

1. **Code Style**: We use Ruff for linting and formatting
   ```bash
   ruff check .
   ruff format .
   ```

2. **Type Hints**: Add type hints to all function signatures
   ```python
   def process_image(path: Path) -> dict[str, Any]:
       ...
   ```

3. **Docstrings**: Use Google-style docstrings
   ```python
   def example_function(param1: str, param2: int) -> bool:
       """
       Brief description of function.

       Longer description if needed.

       Args:
           param1: Description of param1
           param2: Description of param2

       Returns:
           Description of return value

       Raises:
           ValueError: When param2 is negative
       """
   ```

4. **Comments**: Use comments to explain "why", not "what"

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_optimal_image_renamer.py

# Run specific test
pytest tests/test_optimal_image_renamer.py::test_md5sum
```

### Writing Tests

- Write tests for all new features and bug fixes
- Aim for >80% code coverage
- Use descriptive test names that explain what is being tested
- Use fixtures for common test setup
- Mock external dependencies (Ollama API, filesystem operations, etc.)

Example:
```python
def test_process_image_handles_rgba():
    """Test that RGBA images are correctly converted to RGB."""
    # Arrange
    test_image = create_test_rgba_image()

    # Act
    result = renamer._load_image(test_image)

    # Assert
    assert result.mode == "RGB"
```

## Code Quality

### Type Checking

Run mypy to check type hints:
```bash
mypy OPTIMALIMAGERENAMER.py --ignore-missing-imports
```

### Linting

```bash
ruff check .
```

### Formatting

```bash
ruff format .
```

### All Quality Checks

Run all checks before submitting:
```bash
# Format code
ruff format .

# Check linting
ruff check .

# Type check
mypy OPTIMALIMAGERENAMER.py --ignore-missing-imports

# Run tests
pytest --cov=. --cov-report=term-missing
```

## Submitting Changes

### Commit Messages

Write clear, descriptive commit messages:

```
Add support for HEIC image format

- Add HEIC to IMAGE_EXTENSIONS
- Update tests to include HEIC test cases
- Update README with HEIC support information

Fixes #123
```

Format:
- First line: Brief summary (50 chars or less)
- Blank line
- Detailed description (wrap at 72 chars)
- Reference any related issues

### Pull Request Process

1. **Update your branch** with the latest main:
   ```bash
   git checkout main
   git pull upstream main
   git checkout your-branch
   git rebase main
   ```

2. **Run all quality checks** (see above)

3. **Push your changes**:
   ```bash
   git push origin your-branch
   ```

4. **Create a Pull Request** on GitHub with:
   - Clear title describing the change
   - Description of what changed and why
   - Screenshots or examples if applicable
   - Link to any related issues

5. **Respond to feedback** from reviewers

### PR Checklist

Before submitting, ensure:
- [ ] Code follows the project's coding standards
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated (README, docstrings, etc.)
- [ ] No ruff or mypy warnings
- [ ] Commit messages are clear and descriptive
- [ ] Changes are rebased on latest main

## Additional Notes

### Performance Considerations

- Profile code before optimizing
- Consider memory usage for large image batches
- Test with various image sizes and formats

### Security

- Never commit API keys or credentials
- Validate all file paths to prevent directory traversal
- Handle user input carefully

### Questions?

- Open an issue for discussion
- Check existing issues and PRs
- Review the README for additional information

## License

By contributing, you agree that your contributions will be placed in the public domain.

Thank you for contributing!
