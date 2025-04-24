# Contributing Guidelines

Thank you for your interest in contributing to this project\! We value the contributions of each community member and want to make the process as smooth as possible.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please read it before contributing.

## How to Contribute

### Reporting Bugs

- Before submitting a bug report, please check that it has not already been reported
- Use the bug report template when creating an issue
- Include as much detail as possible: steps to reproduce, expected behavior, actual behavior, and environment details

### Suggesting Features

- Use the feature request template when creating an issue
- Clearly describe the problem your feature would solve
- Explain how your solution would work
- Consider the scope of the feature

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes
4. Add or update tests as necessary
5. Ensure all tests pass
6. Commit your changes with clear, descriptive commit messages
7. Push your branch to your fork
8. Submit a pull request to the main repository

## Development Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Style Guidelines

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add comments to explain complex logic
- Write good commit messages

## Testing

- Add tests for new features
- Ensure all tests pass before submitting a pull request
- Aim for good test coverage

## Documentation

- Update documentation for new features or changes in behavior
- Use clear language and provide examples where helpful

## Review Process

- All submissions require review
- Maintainers may suggest changes or improvements
- Once approved, maintainers will merge your contribution

Thank you for contributing\!
