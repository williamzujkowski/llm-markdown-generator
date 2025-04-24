# LLM Markdown Generator

A configurable Python framework that leverages Large Language Models (LLMs) to generate markdown blog posts with customizable 11ty-compatible YAML front matter.

## 🌟 Features

- Generate markdown blog posts using LLMs (supports OpenAI and Google Gemini)
- Customizable prompt templates using Jinja2
- 11ty-compatible YAML front matter generation
- Configuration-driven design for easy customization
- Command-line interface for easy integration into workflows

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/llm-markdown-generator.git
cd llm-markdown-generator

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## 📝 Usage

### Set Environment Variables

Create a `.env` file in the project root with your API keys:

```
# For OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# For Google Gemini
GEMINI_API_KEY=your_gemini_api_key_here
```

### Generate Content

Use the CLI to generate content for a specific topic:

```bash
# Generate a post about Python decorators (using default provider from config)
llm-markdown python --title "Understanding Python Decorators"

# Generate a JavaScript post with custom keywords
llm-markdown javascript --title "Async/Await Patterns" --keywords "async,await,promises,error handling"

# Specify a custom output directory
llm-markdown data_science --title "Introduction to Linear Regression" --output-dir "my-blog/posts"

# Use Google Gemini instead of OpenAI (overriding the config)
llm-markdown python --title "Python Decorators Guide" --provider gemini

# Provide an API key directly on the command line
llm-markdown python --title "Python Tips" --provider gemini --api-key "AIzaSyBiyiSjDToD9rAMR1UXPwpBlrHuuT5CTG0"
```

## ⚙️ Configuration

### Main Configuration

The framework is configured via YAML files in the `config/` directory:

- `config/config.yaml`: Main configuration file
- `config/front_matter_schema.yaml`: Schema for the front matter

You can switch between different LLM providers by editing the `provider_type` in the config file or by using the `--provider` flag in the CLI.

### Supported LLM Providers

- **OpenAI**: Using GPT models via the OpenAI API
- **Google Gemini**: Using Gemini models via the Google AI API

### Prompt Templates

Prompt templates are stored in the `.llmconfig/prompt-templates/` directory as Jinja2 templates. You can create custom templates for different topics or content types.

## 🧪 Testing

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src
```

## 🧹 Code Quality

```bash
# Format code with Black
black .

# Sort imports with isort
isort .

# Lint with flake8
flake8

# Type check with mypy
mypy .
```

## 📜 License

[MIT License](LICENSE)

## 🤝 Contributing

Contributions are welcome! Please check out our [Contributing Guide](CONTRIBUTING.md) for more details.