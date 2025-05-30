# LLM Markdown Generator

A configurable Python framework that leverages Large Language Models (LLMs) to generate markdown blog posts with customizable 11ty-compatible YAML front matter and extensible plugin system.

## 🌟 Features

- Generate markdown blog posts using LLMs (supports OpenAI and Google Gemini)
- Multiple content types with specialized prompt templates (blogs, tutorials, reviews, analysis)
- Customizable prompt templates using Jinja2
- 11ty-compatible YAML front matter generation
- Configuration-driven design for easy customization
- Command-line interface for easy integration into workflows
- Extensible plugin system for content processing and front matter enhancement
- Token usage tracking and reporting for cost management
- Advanced error handling with retries and backoff

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
llm-markdown python --title "Python Tips" --provider gemini --api-key "YOUR_API_KEY_HERE"

# Enable specific plugins
llm-markdown python --title "Python Generators" --plugins "add_timestamp,add_reading_time,add_license"

# Use a custom plugins directory
llm-markdown python --title "Python Tips" --plugins-dir "./custom_plugins"

# Show token usage during generation
llm-markdown python --title "Python Classes" --show-tokens

# Run in dry-run mode (no API calls, no file writes)
llm-markdown python --title "Python Testing" --dry-run --verbose

# Generate a security advisory for a specific CVE
llm-markdown generate-cve-report CVE-2023-1234 --title "Critical Remote Code Execution Vulnerability"
```

### CLI Options

```
Usage: llm-markdown [OPTIONS] COMMAND [ARGS]...

Generate markdown blog posts using LLMs

Commands:
  generate              Generate a markdown blog post for the specified topic.
  generate-cve-report   Generate a comprehensive security advisory for the specified CVE.
  usage-report          Generate a token usage report from a log file.

Options for generate command:
  --config-path TEXT              Path to the configuration file  [default: config/config.yaml]
  --output-dir TEXT               Output directory (overrides the one in config)
  --title TEXT                    Title for the generated post
  --keywords TEXT                 Comma-separated list of keywords to use (adds to those in config)
  --provider TEXT                 Override the LLM provider (openai or gemini)
  --model TEXT                    Override the model name (e.g., gpt-4o, gemini-2.0-flash)
  --api-key TEXT                  Directly provide an API key instead of using environment variables
  --plugins TEXT                  Comma-separated list of plugins to enable (e.g., 'add_timestamp,add_reading_time')
  --plugins-dir TEXT              Directory containing custom plugins
  --no-plugins                    Disable loading of plugins
  -v, --verbose                   Display verbose output including token usage
  --dry-run                       Run without actually calling the LLM API or writing files
  -r, --retries INTEGER           Number of retries for API calls  [default: 3]
  --retry-delay FLOAT             Base delay in seconds between retries  [default: 1.0]
  --show-tokens                    Show token usage information after generation
  -e, --extra TEXT                Extra parameters in JSON format (e.g., '{"temperature": 0.9}')
  --help                          Show this message and exit.
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

Available prompt templates:
- `python_blog.j2`: For Python programming posts
- `javascript_blog.j2`: For JavaScript and web development posts
- `data_science_blog.j2`: For data science and machine learning posts
- `technical_tutorial.j2`: For step-by-step technical guides
- `product_review.j2`: For detailed product reviews
- `comparative_analysis.j2`: For comparing technologies, frameworks, etc.
- `research_summary.j2`: For academic research summaries
- `industry_trend_analysis.j2`: For market and industry trend analysis
- `security_advisory.j2`: For detailed security advisories on specific vulnerabilities
- `daily_cve_report.j2`: For daily reports on critical CVEs with high EPSS scores

## 🔌 Plugin System

The framework includes an extensible plugin system to customize the generated content and front matter. There are two types of plugins:

1. **Content Processors**: Modify the generated markdown content
2. **Front Matter Enhancers**: Add or modify fields in the YAML front matter

### Built-in Plugins

The framework includes several built-in plugins:

**Content Processors:**
- `add_timestamp`: Adds a timestamp to the content
- `add_reading_time`: Adds an estimated reading time
- `add_table_of_contents`: Generates a table of contents from headings
- `add_tag_links`: Converts tags to links

**Front Matter Enhancers:**
- `add_metadata`: Adds additional metadata fields
- `enhance_seo`: Adds SEO-related fields
- `add_series_info`: Adds series information
- `add_readability_stats`: Adds readability statistics

### Creating Custom Plugins

You can create custom plugins by defining functions with the `@plugin_hook` decorator:

```python
from llm_markdown_generator.plugins import plugin_hook

@plugin_hook("content_processor", "my_custom_processor")
def my_custom_processor(content: str, **kwargs) -> str:
    """Custom content processor that adds a footer."""
    footer = "\n\n---\nCustom footer text"
    return content + footer

@plugin_hook("front_matter_enhancer", "my_custom_enhancer")
def my_custom_enhancer(front_matter: dict, **kwargs) -> dict:
    """Custom front matter enhancer that adds a custom field."""
    enhanced = front_matter.copy()
    enhanced["custom_field"] = "Custom value"
    return enhanced
```

Save these in a Python file, place it in a directory, and use the `--plugins-dir` option to load them.

## 🔒 Security Advisories & CVE Reports 

The framework provides specialized capabilities for generating security content:

```bash
# Generate a security advisory for a specific CVE
llm-markdown generate-cve-report CVE-2023-45678 

# Add a custom title for the report
llm-markdown generate-cve-report CVE-2023-45678 --title "Critical Authentication Bypass Vulnerability"

# Save to a specific directory
llm-markdown generate-cve-report CVE-2023-45678 --output-dir "security/advisories"

# Use specific provider and model
llm-markdown generate-cve-report CVE-2023-45678 --provider openai --model gpt-4o
```

The generated CVE reports include:
- Executive summary of the vulnerability
- Vulnerability snapshot with CVSS score, EPSS score, and affected systems
- Technical details and exploit information
- Comprehensive mitigation steps
- References to official advisories
- Industry impact analysis

## 📊 Token Usage Tracking

The framework includes simple token usage tracking to help monitor costs:

```bash
# Show token usage during generation
llm-markdown python --title "Python Classes" --show-tokens
```

### Token Usage Information

The token usage information displays:

- Total tokens used (prompt, completion, and total)
- Estimated cost based on provider pricing
- Detailed record history (with `--detailed` flag)

## 🧪 Testing

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src

# Run specific test modules
pytest tests/unit/test_plugins.py
pytest tests/integration/test_token_tracking.py
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

## 📋 Advanced Error Handling

The framework includes advanced error handling with retries and exponential backoff:

- Automatic retries for API calls with configurable settings
- Error classification for different types of API errors
- Detailed error reporting with verbose mode
- Proper handling of rate limiting with suggested wait times

## 📜 License

[MIT License](LICENSE)

## 🤝 Contributing

Contributions are welcome! Please check out our [Contributing Guide](CONTRIBUTING.md) for more details.