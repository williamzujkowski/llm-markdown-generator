# LLM Markdown Generator Usage Guide

This guide provides detailed instructions for using the LLM Markdown Generator framework in various scenarios.

## Table of Contents

1. [Configuration](#configuration)
2. [Command Line Interface](#command-line-interface)
3. [Specialized Content Generation](#specialized-content-generation)
4. [Plugin System](#plugin-system)
5. [Token Usage Tracking](#token-usage-tracking)
6. [Programmatic Usage](#programmatic-usage)
7. [Advanced Features](#advanced-features)

## Configuration

### Main Configuration File

The primary configuration file is located at `config/config.yaml` and contains the following key sections:

```yaml
# LLM Provider Configuration
provider_type: "openai"  # or "gemini"
model_name: "gpt-4o"     # or "gemini-1.5-pro"

# Path Configuration
templates_dir: ".llmconfig/prompt-templates"
output_dir: "output"
front_matter_schema_path: "config/front_matter_schema.yaml"

# Default Parameters
default_params:
  temperature: 0.7
  max_tokens: 2048

# Topic-specific configurations
topics:
  python:
    title_prefix: "Python Tutorial: "
    keywords:
      - python
      - programming
      - development
  javascript:
    title_prefix: "JavaScript Guide: "
    keywords:
      - javascript
      - web development
      - frontend
  data_science:
    title_prefix: "Data Science Insights: "
    keywords:
      - data science
      - machine learning
      - analytics
```

### Front Matter Schema

The front matter schema is defined in `config/front_matter_schema.yaml`:

```yaml
# Basic Front Matter Schema
title: required
date: auto      # Will be populated automatically with current date
description: optional
keywords: optional
tags: optional
author: optional
layout: optional
```

## Command Line Interface

### Basic Usage

```bash
# Generate a Python blog post
llm-markdown python --title "Understanding Python Decorators"

# Generate a JavaScript post with custom keywords
llm-markdown javascript --title "Modern JavaScript Features" --keywords "ES6,promises,async,modules"
```

### Advanced Options

```bash
# Use a different LLM provider
llm-markdown python --title "Python Tips" --provider gemini

# Specify the output directory
llm-markdown data_science --title "Regression Analysis" --output-dir "blog/posts"

# Enable specific plugins
llm-markdown python --title "Python Classes" --plugins "add_timestamp,add_reading_time"

# Track token usage
llm-markdown python --title "Error Handling" --token-log "usage.jsonl" --usage-report

# Run in dry-run mode (no API calls, no file writes)
llm-markdown python --title "Testing Guide" --dry-run --verbose
```

## Specialized Content Generation

The framework supports various content types through different prompt templates. You can generate specialized content using either the CLI or the example script.

### Using Example Script

The `examples/generate_specialized_content.py` script provides a convenient way to generate different types of content:

```bash
# Generate a technical tutorial
python examples/generate_specialized_content.py technical_tutorial "Building RESTful APIs with FastAPI" --keywords "FastAPI,Python,REST,API"

# Generate a product review
python examples/generate_specialized_content.py product_review "VS Code for Python Development" --audience "Python developers"

# Generate a comparative analysis
python examples/generate_specialized_content.py comparative_analysis "SQL vs NoSQL Databases" --tone "technical and objective"

# Generate a research summary
python examples/generate_specialized_content.py research_summary "Advances in Natural Language Processing" --keywords "NLP,transformers,BERT,GPT"

# Generate an industry trend analysis
python examples/generate_specialized_content.py industry_trend_analysis "DevOps Trends 2025" --keywords "CI/CD,GitOps,observability"
```

### Available Templates

| Template Name | Description | Best For |
|--------------|-------------|----------|
| `python_blog.j2` | Python-specific blog posts | Technical Python topics |
| `javascript_blog.j2` | JavaScript and web development | Web development topics |
| `data_science_blog.j2` | Data science and ML topics | Analytics and ML concepts |
| `technical_tutorial.j2` | Step-by-step guides | How-to content and walkthroughs |
| `product_review.j2` | Detailed product evaluations | Tool and technology assessments |
| `comparative_analysis.j2` | Technology comparisons | Decision-making content |
| `research_summary.j2` | Academic/research overviews | Literature reviews and research |
| `industry_trend_analysis.j2` | Market and industry trends | Forward-looking analyses |

## Plugin System

### Using Built-in Plugins

Enable built-in plugins via the CLI:

```bash
# Enable specific content processors
llm-markdown python --title "Python Tips" --plugins "add_timestamp,add_reading_time,add_table_of_contents"

# Enable front matter enhancers
llm-markdown javascript --title "JavaScript Events" --plugins "add_metadata,enhance_seo"
```

### Creating Custom Plugins

1. Create a new Python file with your plugin functions:

```python
# my_plugins.py
from llm_markdown_generator.plugins import plugin_hook

@plugin_hook("content_processor", "add_custom_footer")
def add_custom_footer(content: str, **kwargs) -> str:
    """Add a custom footer to the content."""
    return content + "\n\n---\n*Generated with LLM Markdown Generator*"

@plugin_hook("front_matter_enhancer", "add_custom_fields")
def add_custom_fields(front_matter: dict, **kwargs) -> dict:
    """Add custom fields to front matter."""
    enhanced = front_matter.copy()
    enhanced["custom_field"] = "Custom value"
    enhanced["generated_by"] = "LLM Markdown Generator"
    return enhanced
```

2. Use your custom plugins:

```bash
# Use custom plugins from a specific directory
llm-markdown python --title "Python Tips" --plugins-dir "./my_plugins" --plugins "add_custom_footer,add_custom_fields"
```

## Token Usage Tracking

### Tracking Usage

Enable token tracking with the `--token-log` option:

```bash
# Track token usage to a JSON Lines file
llm-markdown python --title "Python Classes" --token-log "token_usage.jsonl"
```

### Generating Reports

View a usage report with the `--usage-report` flag:

```bash
# Display usage report after generation
llm-markdown python --title "Async Python" --token-log "token_usage.jsonl" --usage-report
```

Or generate a standalone report:

```bash
# Generate report from existing log file
llm-markdown usage-report token_usage.jsonl --detailed
```

## Programmatic Usage

You can use the framework programmatically in your own Python code:

```python
from llm_markdown_generator.config import Config
from llm_markdown_generator.llm_provider import create_llm_provider
from llm_markdown_generator.prompt_engine import PromptEngine
from llm_markdown_generator.front_matter import FrontMatterGenerator
from llm_markdown_generator.generator import MarkdownGenerator
from llm_markdown_generator.token_tracker import TokenTracker
from llm_markdown_generator.plugins import load_plugins
import os

# Load configuration
config = Config("config/config.yaml")

# Create token tracker
token_tracker = TokenTracker()

# Create LLM provider
llm_client = create_llm_provider(
    provider_type=config.provider_type,
    model_name=config.model_name,
    api_key=os.getenv(f"{config.provider_type.upper()}_API_KEY"),
    token_tracker=token_tracker
)

# Create prompt engine
prompt_engine = PromptEngine(templates_dir=config.templates_dir)

# Create front matter generator
front_matter_generator = FrontMatterGenerator(config.front_matter_schema_path)

# Load plugins
content_processors, front_matter_enhancers = load_plugins()

# Create markdown generator
markdown_generator = MarkdownGenerator(
    llm_client=llm_client,
    prompt_engine=prompt_engine,
    front_matter_generator=front_matter_generator,
    content_processors=content_processors,
    front_matter_enhancers=front_matter_enhancers
)

# Generate markdown content
context = {
    "topic": "Python Decorators",
    "title": "Understanding Python Decorators",
    "keywords": ["python", "decorators", "functions", "metaprogramming"]
}

markdown_content = markdown_generator.generate(
    template_name="python_blog.j2",
    context=context,
    front_matter_data={
        "title": context["title"],
        "description": "A guide to understanding and using Python decorators",
        "tags": ["python", "programming"]
    }
)

# Write to file
with open("output/python-decorators.md", "w") as f:
    f.write(markdown_content)

# Print token usage
print(f"Token usage - Prompt: {token_tracker.prompt_tokens}, Completion: {token_tracker.completion_tokens}")
```

## Advanced Features

### Error Handling

The framework includes advanced error handling with retries and exponential backoff:

```bash
# Configure retry parameters
llm-markdown python --title "Python Classes" --retries 5 --retry-delay 2.0
```

### Custom Parameters

Pass custom parameters to the LLM provider:

```bash
# Pass custom parameters as JSON
llm-markdown python --title "Python Classes" --extra '{"temperature": 0.9, "top_p": 0.8}'
```

### Using with Pydantic

For enhanced validation, you can use the Pydantic configuration models in your code:

```python
from llm_markdown_generator.config_pydantic import PydanticConfig

# Load with Pydantic validation
config = PydanticConfig.from_yaml("config/config.yaml")

# Access validated fields
print(config.provider_type)  # Type-checked
print(config.model_name)     # Validated against allowed values
```

See `examples/generate_with_pydantic.py` for a complete example.