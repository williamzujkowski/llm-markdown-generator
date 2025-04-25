# Configuration Validation with Pydantic

This document describes how to use the enhanced configuration validation features provided by Pydantic in the LLM Markdown Generator framework.

## Introduction

The LLM Markdown Generator framework uses [Pydantic](https://docs.pydantic.dev/) to provide strong typing and validation for configuration files. This helps to ensure that your configuration is valid and well-formed before any content generation begins, preventing runtime errors due to configuration issues.

## Benefits of Pydantic Validation

Using Pydantic for configuration validation provides several benefits:

1. **Strong Typing**: Ensures that configuration values have the correct data types.
2. **Validation Rules**: Enforces constraints on values (e.g., temperature must be between 0.0 and 1.0).
3. **Default Values**: Provides sensible defaults for optional fields.
4. **Error Messages**: Provides detailed error messages when validation fails.
5. **Documentation**: Self-documents configuration through field descriptions.

## Using Pydantic Validation

The framework provides two options for configuration handling:

1. **Default**: The original dataclass-based configuration system.
2. **Enhanced**: The new Pydantic-based configuration system with stronger validation.

### In Your Code

When using the framework programmatically, you can choose to import from either module:

```python
# Original dataclass-based configuration
from llm_markdown_generator.config import Config, load_config, load_front_matter_schema

# Enhanced Pydantic-based configuration
from llm_markdown_generator.config_pydantic import Config, load_config, load_front_matter_schema
```

The CLI automatically attempts to use the Pydantic version first, falling back to the original version if Pydantic is not available.

## Configuration Schema

The Pydantic models define the following schema for configuration:

### Main Configuration

The main `Config` class represents the top-level configuration object:

- `llm_provider`: Configuration for the LLM provider (required)
- `front_matter`: Configuration for front matter (required)
- `topics`: Dictionary of topic configurations (required)
- `output_dir`: Directory where generated markdown files will be saved (default: "output")

### LLM Provider Configuration

The `LLMProviderConfig` class represents the configuration for an LLM provider:

- `provider_type`: Type of LLM provider, must be "openai" or "gemini" (required)
- `model_name`: Name of the model to use (required)
- `api_key_env_var`: Name of the environment variable containing the API key (required)
- `temperature`: Controls randomness in output, must be between 0.0 and 1.0 (default: 0.7)
- `max_tokens`: Maximum tokens to generate, must be at least 1 (optional)
- `additional_params`: Additional parameters to pass to the LLM API (default: {})

### Front Matter Configuration

The `FrontMatterConfig` class represents the configuration for front matter:

- `schema_path`: Path to the front matter schema YAML file (required)

### Topic Configuration

The `TopicConfig` class represents the configuration for a specific content topic:

- `name`: Name of the topic (required)
- `prompt_template`: Name of the prompt template file with extension (required)
- `keywords`: List of keywords for the topic, must have at least one (required)
- `custom_data`: Custom data for prompt context (default: {})

## Error Handling

When using Pydantic validation, configuration errors will be more specific and detailed. For example:

```
ConfigError: Invalid configuration: 1 validation error for Config
topics.python.name
  Field required [type=missing, input_value={'prompt_template': 'python_blog.j2', 'keywords': ['python', 'programming']}, input_type=dict]
```

This error message clearly indicates that the `name` field is missing from the `python` topic configuration.

## Example Usage

See the `examples/generate_with_pydantic.py` file for a complete example of using Pydantic-based configuration validation.

```python
from llm_markdown_generator.config_pydantic import Config, load_config

# Load configuration with Pydantic validation
config = load_config("config/config.yaml")

# Access validated configuration values
provider_type = config.llm_provider.provider_type  # Guaranteed to be "openai" or "gemini"
temperature = config.llm_provider.temperature      # Guaranteed to be between 0.0 and 1.0
```