#!/usr/bin/env python3
"""
Example script demonstrating the use of Pydantic-based configuration validation.

This script shows how to use the enhanced configuration system with Pydantic
for stronger typing and validation.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_markdown_generator.config_pydantic import (
    Config, ConfigError, load_config, load_front_matter_schema
)
from src.llm_markdown_generator.front_matter import FrontMatterGenerator
from src.llm_markdown_generator.generator import MarkdownGenerator
from src.llm_markdown_generator.llm_provider import GeminiProvider, OpenAIProvider
from src.llm_markdown_generator.prompt_engine import PromptEngine

def main():
    """Generate a markdown blog post using Pydantic-based configuration."""
    try:
        # 1. Load the configuration with Pydantic validation
        print("Loading configuration with Pydantic validation...")
        config_path = "config/config.yaml"
        config = load_config(config_path)
        
        # 2. Load the front matter schema
        print(f"Loading front matter schema from {config.front_matter.schema_path}...")
        front_matter_schema = load_front_matter_schema(config.front_matter.schema_path)
        
        # 3. Create the LLM provider based on the configuration
        provider_type = config.llm_provider.provider_type.lower()
        print(f"Creating {provider_type} provider...")
        
        if provider_type == "openai":
            llm_provider = OpenAIProvider(
                model_name=config.llm_provider.model_name,
                api_key_env_var=config.llm_provider.api_key_env_var,
                temperature=config.llm_provider.temperature,
                max_tokens=config.llm_provider.max_tokens,
                additional_params=config.llm_provider.additional_params,
            )
        elif provider_type == "gemini":
            # For Gemini, we'll use a specific model
            gemini_model = "gemini-2.0-flash"  # Latest model
            llm_provider = GeminiProvider(
                model_name=gemini_model,
                api_key_env_var=config.llm_provider.api_key_env_var,
                temperature=config.llm_provider.temperature,
                max_tokens=config.llm_provider.max_tokens,
                additional_params=config.llm_provider.additional_params,
            )
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")
        
        # 4. Create the prompt engine
        print("Creating prompt engine...")
        templates_dir = Path(".llmconfig/prompt-templates")
        prompt_engine = PromptEngine(str(templates_dir))
        
        # 5. Create the front matter generator
        print("Creating front matter generator...")
        front_matter_generator = FrontMatterGenerator(front_matter_schema)
        
        # 6. Create the markdown generator
        print("Creating markdown generator...")
        generator = MarkdownGenerator(
            config=config,
            llm_provider=llm_provider,
            prompt_engine=prompt_engine,
            front_matter_generator=front_matter_generator,
        )
        
        # 7. Generate content for a topic
        topic = "python"  # You can change this to any topic defined in the config
        print(f"Generating content for topic: {topic}")
        content = generator.generate_content(topic)
        
        # 8. Write the content to a file
        print("Writing content to file...")
        output_path = generator.write_to_file(content, title=f"Example {topic.capitalize()} Post")
        print(f"Content written to: {output_path}")
        
        # 9. Display token usage information
        token_usage = llm_provider.get_token_usage()
        print("\nToken Usage Information:")
        print(f"  Prompt tokens: {token_usage.prompt_tokens}")
        print(f"  Completion tokens: {token_usage.completion_tokens}")
        print(f"  Total tokens: {token_usage.total_tokens}")
        
        # Show cost information if available
        if token_usage.cost is not None:
            print(f"  Estimated cost: ${token_usage.cost:.6f}")
        
    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()