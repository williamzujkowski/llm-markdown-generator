"""Command-line interface for the LLM Markdown Generator.

Provides a command-line tool for generating markdown content using LLMs.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import typer
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Import Pydantic-based configuration (while maintaining backward compatibility)
try:
    from llm_markdown_generator.config_pydantic import Config, load_config, load_front_matter_schema
except ImportError:
    # Fall back to dataclass-based configuration if Pydantic is not available
    from llm_markdown_generator.config import Config, load_config, load_front_matter_schema

from llm_markdown_generator.error_handler import (
    AuthError,
    LLMErrorBase,
    NetworkError,
    RateLimitError,
    RetryConfig,
    ServiceUnavailableError,
    TimeoutError
)
from llm_markdown_generator.front_matter import FrontMatterGenerator
from llm_markdown_generator.generator import MarkdownGenerator
from llm_markdown_generator.llm_provider import GeminiProvider, OpenAIProvider
from llm_markdown_generator.prompt_engine import PromptEngine

app = typer.Typer(help="Generate markdown blog posts using LLMs")


@app.command()
def generate(
    topic: str = typer.Argument(..., help="The topic to generate content for"),
    config_path: str = typer.Option(
        "config/config.yaml", help="Path to the configuration file"
    ),
    output_dir: Optional[str] = typer.Option(
        None, help="Output directory (overrides the one in config)"
    ),
    title: Optional[str] = typer.Option(None, help="Title for the generated post"),
    keywords: Optional[str] = typer.Option(
        None, help="Comma-separated list of keywords to use (adds to those in config)"
    ),
    provider: Optional[str] = typer.Option(
        None, help="Override the LLM provider (openai or gemini)"
    ),
    api_key: Optional[str] = typer.Option(
        None, help="Directly provide an API key instead of using environment variables"
    ),
    plugins: Optional[str] = typer.Option(
        None, help="Comma-separated list of plugins to enable (e.g., 'add_timestamp,add_reading_time')"
    ),
    plugins_dir: Optional[str] = typer.Option(
        None, help="Directory containing custom plugins"
    ),
    no_plugins: bool = typer.Option(
        False, help="Disable loading of plugins"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Display verbose output including token usage"),
    max_retries: int = typer.Option(3, "--retries", "-r", help="Number of retries for API calls"),
    retry_delay: float = typer.Option(1.0, "--retry-delay", help="Base delay in seconds between retries"),
) -> None:
    """Generate a markdown blog post for the specified topic."""
    try:
        # Load configuration
        config = load_config(config_path)

        # Override output directory if provided
        if output_dir:
            config.output_dir = output_dir

        # Load front matter schema
        front_matter_schema = load_front_matter_schema(config.front_matter.schema_path)

        # Configure retry strategy
        retry_config = RetryConfig(
            max_retries=max_retries,
            base_delay=retry_delay,
            backoff_factor=2.0,
            jitter=True
        )
        
        # Determine provider type (with CLI override if provided)
        llm_provider_type = provider.lower() if provider else config.llm_provider.provider_type.lower()

        # Create LLM provider
        if llm_provider_type == "openai":
            llm_provider = OpenAIProvider(
                model_name=config.llm_provider.model_name,
                api_key_env_var=config.llm_provider.api_key_env_var,
                temperature=config.llm_provider.temperature,
                max_tokens=config.llm_provider.max_tokens,
                additional_params=config.llm_provider.additional_params,
                retry_config=retry_config,
            )
        elif llm_provider_type == "gemini":
            # For Gemini, use a default model name rather than the one in config
            # (which might be set for OpenAI)
            gemini_model = "gemini-2.0-flash"  # Latest Gemini model with improved capabilities
            
            # For Gemini, prefer direct API key if provided, otherwise use env var
            if api_key:
                llm_provider = GeminiProvider(
                    model_name=gemini_model,
                    api_key=api_key,
                    temperature=config.llm_provider.temperature,
                    max_tokens=config.llm_provider.max_tokens,
                    additional_params=config.llm_provider.additional_params,
                    retry_config=retry_config,
                )
            else:
                llm_provider = GeminiProvider(
                    model_name=gemini_model,
                    api_key_env_var=config.llm_provider.api_key_env_var,
                    temperature=config.llm_provider.temperature,
                    max_tokens=config.llm_provider.max_tokens,
                    additional_params=config.llm_provider.additional_params,
                    retry_config=retry_config,
                )
        else:
            raise LLMErrorBase(f"Unsupported LLM provider type: {llm_provider_type}")

        # Create prompt engine
        # Assume templates are in .llmconfig/prompt-templates/
        templates_dir = Path(".llmconfig/prompt-templates")
        prompt_engine = PromptEngine(str(templates_dir))

        # Create front matter generator
        front_matter_generator = FrontMatterGenerator(front_matter_schema)

        # Create markdown generator
        generator = MarkdownGenerator(
            config=config,
            llm_provider=llm_provider,
            prompt_engine=prompt_engine,
            front_matter_generator=front_matter_generator,
        )
        
        # Handle plugin loading
        if not no_plugins:
            # Set plugins directory if provided
            if plugins_dir:
                setattr(config, 'plugins_dir', plugins_dir)
            
            # Load all available plugins
            try:
                plugin_counts = generator.load_plugins()
                if verbose:
                    for category, count in plugin_counts.items():
                        if count > 0:
                            typer.echo(f"Loaded {count} {category} plugins")
                
                # Enable specific plugins if requested
                if plugins:
                    from llm_markdown_generator.plugins import get_plugin, PluginError
                    
                    # Clear default plugins if specific ones are requested
                    generator.clear_plugins()
                    
                    plugin_names = [p.strip() for p in plugins.split(',')]
                    for plugin_name in plugin_names:
                        try:
                            # Try content processor category first
                            try:
                                plugin_func = get_plugin('content_processor', plugin_name)
                                generator.register_content_processor(plugin_func)
                                if verbose:
                                    typer.echo(f"Enabled content processor plugin: {plugin_name}")
                            except PluginError:
                                # Try front matter enhancer category next
                                plugin_func = get_plugin('front_matter_enhancer', plugin_name)
                                generator.register_front_matter_enhancer(plugin_func)
                                if verbose:
                                    typer.echo(f"Enabled front matter enhancer plugin: {plugin_name}")
                        except PluginError as e:
                            typer.echo(f"Warning: Plugin '{plugin_name}' not found: {str(e)}", err=True)
            except Exception as e:
                typer.echo(f"Warning: Error loading plugins: {str(e)}", err=True)

        # Prepare custom parameters
        custom_params: Dict[str, Any] = {}
        if title:
            custom_params["title"] = title
        if keywords:
            custom_params["additional_keywords"] = [
                k.strip() for k in keywords.split(",")
            ]

        # Generate content
        typer.echo(f"Generating content for topic: {topic} using {llm_provider_type}")
        content = generator.generate_content(topic, custom_params)

        # Get token usage information
        token_usage = llm_provider.get_token_usage()
        
        # Write to file
        output_path = generator.write_to_file(content, title=title)
        
        # Display results
        typer.echo(f"Content written to: {output_path}")
        
        # Show token usage details only if verbose is True
        if verbose:
            typer.echo("\nToken Usage Information:")
            typer.echo(f"  Prompt tokens: {token_usage.prompt_tokens}")
            typer.echo(f"  Completion tokens: {token_usage.completion_tokens}")
            typer.echo(f"  Total tokens: {token_usage.total_tokens}")
            
            # Show cost information if available
            if token_usage.cost is not None:
                typer.echo(f"  Estimated cost: ${token_usage.cost:.6f}")
            
            # Show accumulated usage
            typer.echo("\nAccumulated Usage:")
            typer.echo(f"  Total tokens: {llm_provider.total_usage.total_tokens}")
            if llm_provider.total_usage.cost is not None:
                typer.echo(f"  Total cost: ${llm_provider.total_usage.cost:.6f}")

    except AuthError as e:
        typer.echo(f"Authentication Error: {str(e)}", err=True)
        typer.echo("Please check your API key or environment variables.", err=True)
        raise typer.Exit(code=1)
        
    except RateLimitError as e:
        retry_msg = f" Try again in {e.retry_after} seconds." if e.retry_after else ""
        typer.echo(f"Rate Limit Error: {str(e)}{retry_msg}", err=True)
        typer.echo("Consider reducing your request frequency or upgrading your API tier.", err=True)
        raise typer.Exit(code=2)
        
    except NetworkError as e:
        typer.echo(f"Network Error: {str(e)}", err=True)
        typer.echo("Please check your internet connection and try again.", err=True)
        raise typer.Exit(code=3)
        
    except TimeoutError as e:
        typer.echo(f"Timeout Error: {str(e)}", err=True)
        typer.echo("The request took too long to complete. Please try again later.", err=True)
        raise typer.Exit(code=4)
        
    except ServiceUnavailableError as e:
        typer.echo(f"Service Unavailable: {str(e)}", err=True)
        typer.echo("The LLM service is currently unavailable. Please try again later.", err=True)
        raise typer.Exit(code=5)
        
    except LLMErrorBase as e:
        typer.echo(f"LLM Error: {str(e)}", err=True)
        if verbose:
            # Add detailed error information in verbose mode
            if hasattr(e, 'category'):
                typer.echo(f"  Category: {e.category.value}")
            if hasattr(e, 'status_code') and e.status_code:
                typer.echo(f"  Status Code: {e.status_code}")
            if hasattr(e, 'request_id') and e.request_id:
                typer.echo(f"  Request ID: {e.request_id}")
        raise typer.Exit(code=10)
        
    except Exception as e:
        typer.echo(f"Unexpected Error: {str(e)}", err=True)
        if verbose:
            import traceback
            typer.echo(traceback.format_exc(), err=True)
        raise typer.Exit(code=99)


def main() -> None:
    """Entry point for the CLI."""
    # Load environment variables from .env file if present
    load_dotenv()
    app()


if __name__ == "__main__":
    main()
