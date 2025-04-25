"""Command-line interface for the LLM Markdown Generator.

Provides a command-line tool for generating markdown content using LLMs.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import typer

from llm_markdown_generator.config import Config, load_config, load_front_matter_schema
from llm_markdown_generator.front_matter import FrontMatterGenerator
from llm_markdown_generator.generator import MarkdownGenerator
from llm_markdown_generator.llm_provider import GeminiProvider, LLMError, OpenAIProvider
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
            )
        elif llm_provider_type == "gemini":
            # For Gemini, use a default model name rather than the one in config
            # (which might be set for OpenAI)
            gemini_model = "gemini-1.5-flash"  # Using an available model from the API
            
            # For Gemini, prefer direct API key if provided, otherwise use env var
            if api_key:
                llm_provider = GeminiProvider(
                    model_name=gemini_model,
                    api_key=api_key,
                    temperature=config.llm_provider.temperature,
                    max_tokens=config.llm_provider.max_tokens,
                    additional_params=config.llm_provider.additional_params,
                )
            else:
                llm_provider = GeminiProvider(
                    model_name=gemini_model,
                    api_key_env_var=config.llm_provider.api_key_env_var,
                    temperature=config.llm_provider.temperature,
                    max_tokens=config.llm_provider.max_tokens,
                    additional_params=config.llm_provider.additional_params,
                )
        else:
            raise LLMError(f"Unsupported LLM provider type: {llm_provider_type}")

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
        
        # Show token usage details
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

    except Exception as e:
        typer.echo(f"Error: {str(e)}", err=True)
        raise typer.Exit(code=1)


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
