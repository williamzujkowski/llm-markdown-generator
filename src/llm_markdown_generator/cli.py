"""Command-line interface for the LLM Markdown Generator.

Provides a command-line tool for generating markdown content using LLMs.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

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
from llm_markdown_generator.llm_provider import GeminiProvider, OpenAIProvider, TokenUsage
from llm_markdown_generator.prompt_engine import PromptEngine
from llm_markdown_generator.token_tracker import TokenTracker

# Create a Rich console for prettier output
console = Console()

# Create the Typer app
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
    model: Optional[str] = typer.Option(
        None, help="Override the model name (e.g., gpt-4o, gemini-2.0-flash)"
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
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Display verbose output including token usage"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Run without actually calling the LLM API or writing files"
    ),
    max_retries: int = typer.Option(
        3, "--retries", "-r", help="Number of retries for API calls"
    ),
    retry_delay: float = typer.Option(
        1.0, "--retry-delay", help="Base delay in seconds between retries"
    ),
    token_log_path: Optional[str] = typer.Option(
        None, "--token-log", help="Path to log token usage (e.g., 'token_usage.jsonl')"
    ),
    show_usage_report: bool = typer.Option(
        False, "--usage-report", help="Show detailed token usage report at the end"
    ),
    extra_params: Optional[str] = typer.Option(
        None, "--extra", "-e", help="Extra parameters in JSON format (e.g., '{\"temperature\": 0.9}')"
    ),
) -> None:
    """Generate a markdown blog post for the specified topic."""
    try:
        # Process extra parameters if provided
        additional_params = {}
        if extra_params:
            try:
                additional_params = json.loads(extra_params)
                if verbose:
                    console.print(f"Extra parameters: {additional_params}")
            except json.JSONDecodeError:
                console.print("[red]Error: Invalid JSON in extra parameters[/red]", err=True)
                raise typer.Exit(code=1)

        # Set up token tracker
        token_tracker = TokenTracker(log_path=token_log_path)

        # Load configuration
        config = load_config(config_path)

        # Override output directory if provided
        if output_dir:
            config.output_dir = output_dir

        # Create output directory if it doesn't exist
        if not dry_run:
            os.makedirs(config.output_dir, exist_ok=True)

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

        # Update model name if provided
        model_name = model or config.llm_provider.model_name
        
        if dry_run:
            # Create a mock LLM provider that doesn't make API calls
            class MockProvider(OpenAIProvider):
                def __init__(self, *args, **kwargs):
                    # Call parent init with minimal parameters
                    super().__init__(
                        model_name=kwargs.get('model_name', 'mock-model'),
                        api_key_env_var='MOCK_API_KEY',
                    )
                    # Override with mock data
                    self.mock_model = kwargs.get('model_name', 'mock-model')
                    self.mock_provider = kwargs.get('provider_type', 'mock')
                    # Mock token usage
                    self._token_usage = TokenUsage(prompt_tokens=100, completion_tokens=200, 
                                                  total_tokens=300, cost=0.001)
                    self.total_usage = TokenUsage(prompt_tokens=100, completion_tokens=200, 
                                                 total_tokens=300, cost=0.001)
                
                def generate_text(self, prompt):
                    """Mock implementation that doesn't make API calls."""
                    console.print("[yellow]DRY RUN: Would call LLM API here[/yellow]")
                    console.print(f"[dim]Provider: {self.mock_provider}, Model: {self.mock_model}[/dim]")
                    console.print(f"[dim]Prompt length: {len(prompt)} characters[/dim]")
                    
                    return f"""# Mock Response for "{topic}"

This is a mock response for a dry run. No API call was made.

## Sample Content

This is what content would be generated if this were a real API call.

## Mock Details

- Provider: {self.mock_provider}
- Model: {self.mock_model}
- Topic: {topic}
- Configuration: Using {config_path}
"""
                
                def get_token_usage(self):
                    """Return mock token usage."""
                    return self._token_usage
            
            # Create the mock provider
            llm_provider = MockProvider(
                provider_type=llm_provider_type,
                model_name=model_name
            )
            
            console.print(f"[yellow]DRY RUN: Using mock provider ({llm_provider_type})[/yellow]")
        else:
            # Create the actual LLM provider
            if llm_provider_type == "openai":
                llm_provider = OpenAIProvider(
                    model_name=model_name,
                    api_key_env_var=config.llm_provider.api_key_env_var,
                    temperature=config.llm_provider.temperature,
                    max_tokens=config.llm_provider.max_tokens,
                    additional_params={**config.llm_provider.additional_params, **additional_params},
                    retry_config=retry_config,
                )
            elif llm_provider_type == "gemini":
                # For Gemini, prefer direct API key if provided, otherwise use env var
                if api_key:
                    llm_provider = GeminiProvider(
                        model_name=model_name,
                        api_key=api_key,
                        temperature=config.llm_provider.temperature,
                        max_tokens=config.llm_provider.max_tokens,
                        additional_params={**config.llm_provider.additional_params, **additional_params},
                        retry_config=retry_config,
                    )
                else:
                    llm_provider = GeminiProvider(
                        model_name=model_name,
                        api_key_env_var=config.llm_provider.api_key_env_var,
                        temperature=config.llm_provider.temperature,
                        max_tokens=config.llm_provider.max_tokens,
                        additional_params={**config.llm_provider.additional_params, **additional_params},
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
        plugin_info = []
        if not no_plugins:
            # Set plugins directory if provided
            if plugins_dir:
                setattr(config, 'plugins_dir', plugins_dir)
            
            # Load all available plugins
            try:
                plugin_counts = generator.load_plugins()
                for category, count in plugin_counts.items():
                    if count > 0:
                        plugin_info.append(f"Loaded {count} {category} plugins")
                
                # Enable specific plugins if requested
                if plugins:
                    from llm_markdown_generator.plugins import get_plugin, list_plugins, PluginError
                    
                    # Clear default plugins if specific ones are requested
                    generator.clear_plugins()
                    
                    plugin_names = [p.strip() for p in plugins.split(',')]
                    enabled_plugins = []
                    
                    # Get all available plugins
                    all_plugins = list_plugins()
                    
                    for plugin_name in plugin_names:
                        found = False
                        # Check in each category
                        for category, category_plugins in all_plugins.items():
                            if plugin_name in category_plugins:
                                try:
                                    plugin_func = get_plugin(category, plugin_name)
                                    if category == 'content_processor':
                                        generator.register_content_processor(plugin_func)
                                    else:
                                        generator.register_front_matter_enhancer(plugin_func)
                                    
                                    enabled_plugins.append(f"{plugin_name} ({category})")
                                    found = True
                                    break
                                except PluginError as e:
                                    console.print(f"[yellow]Warning: Error enabling plugin '{plugin_name}': {str(e)}[/yellow]", err=True)
                        
                        if not found:
                            console.print(f"[yellow]Warning: Plugin '{plugin_name}' not found in any category[/yellow]", err=True)
                    
                    if enabled_plugins:
                        plugin_info.append(f"Enabled plugins: {', '.join(enabled_plugins)}")
            except Exception as e:
                console.print(f"[yellow]Warning: Error loading plugins: {str(e)}[/yellow]", err=True)

        # Display plugin information if verbose or there's something to report
        if verbose and plugin_info:
            for info in plugin_info:
                console.print(f"[blue]{info}[/blue]")

        # Prepare custom parameters
        custom_params: Dict[str, Any] = {}
        if title:
            custom_params["title"] = title
        if keywords:
            custom_params["additional_keywords"] = [
                k.strip() for k in keywords.split(",")
            ]
        
        # Add any extra parameters from the command line
        custom_params.update(additional_params)

        # Generate content
        with console.status(f"Generating content for topic: [bold]{topic}[/bold] using [bold]{llm_provider_type}/{model_name}[/bold]..."):
            start_time = datetime.now()
            content = generator.generate_content(topic, custom_params)
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()

        # Get token usage information
        token_usage = llm_provider.get_token_usage()
        
        # Record token usage
        token_tracker.record_usage(
            token_usage=token_usage,
            provider=llm_provider_type,
            model=model_name,
            operation="generate_content",
            topic=topic,
            metadata={
                "title": title or topic,
                "generation_time": generation_time,
                "dry_run": dry_run,
                **custom_params
            }
        )
        
        if dry_run:
            console.print("[yellow]DRY RUN: Would write content to file[/yellow]")
            # Print a preview of the content
            content_preview = content.split("\n\n")[0] + "\n..."
            console.print(f"[dim]Content preview:[/dim]\n{content_preview}")
        else:
            # Write to file
            output_path = generator.write_to_file(content, title=title)
            console.print(f"[green]Content written to:[/green] {output_path}")
        
        # Show token usage details
        tokens_table = Table(title="Token Usage Information")
        tokens_table.add_column("Metric", style="cyan")
        tokens_table.add_column("Value", style="green")
        
        tokens_table.add_row("Provider", llm_provider_type)
        tokens_table.add_row("Model", model_name)
        tokens_table.add_row("Prompt tokens", str(token_usage.prompt_tokens))
        tokens_table.add_row("Completion tokens", str(token_usage.completion_tokens))
        tokens_table.add_row("Total tokens", str(token_usage.total_tokens))
        
        if token_usage.cost is not None:
            tokens_table.add_row("Estimated cost", f"${token_usage.cost:.6f}")
        
        tokens_table.add_row("Generation time", f"{generation_time:.2f} seconds")
        
        if verbose or show_usage_report:
            console.print(tokens_table)
            
            # Show accumulated usage
            total_usage = token_tracker.get_total_usage()
            console.print("\n[bold]Session Token Usage:[/bold]")
            console.print(f"  Total operations: {len(token_tracker.records)}")
            console.print(f"  Total tokens: {total_usage.total_tokens}")
            if total_usage.cost is not None:
                console.print(f"  Total cost: ${total_usage.cost:.6f}")
            
            # Display detailed usage report if requested
            if show_usage_report:
                console.print("\n" + token_tracker.generate_report(detailed=True))

    except AuthError as e:
        console.print(f"[red]Authentication Error:[/red] {str(e)}", err=True)
        console.print("[yellow]Please check your API key or environment variables.[/yellow]", err=True)
        raise typer.Exit(code=1)
        
    except RateLimitError as e:
        retry_msg = f" Try again in {e.retry_after} seconds." if e.retry_after else ""
        console.print(f"[red]Rate Limit Error:[/red] {str(e)}{retry_msg}", err=True)
        console.print("[yellow]Consider reducing your request frequency or upgrading your API tier.[/yellow]", err=True)
        raise typer.Exit(code=2)
        
    except NetworkError as e:
        console.print(f"[red]Network Error:[/red] {str(e)}", err=True)
        console.print("[yellow]Please check your internet connection and try again.[/yellow]", err=True)
        raise typer.Exit(code=3)
        
    except TimeoutError as e:
        console.print(f"[red]Timeout Error:[/red] {str(e)}", err=True)
        console.print("[yellow]The request took too long to complete. Please try again later.[/yellow]", err=True)
        raise typer.Exit(code=4)
        
    except ServiceUnavailableError as e:
        console.print(f"[red]Service Unavailable:[/red] {str(e)}", err=True)
        console.print("[yellow]The LLM service is currently unavailable. Please try again later.[/yellow]", err=True)
        raise typer.Exit(code=5)
        
    except LLMErrorBase as e:
        console.print(f"[red]LLM Error:[/red] {str(e)}", err=True)
        if verbose:
            # Add detailed error information in verbose mode
            if hasattr(e, 'category'):
                console.print(f"  Category: {e.category.value}")
            if hasattr(e, 'status_code') and e.status_code:
                console.print(f"  Status Code: {e.status_code}")
            if hasattr(e, 'request_id') and e.request_id:
                console.print(f"  Request ID: {e.request_id}")
        raise typer.Exit(code=10)
        
    except Exception as e:
        console.print(f"[red]Unexpected Error:[/red] {str(e)}", err=True)
        if verbose:
            import traceback
            console.print(traceback.format_exc(), err=True)
        raise typer.Exit(code=99)


@app.command()
def usage_report(
    log_path: str = typer.Argument(..., help="Path to the token usage log file"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed usage records"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save report to file instead of displaying"),
) -> None:
    """Generate a token usage report from a log file."""
    try:
        # Check if log file exists
        if not os.path.exists(log_path):
            console.print(f"[red]Error: Log file not found at {log_path}[/red]", err=True)
            raise typer.Exit(code=1)
            
        # Create token tracker and load from log file
        token_tracker = TokenTracker(log_path=log_path)
        
        # Generate the report
        report = token_tracker.generate_report(detailed=detailed)
        
        if output:
            # Save to file
            with open(output, "w") as f:
                f.write(report)
            console.print(f"[green]Report saved to:[/green] {output}")
        else:
            # Display the report
            console.print(report)
            
    except Exception as e:
        console.print(f"[red]Error generating usage report:[/red] {str(e)}", err=True)
        raise typer.Exit(code=1)


def main() -> None:
    """Entry point for the CLI."""
    # Load environment variables from .env file if present
    load_dotenv()
    app()


if __name__ == "__main__":
    main()
