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

# Import dataclass-based configuration (instead of Pydantic for now - addressing compatibility issues)
from llm_markdown_generator.config import Config, TopicConfig, load_config, load_front_matter_schema

from llm_markdown_generator.error_handler import (
    AuthError,
    LLMErrorBase,
    NetworkError,
    RateLimitError,
    # RetryConfig, # Removed - LangChain handles retries
    ServiceUnavailableError,
    TimeoutError
)
# LangChain components
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel

from llm_markdown_generator.front_matter import FrontMatterGenerator
from llm_markdown_generator.generator import MarkdownGenerator
# LLMProvider, OpenAIProvider, GeminiProvider are removed
# PromptEngine is removed (will be removed later if still present)

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
        False, "--verbose", "-v", help="Display verbose output including generation details"
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
                console.print("[red]Error: Invalid JSON in extra parameters[/red]", style="red")
                raise typer.Exit(code=1)

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

        # RetryConfig removed - LangChain handles retries
        
        # Determine provider type (with CLI override if provided)
        llm_provider_type = provider.lower() if provider else config.llm_provider.provider_type.lower()

        # Update model name if provided
        model_name = model or config.llm_provider.model_name
        
        llm_model: Optional[BaseChatModel] = None # Initialize llm_model
        if dry_run:
            # Dry run: Set llm_model to None. The generator handles the rest.
            llm_model = None
            console.print(f"[yellow]DRY RUN: Skipping LLM instantiation[/yellow]")
        else:
            # Create the actual LangChain ChatModel
            llm_model: Optional[BaseChatModel] = None
            model_kwargs = {
                "model": model_name, # Use 'model' for Google, 'model_name' for OpenAI
                "temperature": config.llm_provider.temperature,
                **{k: v for k, v in config.llm_provider.additional_params.items()}, # Add base params from config
                **{k: v for k, v in additional_params.items()} # Add CLI extra params
            }
            if config.llm_provider.max_tokens:
                 model_kwargs["max_tokens"] = config.llm_provider.max_tokens # Common param name

            if llm_provider_type == "openai":
                # OpenAI uses model_name, LangChain handles API key via env var OPENAI_API_KEY
                 model_kwargs["model_name"] = model_kwargs.pop("model") # Rename key
                 if api_key: # Allow direct API key override
                     model_kwargs["api_key"] = api_key
                 llm_model = ChatOpenAI(**model_kwargs)
                 console.print(f"Using LangChain OpenAI model: {model_name}")

            elif llm_provider_type == "gemini":
                 # Google uses model, LangChain handles API key via env var GOOGLE_API_KEY or direct param
                 if api_key: # Allow direct API key override
                     model_kwargs["google_api_key"] = api_key
                 llm_model = ChatGoogleGenerativeAI(**model_kwargs)
                 console.print(f"Using LangChain Google Gemini model: {model_name}")
            else:
                raise LLMErrorBase(f"Unsupported LLM provider type: {llm_provider_type}")

        # Front matter generator (will be updated later to use Pydantic)
        front_matter_generator = FrontMatterGenerator(front_matter_schema)

        # Create markdown generator with the LangChain model
        generator = MarkdownGenerator(
            config=config,
            llm_model=llm_model, # Pass the instantiated LangChain model
            front_matter_generator=front_matter_generator,
        )
        # Set dry run mode on the generator instance
        generator.set_dry_run(dry_run)

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
                                    console.print(f"[yellow]Warning: Error enabling plugin '{plugin_name}': {str(e)}[/yellow]", style="yellow")
                        
                        if not found:
                            console.print(f"[yellow]Warning: Plugin '{plugin_name}' not found in any category[/yellow]", style="yellow")
                    
                    if enabled_plugins:
                        plugin_info.append(f"Enabled plugins: {', '.join(enabled_plugins)}")
            except Exception as e:
                console.print(f"[yellow]Warning: Error loading plugins: {str(e)}[/yellow]", style="yellow")

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
        
        if dry_run:
            console.print("[yellow]DRY RUN: Would write content to file[/yellow]")
            # Print a preview of the content
            content_preview = content.split("\n\n")[0] + "\n..."
            console.print(f"[dim]Content preview:[/dim]\n{content_preview}")
        else:
            # Write to file
            output_path = generator.write_to_file(content, title=title)
            console.print(f"[green]Content written to:[/green] {output_path}")
        
        # Display generation time
        if verbose:
            console.print(f"Generation time: {generation_time:.2f} seconds")

    except AuthError as e:
        console.print(f"[red]Authentication Error:[/red] {str(e)}", style="red")
        console.print("[yellow]Please check your API key or environment variables.[/yellow]", style="yellow")
        raise typer.Exit(code=1)
        
    except RateLimitError as e:
        retry_msg = f" Try again in {e.retry_after} seconds." if e.retry_after else ""
        console.print(f"[red]Rate Limit Error:[/red] {str(e)}{retry_msg}", style="red")
        console.print("[yellow]Consider reducing your request frequency or upgrading your API tier.[/yellow]", style="yellow")
        raise typer.Exit(code=2)
        
    except NetworkError as e:
        console.print(f"[red]Network Error:[/red] {str(e)}", style="red")
        console.print("[yellow]Please check your internet connection and try again.[/yellow]", style="yellow")
        raise typer.Exit(code=3)
        
    except TimeoutError as e:
        console.print(f"[red]Timeout Error:[/red] {str(e)}", style="red")
        console.print("[yellow]The request took too long to complete. Please try again later.[/yellow]", style="yellow")
        raise typer.Exit(code=4)
        
    except ServiceUnavailableError as e:
        console.print(f"[red]Service Unavailable:[/red] {str(e)}", style="red")
        console.print("[yellow]The LLM service is currently unavailable. Please try again later.[/yellow]", style="yellow")
        raise typer.Exit(code=5)
        
    except LLMErrorBase as e:
        console.print(f"[red]LLM Error:[/red] {str(e)}", style="red")
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
        console.print(f"[red]Unexpected Error:[/red] {str(e)}", style="red")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(code=99)


@app.command()
def generate_cve_report(
    cve_ids: List[str] = typer.Argument(..., help="One or more CVE identifiers (e.g., CVE-2024-45410 CVE-2023-1234)"),
    config_path: str = typer.Option(
        "config/config.yaml", help="Path to the configuration file"
    ),
    output_dir: Optional[str] = typer.Option(
        None, help="Output directory (overrides the one in config)"
    ),
    title_prefix: Optional[str] = typer.Option(
        "Security Advisory", help="Prefix for the report titles"
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
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Display verbose output including generation details"
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
) -> None:
    """Generate comprehensive security advisories for the specified CVE IDs."""
    try:
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

        # RetryConfig removed - LangChain handles retries
        
        # Determine provider type (with CLI override if provided)
        llm_provider_type = provider.lower() if provider else config.llm_provider.provider_type.lower()

        # Update model name if provided
        model_name = model or config.llm_provider.model_name
        
        llm_model: Optional[BaseChatModel] = None # Initialize llm_model
        if dry_run:
            # Dry run: Set llm_model to None. The generator handles the rest.
            llm_model = None
            console.print(f"[yellow]DRY RUN: Skipping LLM instantiation[/yellow]")
        else:
             # Create the actual LangChain ChatModel
            llm_model: Optional[BaseChatModel] = None
            model_kwargs = {
                "model": model_name, # Use 'model' for Google, 'model_name' for OpenAI
                "temperature": config.llm_provider.temperature,
                **{k: v for k, v in config.llm_provider.additional_params.items()}, # Add base params from config
                # No additional CLI params for this command yet
            }
            if config.llm_provider.max_tokens:
                 model_kwargs["max_tokens"] = config.llm_provider.max_tokens # Common param name

            if llm_provider_type == "openai":
                 model_kwargs["model_name"] = model_kwargs.pop("model") # Rename key
                 if api_key: # Allow direct API key override
                     model_kwargs["api_key"] = api_key
                 llm_model = ChatOpenAI(**model_kwargs)
                 console.print(f"Using LangChain OpenAI model: {model_name}")

            elif llm_provider_type == "gemini":
                 if api_key: # Allow direct API key override
                     model_kwargs["google_api_key"] = api_key
                 llm_model = ChatGoogleGenerativeAI(**model_kwargs)
                 console.print(f"Using LangChain Google Gemini model: {model_name}")
            else:
                raise LLMErrorBase(f"Unsupported LLM provider type: {llm_provider_type}")

        # Front matter generator (will be updated later to use Pydantic)
        front_matter_generator = FrontMatterGenerator(front_matter_schema)

        # Create markdown generator with the LangChain model
        generator = MarkdownGenerator(
            config=config,
            llm_model=llm_model, # Pass the instantiated LangChain model
            front_matter_generator=front_matter_generator,
        )
        # Set dry run mode on the generator instance
        generator.set_dry_run(dry_run)

        # Set up the temporary topic config for security advisory
        topic_name = "security_advisory"
        
        # Create a report for each CVE ID
        successful_reports = []
        failed_reports = []
        
        console.print(f"Generating reports for [bold cyan]{len(cve_ids)}[/bold cyan] CVE IDs")
        
        for cve_id in cve_ids:
            console.print(f"\n[bold]Processing: {cve_id}[/bold]")
            
            # Set a specific title for this CVE
            report_title = f"{title_prefix}: {cve_id} - Critical Vulnerability Report"
            
            # Prepare custom parameters for this CVE
            custom_params = {
                "topic": cve_id,
                "title": report_title,
                "audience": "security professionals and IT administrators",
                "keywords": ["cybersecurity", "vulnerability", "CVSS", "EPSS", "mitigation", "remediation", cve_id],
            }
            
            # Update topic config for this CVE
            keywords = list(custom_params["keywords"]) if "keywords" in custom_params else []
            config.topics[topic_name] = TopicConfig(
                name=topic_name,
                prompt_template="security_advisory.j2",
                keywords=keywords,
                custom_data={}
            )
            
            try:
                # Generate content for this CVE
                with console.status(f"Generating report for: [bold]{cve_id}[/bold] using [bold]{llm_provider_type}/{model_name}[/bold]..."):
                    start_time = datetime.now()
                    content = generator.generate_content(topic_name, custom_params)
                    end_time = datetime.now()
                    generation_time = (end_time - start_time).total_seconds()
                
                if dry_run:
                    console.print("[yellow]DRY RUN: Would write content to file[/yellow]")
                    # Print a preview of the content
                    content_preview = "\n".join(content.split("\n")[:10]) + "\n..."
                    console.print(f"[dim]Content preview:[/dim]\n{content_preview}")
                else:
                    # Write to file with CVE-specific filename
                    output_path = generator.write_to_file(content, title=f"{cve_id.lower()}-vulnerability-report")
                    console.print(f"[green]Report written to:[/green] {output_path}")
                    successful_reports.append({"cve_id": cve_id, "path": output_path})
                
                # Show generation time for this CVE if verbose
                if verbose:
                    console.print(f"Generation time: {generation_time:.2f} seconds")
            
            except Exception as e:
                console.print(f"[red]Error generating report for {cve_id}:[/red] {str(e)}")
                failed_reports.append({"cve_id": cve_id, "error": str(e)})
                if verbose:
                    import traceback
                    console.print(traceback.format_exc())
        
        # Print summary of all reports
        console.print("\n[bold]Report Generation Summary:[/bold]")
        if successful_reports:
            console.print(f"[green]Successfully generated {len(successful_reports)} reports:[/green]")
            for report in successful_reports:
                console.print(f"  - {report['cve_id']}")
        
        if failed_reports:
            console.print(f"[red]Failed to generate {len(failed_reports)} reports:[/red]")
            for report in failed_reports:
                console.print(f"  - {report['cve_id']}: {report['error']}")

    except AuthError as e:
        console.print(f"[red]Authentication Error:[/red] {str(e)}", style="red")
        console.print("[yellow]Please check your API key or environment variables.[/yellow]", style="yellow")
        raise typer.Exit(code=1)
        
    except RateLimitError as e:
        retry_msg = f" Try again in {e.retry_after} seconds." if e.retry_after else ""
        console.print(f"[red]Rate Limit Error:[/red] {str(e)}{retry_msg}", style="red")
        console.print("[yellow]Consider reducing your request frequency or upgrading your API tier.[/yellow]", style="yellow")
        raise typer.Exit(code=2)
        
    except Exception as e:
        console.print(f"[red]Error generating CVE report:[/red] {str(e)}", style="red")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(code=1)


@app.command()
def enhanced_cve_report(
    cve_id: str = typer.Argument(..., help="CVE identifier (e.g., CVE-2024-29896)"),
    config_path: str = typer.Option(
        "config/config.yaml", help="Path to the configuration file"
    ),
    front_matter_schema: str = typer.Option(
        "config/cve_front_matter_schema.yaml", help="Path to the CVE front matter schema"
    ),
    output_dir: str = typer.Option(
        "output/vulnerabilities", help="Output directory"
    ),
    title: Optional[str] = typer.Option(
        None, help="Custom title for the report (default: auto-generated from CVE ID)"
    ),
    severity: str = typer.Option(
        "Critical", help="CVSS severity level", 
        show_default=True
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
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Display verbose output including generation details"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Run without actually calling the LLM API or writing files"
    ),
    max_retries: int = typer.Option(
        3, "--retries", "-r", help="Number of retries for API calls"
    ),
) -> None:
    """Generate a CVE vulnerability report with enhanced front matter.
    
    This command uses the improved CVE front matter schema and extract plugin to generate
    comprehensive, structured security advisories with detailed metadata.
    """
    try:
        # Load main configuration
        config = load_config(config_path)
        
        # Load CVE-specific front matter schema
        try:
            cve_schema = load_front_matter_schema(front_matter_schema)
            if verbose:
                console.print(f"Loaded CVE front matter schema from {front_matter_schema}")
        except Exception as e:
            console.print(f"[red]Error loading CVE front matter schema: {str(e)}[/red]")
            console.print("[yellow]Falling back to default front matter schema[/yellow]")
            cve_schema = load_front_matter_schema(config.front_matter.schema_path)
        
        # Create front matter generator with CVE schema
        front_matter_generator = FrontMatterGenerator(schema=cve_schema)
        
        # RetryConfig removed - LangChain handles retries
        
        # Determine provider type (with CLI override if provided)
        llm_provider_type = provider.lower() if provider else config.llm_provider.provider_type.lower()

        # Update model name if provided
        model_name = model or config.llm_provider.model_name
        
        llm_model: Optional[BaseChatModel] = None # Initialize llm_model
        if dry_run:
            # Dry run: Set llm_model to None. The generator handles the rest.
            llm_model = None
            console.print(f"[yellow]DRY RUN: Skipping LLM instantiation[/yellow]")
        else:
            # Create the actual LangChain ChatModel
            llm_model: Optional[BaseChatModel] = None
            model_kwargs = {
                "model": model_name, # Use 'model' for Google, 'model_name' for OpenAI
                "temperature": config.llm_provider.temperature,
                **{k: v for k, v in config.llm_provider.additional_params.items()}, # Add base params from config
                # No additional CLI params for this command yet
            }
            if config.llm_provider.max_tokens:
                 model_kwargs["max_tokens"] = config.llm_provider.max_tokens # Common param name

            if llm_provider_type == "openai":
                 model_kwargs["model_name"] = model_kwargs.pop("model") # Rename key
                 if api_key: # Allow direct API key override
                     model_kwargs["api_key"] = api_key
                 llm_model = ChatOpenAI(**model_kwargs)
                 console.print(f"Using LangChain OpenAI model: {model_name}")

            elif llm_provider_type == "gemini":
                 if api_key: # Allow direct API key override
                     model_kwargs["google_api_key"] = api_key
                 llm_model = ChatGoogleGenerativeAI(**model_kwargs)
                 console.print(f"Using LangChain Google Gemini model: {model_name}")
            else:
                raise LLMErrorBase(f"Unsupported LLM provider type: {llm_provider_type}")

        # Front matter generator (will be updated later to use Pydantic)
        front_matter_generator = FrontMatterGenerator(schema=cve_schema) # Use the correct schema

        # Create markdown generator with the LangChain model
        markdown_generator = MarkdownGenerator(
            config=config,
            llm_model=llm_model, # Pass the instantiated LangChain model
            front_matter_generator=front_matter_generator
        )
        # Set dry run mode on the generator instance
        markdown_generator.set_dry_run(dry_run)

        # Ensure our CVE front matter enhancer plugin is loaded
        # This will extract details from the content and add them to front matter
        try:
            # Import here to ensure it's registered
            import llm_markdown_generator.plugins.cve_front_matter_enhancer
            
            # Load all available plugins
            plugins_loaded = markdown_generator.load_plugins()
            
            if verbose:
                console.print(f"[blue]Loaded plugins: {plugins_loaded}[/blue]")
                
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load CVE enhancer plugin: {str(e)}[/yellow]")
        
        # Prepare default title if not provided
        if not title:
            title = f"{cve_id}: Critical Vulnerability Assessment"
            
        # Prepare front matter with initial values
        # The CVE enhancer plugin will extract and add more fields from the generated content
        front_matter_data = {
            "title": title,
            "cveId": cve_id,
            "publishDate": datetime.now().strftime("%Y-%m-%d"),
            "cvssSeverity": severity,
            "tags": ["cybersecurity", "vulnerability", "CVE", cve_id],
            "author": "AI Content Generator"
        }
        
        # Prepare custom parameters for the security_advisory template
        custom_params = {
            "topic": cve_id,  # The CVE ID is used as the topic
            "title": title,
            "audience": "security professionals and IT administrators",
            "keywords": [
                "cybersecurity",
                "vulnerability",
                "CVSS",
                "EPSS",
                "mitigation",
                "remediation",
                cve_id
            ],
            "front_matter": front_matter_data
        }
        
        # Update the config with security_advisory topic if not present
        if "security_advisory" not in config.topics:
            config.topics["security_advisory"] = TopicConfig(
                name="security_advisory",
                prompt_template="security_advisory.j2",
                keywords=custom_params["keywords"]
            )
        
        # Display prompt if verbose
        if verbose:
            console.print("\n[blue]Prompt Template:[/blue] security_advisory.j2")
            console.print(f"[blue]CVE ID:[/blue] {cve_id}")
            console.print(f"[blue]Front Matter Schema:[/blue] {front_matter_schema}")
        
        # Generate content or mock content for dry-run
        with console.status(f"Generating vulnerability report for [bold]{cve_id}[/bold] using [bold]{llm_provider_type}/{model_name}[/bold]..."):
            start_time = datetime.now()
            
            # Generate the content
            try:
                markdown_content = markdown_generator.generate_content(
                    topic_name="security_advisory",
                    custom_params=custom_params
                )
                end_time = datetime.now()
                generation_time = (end_time - start_time).total_seconds()
            except Exception as e:
                console.print(f"[red]Error generating content: {str(e)}[/red]")
                if verbose:
                    import traceback
                    console.print(traceback.format_exc())
                raise typer.Exit(code=1)
            
        # Ensure output directory exists
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename from CVE ID
        filename = f"{cve_id.lower()}.md"
        
        if dry_run:
            console.print("[yellow]DRY RUN: Would write content to file[/yellow]")
            # Print a preview of the content
            content_preview = "\n".join(markdown_content.split("\n")[:20]) + "\n..."
            console.print(f"[dim]Content preview:[/dim]\n{content_preview}")
        else:
            # Write content to file
            output_path = output_dir_path / filename
            with open(output_path, "w") as f:
                f.write(markdown_content)
            console.print(f"[green]Report written to:[/green] {output_path}")
        
        # Display generation time
        if verbose:
            console.print(f"Generation time: {generation_time:.2f} seconds")
            
        console.print(f"\n[bold green]Successfully generated vulnerability report for {cve_id}[/bold green]")
        console.print(f"\n[bold]Next steps:[/bold]")
        console.print(f"  1. Review the generated report" + (f" at {output_path}" if not dry_run else ""))
        console.print(f"  2. Make any necessary edits or adjustments to the content")
        console.print(f"  3. Use the enhanced front matter for indexing and filtering in your applications")

    except Exception as e:
        console.print(f"[red]Error generating enhanced CVE report:[/red] {str(e)}", style="red")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(code=1)


def main() -> None:
    """Entry point for the CLI."""
    # Load environment variables from .env file if present
    load_dotenv()
    app()


if __name__ == "__main__":
    main()
