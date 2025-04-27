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
    RetryConfig,
    ServiceUnavailableError,
    TimeoutError
)
from llm_markdown_generator.front_matter import FrontMatterGenerator
from llm_markdown_generator.generator import MarkdownGenerator
from llm_markdown_generator.llm_provider import GeminiProvider, LLMProvider, OpenAIProvider

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
            class MockProvider(LLMProvider):
                def __init__(self, provider_type: str = "mock", model_name: str = "mock-model", *args: Any, **kwargs: Any) -> None:
                    super().__init__()
                    # For mock provider, set up the environment variable first
                    import os
                    os.environ['MOCK_API_KEY'] = 'mock-key-for-dry-run'
                    # Store mock data
                    self.mock_model = model_name
                    self.mock_provider = provider_type
                
                def generate_text(self, prompt: str) -> str:
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
                    # For mock provider, set up the environment variable first
                    import os
                    os.environ['MOCK_API_KEY'] = 'mock-key-for-dry-run'
                    # Call parent init with minimal parameters
                    super().__init__(
                        model_name=kwargs.get('model_name', 'mock-model'),
                        api_key_env_var='MOCK_API_KEY',
                    )
                    # Override with mock data
                    self.mock_model = kwargs.get('model_name', 'mock-model')
                    self.mock_provider = kwargs.get('provider_type', 'mock')
                
                def generate_text(self, prompt):
                    """Mock implementation that doesn't make API calls."""
                    console.print("[yellow]DRY RUN: Would call LLM API here[/yellow]")
                    console.print(f"[dim]Provider: {self.mock_provider}, Model: {self.mock_model}[/dim]")
                    console.print(f"[dim]Prompt length: {len(prompt)} characters[/dim]")
                    
                    # Extract the CVE ID from the prompt
                    import re
                    cve_match = re.search(r'CVE-\d{4}-\d+', prompt)
                    current_cve = cve_match.group(0) if cve_match else "CVE-XXXX-XXXXX"
                    
                    return f"""### {current_cve}: Critical Remote Code Execution Vulnerability

#### Vulnerability Snapshot
- **CVE ID**: [{current_cve}](https://www.cve.org/CVERecord?id={current_cve})
- **CVSS Score**: 9.8 ([CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H](https://www.first.org/cvss/calculator/3.1#CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H))
- **CVSS Severity**: Critical
- **EPSS Score**: [0.85](https://epss.cyentia.com/) (85% probability of exploitation)
- **CWE Category**: [CWE-787](https://cwe.mitre.org/data/definitions/787.html) (Out-of-bounds Write)
- **Affected Products**: [Example Product 1.0 - 3.2](https://example.com/products)
- **Vulnerability Type**: Remote Code Execution
- **Patch Availability**: [Yes](https://example.com/security/advisory)
- **Exploitation Status**: [PoC Available](https://example.com/security/disclosures)

#### Technical Details
This is a mock CVE report for {current_cve}. In a real run, this would contain detailed information about this vulnerability, including a technical description, attack vectors, and root cause analysis.

#### Exploitation Context
At present, multiple security researchers have developed proof-of-concept exploits demonstrating the vulnerability. While no active exploitation has been confirmed, scanning activity has increased.

#### Impact Assessment
This vulnerability allows attackers to gain unauthorized access to affected systems, resulting in:

- Complete control over the vulnerable system
- Ability to access, modify, or destroy data
- Potential for lateral movement to connected systems

#### Mitigation and Remediation
- Apply vendor patches immediately
- If patching is not immediately possible:
  - Implement network segmentation
  - Enable MFA for all administrative access
  - Monitor logs for suspicious activity
- Detection methods:
  - Monitor for unusual authentication events
  - Watch for unexpected system activities

#### References
- [Vendor Security Advisory](https://example.com/security/advisory)
- [NIST NVD Entry](https://nvd.nist.gov/vuln/detail/{current_cve})
- [Security Researcher Blog](https://example.com/security/blog)
"""
            
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
                    additional_params=config.llm_provider.additional_params,
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
                        additional_params=config.llm_provider.additional_params,
                        retry_config=retry_config,
                    )
                else:
                    llm_provider = GeminiProvider(
                        model_name=model_name,
                        api_key_env_var=config.llm_provider.api_key_env_var,
                        temperature=config.llm_provider.temperature,
                        max_tokens=config.llm_provider.max_tokens,
                        additional_params=config.llm_provider.additional_params,
                        retry_config=retry_config,
                    )
            else:
                raise LLMErrorBase(f"Unsupported LLM provider type: {llm_provider_type}")

        # Create prompt engine
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
        
        # Configure retry strategy
        retry_config = RetryConfig(
            max_retries=max_retries,
            base_delay=1.0,
            backoff_factor=2.0,
            jitter=True
        )
        
        # Determine provider type (with CLI override if provided)
        llm_provider_type = provider.lower() if provider else config.llm_provider.provider_type.lower()

        # Update model name if provided
        model_name = model or config.llm_provider.model_name
        
        if dry_run:
            # Create a mock LLM provider that doesn't make API calls
            class MockProvider(LLMProvider):
                def __init__(self, provider_type: str = "mock", model_name: str = "mock-model", *args: Any, **kwargs: Any) -> None:
                    super().__init__()
                    # For mock provider, set up the environment variable first
                    import os
                    os.environ['MOCK_API_KEY'] = 'mock-key-for-dry-run'
                    # Store mock data
                    self.mock_model = model_name
                    self.mock_provider = provider_type
                
                def generate_text(self, prompt: str) -> str:
                    """Mock implementation that doesn't make API calls."""
                    console.print("[yellow]DRY RUN: Would call LLM API here[/yellow]")
                    console.print(f"[dim]Provider: {self.mock_provider}, Model: {self.mock_model}[/dim]")
                    console.print(f"[dim]Prompt length: {len(prompt)} characters[/dim]")
                    
                    return f"""### {cve_id}: Critical Remote Code Execution Vulnerability in "ExampleCorp SecureFile Transfer Application"

#### Vulnerability Snapshot
- **CVE ID**: [{cve_id}](https://www.cve.org/CVERecord?id={cve_id})
- **CVSS Score**: 9.8 ([CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H](https://www.first.org/cvss/calculator/3.1#CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H))
- **CVSS Severity**: Critical
- **EPSS Score**: [0.92](https://epss.cyentia.com/) (92% probability of exploitation)
- **CWE Category**: [CWE-787](https://cwe.mitre.org/data/definitions/787.html) (Out-of-bounds Write)
- **Affected Products**: ExampleCorp SecureFile Transfer Application v3.0 - v3.5 ([Vendor Advisory](https://examplecorp.com/security-advisories/ESA-2024-001))
- **Vulnerability Type**: Remote Code Execution (RCE)
- **Patch Availability**: [Yes](https://examplecorp.com/downloads/securefile-transfer)
- **Exploitation Status**: [PoC Available](https://github.com/security-researcher/CVE-2024-29896-PoC)


#### Technical Details
{cve_id} is a critical remote code execution vulnerability affecting ExampleCorp's SecureFile Transfer Application. The vulnerability stems from an out-of-bounds write condition within the application's file processing module. Specifically, a specially crafted file name can trigger a buffer overflow, allowing an attacker to overwrite critical memory regions and inject malicious code.

The vulnerability exists due to insufficient bounds checking when parsing filenames provided during file upload requests. An attacker can exploit this flaw by sending a specially crafted filename exceeding the allocated buffer size. This overwrite allows the attacker to control the instruction pointer and execute arbitrary code within the context of the application.

#### Exploitation Context
A proof-of-concept exploit for {cve_id} has been publicly released and is actively being shared within security communities. While widespread exploitation has not yet been confirmed, the ease of exploitation combined with the availability of a public PoC significantly increases the likelihood of imminent attacks.

Given the high EPSS score of 0.92 and the nature of the vulnerability, active exploitation is expected within 24-48 hours. The vulnerability is remotely exploitable without authentication, making it a prime target for automated attacks. Organizations using the affected versions of ExampleCorp SecureFile Transfer Application are strongly urged to apply available patches immediately.

#### Impact Assessment
Successful exploitation of {cve_id} could have severe consequences, including:

- **Complete system compromise:** Attackers can gain full control of the affected server.
- **Data breaches:** Sensitive data transferred through the application could be exfiltrated.
- **Denial of Service:** Attackers could disrupt the availability of the file transfer service.
- **Lateral movement:** Compromised servers can be used as a pivot point for attacks on other internal systems.

The severity of the impact is compounded by the fact that the application is often used to transfer sensitive files, potentially leading to significant data breaches and reputational damage.

#### Mitigation and Remediation
- **Apply patches immediately:** Upgrade to the latest version of ExampleCorp SecureFile Transfer Application (v3.6 or later) available at [https://examplecorp.com/downloads/securefile-transfer](https://examplecorp.com/downloads/securefile-transfer).
- **Workarounds (if patching is not immediately possible):**
    - Disable the affected application if it is not essential.
    - Implement strict network access controls to limit access to the application only to trusted sources.
    - Monitor application logs for suspicious activity.
- **Configuration changes:** None required after patching.
- **Detection methods:**
    - Monitor system logs for unusual process creation or network activity.
    - Implement intrusion detection/prevention systems (IDS/IPS) with signatures designed to detect exploitation attempts. ExampleCorp has released a set of Snort rules for this vulnerability.
    - Analyze network traffic for malicious payloads associated with the exploit.


#### References
- [ExampleCorp Security Advisory ESA-2024-001](https://examplecorp.com/security-advisories/ESA-2024-001)
- [CVE-2024-29896 NVD Entry](https://nvd.nist.gov/vuln/detail/{cve_id}) (Placeholder - will be populated when the CVE is officially published)
- [GitHub Repository with PoC Exploit](https://github.com/security-researcher/{cve_id}-PoC) (Fictional example)
- [Snort Rules for {cve_id}](https://examplecorp.com/security-resources/snort-rules) (Fictional example)
"""
            
            # Create the mock provider
            llm_provider = MockProvider(
                provider_type=llm_provider_type,
                model_name=model_name
            )
            
            console.print(f"[yellow]DRY RUN: Using mock provider ({llm_provider_type})[/yellow]")
        else:
            # Create the actual LLM provider with token tracker
            if llm_provider_type == "openai":
                llm_provider = OpenAIProvider(
                    model_name=model_name,
                    api_key_env_var=config.llm_provider.api_key_env_var,
                    temperature=config.llm_provider.temperature,
                    max_tokens=config.llm_provider.max_tokens,
                    additional_params=config.llm_provider.additional_params,
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
                        additional_params=config.llm_provider.additional_params,
                        retry_config=retry_config,
                    )
                else:
                    llm_provider = GeminiProvider(
                        model_name=model_name,
                        api_key_env_var=config.llm_provider.api_key_env_var,
                        temperature=config.llm_provider.temperature,
                        max_tokens=config.llm_provider.max_tokens,
                        additional_params=config.llm_provider.additional_params,
                        retry_config=retry_config,
                    )
            else:
                raise LLMErrorBase(f"Unsupported LLM provider type: {llm_provider_type}")

        # Create prompt engine with default template location
        templates_dir = ".llmconfig/prompt-templates"  # Default location for templates
        prompt_engine = PromptEngine(templates_dir=templates_dir)
        
        # Create markdown generator
        markdown_generator = MarkdownGenerator(
            config=config,
            llm_provider=llm_provider,
            prompt_engine=prompt_engine,
            front_matter_generator=front_matter_generator
        )
        
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