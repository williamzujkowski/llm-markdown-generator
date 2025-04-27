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
from llm_markdown_generator.llm_provider import GeminiProvider, LLMProvider, OpenAIProvider, TokenUsage
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
                console.print("[red]Error: Invalid JSON in extra parameters[/red]", style="red")
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
            class MockProvider(LLMProvider):
                def __init__(self, provider_type: str = "mock", model_name: str = "mock-model", *args: Any, **kwargs: Any) -> None:
                    super().__init__()
                    # For mock provider, set up the environment variable first
                    import os
                    os.environ['MOCK_API_KEY'] = 'mock-key-for-dry-run'
                    # Store mock data
                    self.mock_model = model_name
                    self.mock_provider = provider_type
                    # Mock token usage
                    self._token_usage = TokenUsage(prompt_tokens=100, completion_tokens=200, 
                                                  total_tokens=300, cost=0.001)
                    self.total_usage = TokenUsage(prompt_tokens=100, completion_tokens=200, 
                                                 total_tokens=300, cost=0.001)
                
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
                
                def get_token_usage(self) -> TokenUsage:
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
) -> None:
    """Generate comprehensive security advisories for the specified CVE IDs."""
    try:
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
        total_token_usage = TokenUsage()
        
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
                
                # Get token usage information for this CVE
                token_usage = llm_provider.get_token_usage()
                
                # Record token usage for this CVE
                token_tracker.record_usage(
                    token_usage=token_usage,
                    provider=llm_provider_type,
                    model=model_name,
                    operation="generate_cve_report",
                    topic=cve_id,
                    metadata={
                        "title": report_title,
                        "generation_time": generation_time,
                        "dry_run": dry_run
                    }
                )
                
                # Update total token usage
                total_token_usage.prompt_tokens += token_usage.prompt_tokens
                total_token_usage.completion_tokens += token_usage.completion_tokens
                total_token_usage.total_tokens += token_usage.total_tokens
                if total_token_usage.cost is None:
                    total_token_usage.cost = 0
                if token_usage.cost is not None:
                    total_token_usage.cost += token_usage.cost
                
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
                
                # Show brief token usage for this CVE if verbose
                if verbose:
                    console.print(f"Tokens: {token_usage.total_tokens} | Time: {generation_time:.2f}s")
            
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
        
        # Show token usage details
        tokens_table = Table(title="Total Token Usage Information")
        tokens_table.add_column("Metric", style="cyan")
        tokens_table.add_column("Value", style="green")
        
        tokens_table.add_row("Provider", llm_provider_type)
        tokens_table.add_row("Model", model_name)
        tokens_table.add_row("Total prompt tokens", str(total_token_usage.prompt_tokens))
        tokens_table.add_row("Total completion tokens", str(total_token_usage.completion_tokens))
        tokens_table.add_row("Total tokens", str(total_token_usage.total_tokens))
        
        if total_token_usage.cost is not None:
            tokens_table.add_row("Estimated total cost", f"${total_token_usage.cost:.6f}")
        
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
def usage_report(
    log_path: str = typer.Argument(..., help="Path to the token usage log file"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed usage records"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save report to file instead of displaying"),
) -> None:
    """Generate a token usage report from a log file."""
    try:
        # Check if log file exists
        if not os.path.exists(log_path):
            console.print(f"[red]Error: Log file not found at {log_path}[/red]", style="red")
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
        console.print(f"[red]Error generating usage report:[/red] {str(e)}", style="red")
        raise typer.Exit(code=1)


def main() -> None:
    """Entry point for the CLI."""
    # Load environment variables from .env file if present
    load_dotenv()
    app()


if __name__ == "__main__":
    main()
