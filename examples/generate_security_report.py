"""
Example script for generating security-focused content using the LLM Markdown Generator.

This example demonstrates how to generate daily CVE reports and security advisories
for high-risk vulnerabilities with CVSS scores of 9.0 or higher and high EPSS scores.
"""

import os
import argparse
import datetime
from pathlib import Path

from llm_markdown_generator.config import load_config, TopicConfig
from llm_markdown_generator.llm_provider import OpenAIProvider, GeminiProvider
from llm_markdown_generator.prompt_engine import PromptEngine
from llm_markdown_generator.front_matter import FrontMatterGenerator
from llm_markdown_generator.generator import MarkdownGenerator
from llm_markdown_generator.token_tracker import TokenTracker
from llm_markdown_generator.plugins import list_plugins, get_plugin


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate security reports with LLMs")
    parser.add_argument(
        "report_type", 
        type=str, 
        choices=["daily_cve_report", "security_advisory"],
        help="Type of security report to generate"
    )
    parser.add_argument(
        "topic", 
        type=str, 
        help="For daily_cve_report: date (YYYY-MM-DD) or 'today'; For security_advisory: specific CVE or vulnerability topic"
    )
    parser.add_argument("--title", type=str, help="Custom title for the report")
    parser.add_argument(
        "--cves", 
        type=str, 
        help="Comma-separated list of CVE IDs to include (for security_advisory)"
    )
    parser.add_argument(
        "--keywords", 
        type=str, 
        default="cybersecurity,vulnerability,CVSS,EPSS,remediation,mitigation,CVE",
        help="Comma-separated keywords"
    )
    parser.add_argument(
        "--audience", 
        type=str, 
        default="security teams and IT administrators",
        help="Target audience"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="output/security",
        help="Output directory"
    )
    parser.add_argument(
        "--config-path", 
        type=str, 
        default="config/config.yaml",
        help="Path to config file"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Run without making API calls")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config_path)
    
    # Prepare context for template rendering
    context = {
        "keywords": [kw.strip() for kw in args.keywords.split(",")]
    }
    
    # Handle report-specific parameters
    if args.report_type == "daily_cve_report":
        if args.topic.lower() == "today":
            report_date = datetime.datetime.now().strftime("%Y-%m-%d")
        else:
            # Validate date format
            try:
                datetime.datetime.strptime(args.topic, "%Y-%m-%d")
                report_date = args.topic
            except ValueError:
                print("Error: Date must be in YYYY-MM-DD format or 'today'")
                return
        
        context["report_date"] = report_date
        context["topic"] = f"critical vulnerabilities published on {report_date}"
        
        if args.title:
            context["title"] = args.title
        else:
            context["title"] = f"Critical Vulnerability Alert: Daily CVE Report ({report_date})"
            
    elif args.report_type == "security_advisory":
        context["topic"] = args.topic
        
        if args.cves:
            context["additional_keywords"] = [cve.strip() for cve in args.cves.split(",")]
            
        if args.title:
            context["title"] = args.title
        else:
            if "CVE-" in args.topic:
                context["title"] = f"Security Advisory: {args.topic} - Critical Vulnerability Alert"
            else:
                context["title"] = f"Security Advisory: {args.topic}"
    
    # Add audience to context
    context["audience"] = args.audience
    
    # Set up token tracker
    token_log = os.path.join(args.output_dir, "token_usage.jsonl") if args.verbose else None
    token_tracker = TokenTracker(log_path=token_log)
    
    # Create LLM provider
    provider_config = config.llm_provider
    if provider_config.provider_type.lower() == "openai":
        llm_client = OpenAIProvider(
            model_name=provider_config.model_name,
            api_key_env_var=provider_config.api_key_env_var,
            temperature=provider_config.temperature,
            max_tokens=provider_config.max_tokens,
            additional_params=provider_config.additional_params
        )
    elif provider_config.provider_type.lower() == "gemini":
        llm_client = GeminiProvider(
            model_name=provider_config.model_name,
            api_key_env_var=provider_config.api_key_env_var,
            temperature=provider_config.temperature,
            max_tokens=provider_config.max_tokens,
            additional_params=provider_config.additional_params
        )
    else:
        raise ValueError(f"Unsupported provider type: {provider_config.provider_type}")
    
    # Create prompt engine with specified template
    templates_dir = ".llmconfig/prompt-templates"  # Default location for templates
    prompt_engine = PromptEngine(templates_dir=templates_dir)
    
    # Create front matter generator
    front_matter_generator = FrontMatterGenerator(config.front_matter.schema_path)
    
    # Load plugins from the registry
    content_processors = []
    front_matter_enhancers = []
    
    # Get available plugins
    available_plugins = list_plugins()
    
    # Load content processor plugins
    if 'content_processor' in available_plugins:
        for plugin_name in available_plugins['content_processor']:
            plugin = get_plugin('content_processor', plugin_name)
            content_processors.append(plugin)
    
    # Load front matter enhancer plugins
    if 'front_matter_enhancer' in available_plugins:
        for plugin_name in available_plugins['front_matter_enhancer']:
            plugin = get_plugin('front_matter_enhancer', plugin_name)
            front_matter_enhancers.append(plugin)
    
    # Create markdown generator 
    markdown_generator = MarkdownGenerator(
        config=config,
        llm_provider=llm_client,
        prompt_engine=prompt_engine,
        front_matter_generator=front_matter_generator
    )
    
    # Add plugins
    for processor in content_processors:
        markdown_generator.register_content_processor(processor)
    
    for enhancer in front_matter_enhancers:
        markdown_generator.register_front_matter_enhancer(enhancer)
    
    # Prepare custom parameters for the generator
    custom_params = {
        # Include all elements from context
        **context,
        # Add front matter data
        "front_matter": {
            "title": context["title"],
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "description": f"Security report on {context['topic']}",
            "tags": ["security", "vulnerability", "CVE", "cybersecurity"],
            "author": "Security Operations Team",
            "severity": "Critical"
        }
    }
    
    # Create a temporary topic config to use with the generator
    # We'll use the existing TopicConfig class from the config module
    
    # Add the security topic to the config
    topic_name = args.report_type
    config.topics = {
        topic_name: TopicConfig(
            name=topic_name,
            prompt_template=f"{args.report_type}.j2",
            keywords=context.get("keywords", [])
        )
    }
    
    # For demo/testing purposes only - Add a dry run option to just display the prompt
    if args.verbose:
        print("\n----- PROMPT THAT WOULD BE SENT TO LLM -----")
        # Get the rendered prompt without calling the API
        rendered_prompt = prompt_engine.render_prompt(
            f"{args.report_type}.j2", 
            {k: v for k, v in custom_params.items() if k != 'front_matter'}
        )
        print(rendered_prompt)
        print("----- END OF PROMPT -----\n")
    
    # In a real run, we would generate the content
    if args.dry_run:
        # Create a mock response
        markdown_content = f"""---
title: "{custom_params['front_matter']['title']}"
date: "{custom_params['front_matter']['date']}"
description: "{custom_params['front_matter']['description']}"
tags: {custom_params['front_matter']['tags']}
author: "{custom_params['front_matter']['author']}"
severity: "{custom_params['front_matter']['severity']}"
---

# Security Advisory: {args.topic}

This is a mock response for the dry run. In a real run, this would be generated content from the LLM.
"""
    else:
        # Generate actual content (this might time out in the test environment)
        try:
            markdown_content = markdown_generator.generate_content(
                topic_name=topic_name,
                custom_params=custom_params
            )
        except Exception as e:
            # Fallback to mock content if generation fails
            print(f"Error generating content: {e}")
            print("Falling back to mock content for demonstration purposes")
            markdown_content = f"""---
title: "{custom_params['front_matter']['title']}"
date: "{custom_params['front_matter']['date']}"
description: "{custom_params['front_matter']['description']}"
tags: {custom_params['front_matter']['tags']}
author: "{custom_params['front_matter']['author']}"
severity: "{custom_params['front_matter']['severity']}"
---

# Security Advisory: {args.topic}

[Generated content would appear here in a real run]

This is a mock response since the LLM API call timed out.
"""
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create slugified filename from title
    slug = context["title"].lower().replace(" ", "-")
    for char in "?!:;,.()[]{}\"'":
        slug = slug.replace(char, "")
    filename = f"{slug}.md"
    
    # Write content to file
    output_path = output_dir / filename
    with open(output_path, "w") as f:
        f.write(markdown_content)
    
    # Print summary
    print(f"\nGenerated {args.report_type} for '{context['topic']}'")
    print(f"Output file: {output_path}")
    
    if args.verbose:
        print("\nToken Usage:")
        total_usage = token_tracker.get_total_usage()
        print(f"  Prompt tokens: {total_usage.prompt_tokens}")
        print(f"  Completion tokens: {total_usage.completion_tokens}")
        print(f"  Total tokens: {total_usage.total_tokens}")
        
        # Calculate cost if available
        if total_usage.cost is not None:
            print(f"  Estimated cost: ${total_usage.cost:.6f}")


if __name__ == "__main__":
    main()