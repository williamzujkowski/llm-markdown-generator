"""
Example script demonstrating how to generate specialized content using different prompt templates.

This example shows how to use the different prompt templates available in the framework
to generate various types of content beyond standard blog posts.
"""

import os
import argparse
from pathlib import Path

# We'll use the core components directly instead of the CLI module
from llm_markdown_generator.config import Config
from llm_markdown_generator.llm_provider import create_llm_provider
from llm_markdown_generator.prompt_engine import PromptEngine
from llm_markdown_generator.front_matter import FrontMatterGenerator
from llm_markdown_generator.generator import MarkdownGenerator
from llm_markdown_generator.token_tracker import TokenTracker
from llm_markdown_generator.plugins import load_plugins


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate specialized content with LLMs")
    parser.add_argument("template", type=str, help="Template name (e.g., technical_tutorial, product_review)")
    parser.add_argument("topic", type=str, help="The topic to write about")
    parser.add_argument("--title", type=str, help="Title for the content")
    parser.add_argument("--audience", type=str, help="Target audience")
    parser.add_argument("--tone", type=str, help="Tone of the content")
    parser.add_argument("--keywords", type=str, help="Comma-separated keywords")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--config-path", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    # Load configuration
    config = Config(args.config_path)
    
    # Prepare context for template rendering
    context = {
        "topic": args.topic,
        "title": args.title if args.title else f"{args.topic.title()} - {args.template.replace('_', ' ').title()}"
    }
    
    # Add optional parameters to context if provided
    if args.audience:
        context["audience"] = args.audience
    if args.tone:
        context["tone"] = args.tone
    if args.keywords:
        context["keywords"] = [kw.strip() for kw in args.keywords.split(",")]
    
    # Set up token tracker
    token_tracker = TokenTracker()
    
    # Create LLM provider
    llm_client = create_llm_provider(
        provider_type=config.provider_type,
        model_name=config.model_name,
        api_key=os.getenv(f"{config.provider_type.upper()}_API_KEY"),
        token_tracker=token_tracker
    )
    
    # Create prompt engine with specified template
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
    markdown_content = markdown_generator.generate(
        template_name=f"{args.template}.j2",
        context=context,
        front_matter_data={
            "title": context["title"],
            "topic": args.topic,
            "template_type": args.template
        }
    )
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create slugified filename from title
    slug = context["title"].lower().replace(" ", "-").replace("/", "-")
    for char in "?!:;,.()[]{}\"'":
        slug = slug.replace(char, "")
    filename = f"{slug}.md"
    
    # Write content to file
    output_path = output_dir / filename
    with open(output_path, "w") as f:
        f.write(markdown_content)
    
    # Print summary
    print(f"\nGenerated {args.template} content for '{args.topic}'")
    print(f"Output file: {output_path}")
    
    if args.verbose:
        print("\nToken Usage:")
        print(f"  Prompt tokens: {token_tracker.prompt_tokens}")
        print(f"  Completion tokens: {token_tracker.completion_tokens}")
        print(f"  Total tokens: {token_tracker.total_tokens}")
        
        # Calculate cost if available
        if token_tracker.estimated_cost is not None:
            print(f"  Estimated cost: ${token_tracker.estimated_cost:.6f}")


if __name__ == "__main__":
    main()