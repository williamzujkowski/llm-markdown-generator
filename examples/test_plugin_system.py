#!/usr/bin/env python
"""Example script to test the plugin system.

This script demonstrates how to load and use plugins in the LLM Markdown Generator.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path if running as a script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_markdown_generator.config import Config, TopicConfig, load_config
from llm_markdown_generator.front_matter import FrontMatterGenerator
from llm_markdown_generator.generator import MarkdownGenerator
from llm_markdown_generator.llm_provider import LLMProvider
from llm_markdown_generator.plugins import plugin_hook, list_plugins


class MockPromptEngine:
    """Mock prompt engine for testing."""
    
    def __init__(self, templates_dir=None):
        """Initialize the mock prompt engine."""
        self.templates_dir = templates_dir
        
    def render_prompt(self, template_name, context):
        """Mock implementation of render_prompt.
        
        Args:
            template_name: Name of the template to render
            context: Context variables to use in rendering
            
        Returns:
            A simple static prompt
        """
        topic = context.get('topic', 'unknown topic')
        return f"Generate a blog post about {topic}"


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, *args, **kwargs):
        """Initialize the mock provider."""
        self.mock_response = """
# Sample Blog Post

This is a sample blog post generated by the LLM Markdown Generator.

## Features

- Easy to use
- Extensible plugin system
- Customizable output

## Code Example

```
def hello_world():
    print("Hello, World!")
```

## Conclusion

Thanks for using our generator!
"""

    def generate_text(self, prompt: str) -> str:
        """Generate text based on the given prompt.
        
        Args:
            prompt: The prompt to generate text from
            
        Returns:
            The generated text
        """
        return self.mock_response
        


def create_simple_config() -> Config:
    """Create a simple configuration for testing.
    
    Returns:
        A simple Config object
    """
    # Create a topic
    topic = TopicConfig(
        name="Sample Topic",
        keywords=["sample", "test", "plugin"],
        custom_data={},
        prompt_template="sample.j2"  # This won't be used with our mock
    )
    
    # Create a minimal config
    from llm_markdown_generator.config import LLMProviderConfig, FrontMatterConfig
    
    # Create dummy configs for required fields
    llm_provider_config = LLMProviderConfig(
        provider_type="mock",
        model_name="mock-model",
        api_key_env_var="MOCK_API_KEY"
    )
    
    front_matter_config = FrontMatterConfig(
        schema_path="dummy_path"
    )
    
    # Create a minimal config
    config = Config(
        llm_provider=llm_provider_config,
        front_matter=front_matter_config,
        topics={"sample": topic},
        output_dir="output"
    )
    
    return config


def main():
    """Run the example script."""
    parser = argparse.ArgumentParser(description="Test the plugin system")
    parser.add_argument("--plugins-dir", help="Directory containing custom plugins")
    parser.add_argument("--plugins", help="Comma-separated list of plugins to enable")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a simple configuration
    config = create_simple_config()
    config.output_dir = args.output_dir
    
    # Set up plugins directory if provided
    if args.plugins_dir:
        setattr(config, 'plugins_dir', args.plugins_dir)
    
    # Create a front matter generator with a simple schema
    schema = {
        "title": "{title}",
        "date": "{date}",
        "tags": ["{tags}"],
        "author": "Example User"
    }
    front_matter_generator = FrontMatterGenerator(schema)
    
    # Create a mock LLM provider
    llm_provider = MockLLMProvider()
    
    # Create a mock prompt engine
    prompt_engine = MockPromptEngine()
    
    # Create a markdown generator
    generator = MarkdownGenerator(
        config=config,
        llm_provider=llm_provider,
        prompt_engine=prompt_engine,
        front_matter_generator=front_matter_generator
    )
    
    # Load available plugins
    print("Loading plugins...")
    plugin_counts = generator.load_plugins()
    
    for category, count in plugin_counts.items():
        if count > 0:
            print(f"- Loaded {count} {category} plugins")
    
    # List all available plugins
    all_plugins = list_plugins()
    for category, plugins in all_plugins.items():
        print(f"\n{category.upper()} PLUGINS:")
        for plugin_name in plugins:
            print(f"- {plugin_name}")
    
    # Enable specific plugins if requested
    if args.plugins:
        from llm_markdown_generator.plugins import get_plugin, PluginError
        
        # Clear existing plugins
        generator.clear_plugins()
        
        # Enable specified plugins
        plugin_names = [p.strip() for p in args.plugins.split(',')]
        for plugin_name in plugin_names:
            found = False
                
            # Check if plugin exists in any category
            for category in all_plugins:
                if plugin_name in all_plugins[category]:
                    found = True
                    try:
                        plugin = get_plugin(category, plugin_name)
                        if category == "content_processor":
                            generator.register_content_processor(plugin)
                            print(f"Enabled content processor plugin: {plugin_name}")
                        else:
                            generator.register_front_matter_enhancer(plugin)
                            print(f"Enabled front matter enhancer plugin: {plugin_name}")
                    except PluginError as e:
                        print(f"Error enabling plugin '{plugin_name}': {str(e)}")
            
            if not found:
                print(f"Warning: Plugin '{plugin_name}' not found in any category")
    
    # Generate content with plugins
    print("\nGenerating content with plugins...")
    custom_params = {
        "title": "Testing Plugin System",
        "custom_source": "Example Script", 
        "author": "Test User",
        "license": "MIT",
        "license_url": "https://opensource.org/licenses/MIT"
    }
    
    if args.debug:
        print(f"Custom parameters: {custom_params}")
        
    content = generator.generate_content("sample", custom_params)
    
    # Write the content to a file
    output_path = generator.write_to_file(content)
    print(f"Content written to: {output_path}")
    
    # Display front matter data
    import yaml
    
    # Extract front matter from content
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            front_matter_str = parts[1]
            try:
                front_matter_data = yaml.safe_load(front_matter_str)
                print("\nExtracted front matter data:")
                
                # Check if we have front matter data
                if front_matter_data and isinstance(front_matter_data, dict):
                    # Print in sorted order for clarity
                    for key in sorted(front_matter_data.keys()):
                        value = front_matter_data[key]
                        print(f"  {key}: {value}")
                        
                    # Explicitly check for license fields
                    if "license" in front_matter_data:
                        print("\nLicense information found ✓")
                    else:
                        print("\nWARNING: License information missing ✗")
                else:
                    print("  Front matter is empty or invalid")
            except Exception as e:
                print(f"Error parsing front matter: {str(e)}")
    
    # Display the content
    print("\nGenerated content:")
    print("-" * 40)
    print(content)
    print("-" * 40)


if __name__ == "__main__":
    main()