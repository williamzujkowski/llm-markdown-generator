"""Integration tests for the plugin system.

These tests verify the integration of the plugin system with the rest of the framework,
including loading plugins from directories, applying plugins to generated content,
and interacting with other components.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest import mock

import pytest
import yaml

from llm_markdown_generator.config import Config, load_config, load_front_matter_schema
from llm_markdown_generator.front_matter import FrontMatterGenerator
from llm_markdown_generator.generator import MarkdownGenerator
from llm_markdown_generator.llm_provider import OpenAIProvider
from llm_markdown_generator.prompt_engine import PromptEngine
from llm_markdown_generator.plugins import plugin_hook, register_plugin, list_plugins, get_plugin


class MockLLMProvider:
    """Mock LLM provider for testing without API calls."""
    
    def __init__(self, mock_response=None):
        """Initialize the mock provider.
        
        Args:
            mock_response: The text to return when generate_text is called
        """
        self.mock_response = mock_response or "This is a mock LLM response."
        self.prompts = []
        self.total_usage = mock.MagicMock()
        self.total_usage.prompt_tokens = 10
        self.total_usage.completion_tokens = 20
        self.total_usage.total_tokens = 30
        self.total_usage.cost = 0.001
    
    def generate_text(self, prompt):
        """Mock implementation of generate_text.
        
        Args:
            prompt: The prompt to store
            
        Returns:
            str: The mock response
        """
        self.prompts.append(prompt)
        return self.mock_response
    
    def get_token_usage(self):
        """Mock implementation of get_token_usage."""
        return self.total_usage


@pytest.fixture
def temp_plugin_environment():
    """Create a temporary environment with plugin files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create plugin directory
        plugin_dir = Path(temp_dir) / "plugins"
        plugin_dir.mkdir()
        
        # Create a plugin file
        plugin_file = plugin_dir / "test_plugins.py"
        plugin_content = """
from llm_markdown_generator.plugins import plugin_hook

@plugin_hook("content_processor", "test_processor")
def add_footer(content, **kwargs):
    custom_footer = kwargs.get("footer_text", "Custom Footer")
    return content + f"\\n\\n---\\n{custom_footer}"

@plugin_hook("front_matter_enhancer", "test_enhancer")
def add_metadata(front_matter, **kwargs):
    enhanced = front_matter.copy()
    enhanced["plugin_enhanced"] = True
    enhanced["test_meta"] = kwargs.get("test_meta", "default")
    return enhanced

def register():
    # Required function that will be called when loading plugins
    pass
"""
        with open(plugin_file, "w") as f:
            f.write(plugin_content)
        
        # Create config directory
        config_dir = Path(temp_dir) / "config"
        config_dir.mkdir()
        
        # Create templates directory
        templates_dir = Path(temp_dir) / "templates"
        templates_dir.mkdir()
        
        # Create output directory
        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir()
        
        # Create a test config file with plugin_dir configured
        config_content = {
            "llm_provider": {
                "provider_type": "mock",
                "model_name": "mock-model",
                "api_key_env_var": "MOCK_API_KEY",
                "temperature": 0.7,
                "max_tokens": 2000,
                "additional_params": {}
            },
            "prompt": {
                "template": "test_template.j2",
                "include_front_matter": True
            },
            "front_matter": {
                "schema_path": str(config_dir / "front_matter_schema.yaml"),
                "date_format": "%Y-%m-%d"
            },
            "output_dir": str(output_dir),
            "filename_format": "{slug}.md",
            "plugins_dir": str(plugin_dir),  # Include the plugins directory in config
            "topics": {
                "Test Topic": {
                    "name": "Test Topic",
                    "description": "Test topic for plugin integration",
                    "keywords": ["test", "plugin", "integration"],
                    "tags": ["test", "plugin"],
                    "prompt_template": "test_template.j2"
                }
            }
        }
        
        with open(config_dir / "config.yaml", "w") as f:
            yaml.dump(config_content, f)
        
        # Create a test front matter schema
        front_matter_schema = {
            "layout": "post",
            "title": "{title}",
            "date": "{date}",
            "description": "{description}",
            "tags": ["{tags}"],
            "keywords": ["{keywords}"],
            "author": "Test Author"
        }
        
        with open(config_dir / "front_matter_schema.yaml", "w") as f:
            yaml.dump(front_matter_schema, f)
        
        # Create a test template
        template_content = """
# {{ title }}

{{ content }}
"""
        with open(templates_dir / "test_template.j2", "w") as f:
            f.write(template_content)
        
        # Return the paths for use in tests
        yield {
            "temp_dir": temp_dir,
            "plugin_dir": plugin_dir,
            "config_dir": config_dir, 
            "config_file": config_dir / "config.yaml",
            "templates_dir": templates_dir,
            "output_dir": output_dir
        }


def test_plugin_direct_registration():
    """Test direct registration of plugins with the generator."""
    # Define a test plugin function
    def test_function(value, **kwargs):
        return f"Processed: {value}"
    
    # Register it using the plugin API
    register_plugin("test_category", "test_plugin", test_function)
    
    # Create minimal configuration
    config = Config(
        llm_provider=mock.MagicMock(),
        front_matter=mock.MagicMock(),
        output_dir="/tmp",  # Just a placeholder
        topics={"test": mock.MagicMock()}
    )
    
    # Create the generator with mock components
    generator = MarkdownGenerator(
        config=config,
        llm_provider=MockLLMProvider(),
        prompt_engine=mock.MagicMock(),
        front_matter_generator=mock.MagicMock()
    )
    
    # Verify the plugin is available
    all_plugins = list_plugins()
    assert "test_category" in all_plugins
    assert "test_plugin" in all_plugins["test_category"]
    
    # Get and test the plugin function
    plugin_func = get_plugin("test_category", "test_plugin")
    assert plugin_func("test") == "Processed: test"


def test_direct_content_processor():
    """Test content processors directly registered to the generator."""
    # Define a test content processor
    def test_footer_processor(content: str, **kwargs) -> str:
        footer_text = kwargs.get("footer_text", "Default Footer")
        return content + f"\n\n---\n{footer_text}"
    
    # Register the processor manually
    register_plugin("content_processor", "test_footer", test_footer_processor)
    
    # Create minimal configuration
    config = Config(
        llm_provider=mock.MagicMock(),
        front_matter=mock.MagicMock(),
        output_dir="/tmp",  # Just a placeholder
        topics={"test": mock.MagicMock()}
    )
    
    # Setup config.topics for topic check
    config.topics = {"test_topic": mock.MagicMock()}
    config.topics["test_topic"].keywords = ["test"]
    config.topics["test_topic"].custom_data = {}
    
    # Create mock objects
    llm_provider = MockLLMProvider("Test content for plugin processing")
    prompt_engine = mock.MagicMock()
    front_matter_generator = mock.MagicMock()
    front_matter_generator.generate.return_value = "---\ntitle: Test\n---\n\n"
    
    # Create the generator
    generator = MarkdownGenerator(
        config=config,
        llm_provider=llm_provider,
        prompt_engine=prompt_engine,
        front_matter_generator=front_matter_generator
    )
    
    # Register the content processor
    generator.register_content_processor(get_plugin("content_processor", "test_footer"))
    
    # Generate content with custom parameters for plugins
    content = generator.generate_content("test_topic", {
        "footer_text": "This footer was added by a plugin"
    })
    
    # Verify the content was processed
    assert "Test content for plugin processing" in content
    assert "This footer was added by a plugin" in content


def test_selective_plugin_loading():
    """Test selectively loading and enabling specific plugins."""
    # Define test plugin functions
    def processor_a(content, **kwargs):
        return content + "\n\nProcessed by Plugin A"
        
    def processor_b(content, **kwargs):
        return content + "\n\nProcessed by Plugin B"
        
    def enhancer_a(front_matter, **kwargs):
        enhanced = front_matter.copy()
        enhanced["enhanced_by"] = "Plugin A"
        return enhanced
        
    def enhancer_b(front_matter, **kwargs):
        enhanced = front_matter.copy()
        enhanced["enhanced_by"] = "Plugin B"
        return enhanced
    
    # Register all plugins manually
    register_plugin("content_processor", "plugin_a", processor_a)
    register_plugin("content_processor", "plugin_b", processor_b)
    register_plugin("front_matter_enhancer", "enhancer_a", enhancer_a)
    register_plugin("front_matter_enhancer", "enhancer_b", enhancer_b)
    
    # Create minimal configuration
    config = Config(
        llm_provider=mock.MagicMock(),
        front_matter=mock.MagicMock(),
        output_dir="/tmp",  # Just a placeholder
        topics={"test": mock.MagicMock()}
    )
    
    # Setup config.topics for topic check
    config.topics = {"test_topic": mock.MagicMock()}
    config.topics["test_topic"].keywords = ["test"]
    config.topics["test_topic"].custom_data = {}
    
    # Create mock front matter generator
    front_matter_generator = mock.MagicMock()
    front_matter_data = {}
    
    def mock_generate(data):
        nonlocal front_matter_data
        front_matter_data = data
        content = "---\n"
        for key, value in data.items():
            content += f"{key}: {value}\n"
        content += "---\n\n"
        return content
        
    front_matter_generator.generate = mock_generate
    
    # Create the generator with mock components
    llm_provider = MockLLMProvider("Test content")
    
    generator = MarkdownGenerator(
        config=config,
        llm_provider=llm_provider,
        prompt_engine=mock.MagicMock(),
        front_matter_generator=front_matter_generator
    )
    
    # Clear any plugins that might have been registered
    generator.clear_plugins()
    
    # Selectively enable only plugin_a and enhancer_b
    plugin_a = get_plugin("content_processor", "plugin_a")
    enhancer_b = get_plugin("front_matter_enhancer", "enhancer_b")
    
    generator.register_content_processor(plugin_a)
    generator.register_front_matter_enhancer(enhancer_b)
    
    # Generate content
    content = generator.generate_content("test_topic", {})
    
    # Verify only plugin_a processed the content
    assert "Processed by Plugin A" in content
    assert "Processed by Plugin B" not in content
    
    # Verify only enhancer_b enhanced the front matter
    assert "enhanced_by: Plugin B" in content
    assert "enhanced_by: Plugin A" not in content
    
    # Verify the front matter data was updated correctly
    assert front_matter_data["enhanced_by"] == "Plugin B"


def test_plugin_error_handling():
    """Test that errors in plugins are properly handled."""
    # Define a plugin that will raise an error
    def error_plugin(content, **kwargs):
        # This plugin will raise an exception
        raise ValueError("Test error in plugin")
    
    # Define a good plugin
    def good_plugin(content, **kwargs):
        return content + "\n\nProcessed by good plugin"
    
    # Register the plugins
    register_plugin("content_processor", "error_plugin", error_plugin)
    register_plugin("content_processor", "good_plugin", good_plugin)
    
    # Create minimal configuration
    config = Config(
        llm_provider=mock.MagicMock(),
        front_matter=mock.MagicMock(),
        output_dir="/tmp",  # Just a placeholder
        topics={"test": mock.MagicMock()}
    )
    
    # Setup config.topics for topic check
    config.topics = {"test_topic": mock.MagicMock()}
    config.topics["test_topic"].keywords = ["test"]
    config.topics["test_topic"].custom_data = {}
    
    # Create the generator with mock components
    generator = MarkdownGenerator(
        config=config,
        llm_provider=MockLLMProvider("Test content"),
        prompt_engine=mock.MagicMock(),
        front_matter_generator=mock.MagicMock()
    )
    
    # Clear any existing plugins and register both plugins
    generator.clear_plugins()
    generator.register_content_processor(get_plugin("content_processor", "error_plugin"))
    generator.register_content_processor(get_plugin("content_processor", "good_plugin"))
    
    # Generate content - this should not raise an exception
    # despite one of the plugins having an error
    content = generator.generate_content("test_topic", {})
    
    # Check that the good plugin was still applied
    assert "Processed by good plugin" in content
    
    # Verify the error-causing plugin was skipped but didn't crash the process
    assert "Test content" in content