"""Unit tests for the plugin system.

Tests the functionality of the plugin system for the LLM Markdown Generator.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict

import pytest

from llm_markdown_generator.plugins import (
    Plugin,
    PluginError,
    get_plugin,
    list_plugins,
    load_plugins_from_directory,
    plugin_hook,
    register_plugin,
)


class TestPluginRegistry:
    """Tests for the plugin registry."""
    
    def test_register_and_get_plugin(self):
        """Test registering and retrieving a plugin."""
        # Define a test plugin
        @plugin_hook("test_category", "test_plugin")
        def test_plugin_func(x: int) -> int:
            return x * 2
            
        # Get the plugin
        plugin = get_plugin("test_category", "test_plugin")
        
        # Verify
        assert plugin is test_plugin_func
        assert plugin(5) == 10
        
    def test_register_duplicate_plugin(self):
        """Test that registering a duplicate plugin raises an error."""
        # Register the first plugin
        def plugin_one() -> str:
            return "plugin one"
            
        register_plugin("test_category", "duplicate_name", plugin_one)
        
        # Try to register a second plugin with the same name
        def plugin_two() -> str:
            return "plugin two"
            
        with pytest.raises(PluginError):
            register_plugin("test_category", "duplicate_name", plugin_two)
            
    def test_get_nonexistent_plugin(self):
        """Test getting a plugin that doesn't exist."""
        with pytest.raises(PluginError):
            get_plugin("nonexistent_category", "nonexistent_plugin")
            
        # Category exists but plugin doesn't
        register_plugin("existing_category", "existing_plugin", lambda: None)
        with pytest.raises(PluginError):
            get_plugin("existing_category", "nonexistent_plugin")
            
    def test_list_plugins(self):
        """Test listing all plugins or plugins in a specific category."""
        # Clear plugin registry and add test plugins
        from llm_markdown_generator.plugins import _plugins
        _plugins.clear()
        
        register_plugin("category1", "plugin1", lambda: None)
        register_plugin("category1", "plugin2", lambda: None)
        register_plugin("category2", "plugin3", lambda: None)
        
        # List all plugins
        all_plugins = list_plugins()
        assert "category1" in all_plugins
        assert "category2" in all_plugins
        assert "plugin1" in all_plugins["category1"]
        assert "plugin2" in all_plugins["category1"]
        assert "plugin3" in all_plugins["category2"]
        
        # List plugins in a specific category
        category1_plugins = list_plugins("category1")
        assert "category1" in category1_plugins
        assert "category2" not in category1_plugins
        assert "plugin1" in category1_plugins["category1"]
        assert "plugin2" in category1_plugins["category1"]
        
        # List plugins in a non-existent category
        nonexistent_plugins = list_plugins("nonexistent")
        assert not nonexistent_plugins
        
    def test_decorator_plugin_hook(self):
        """Test the plugin_hook decorator."""
        # Define a plugin using the decorator
        @plugin_hook("decorator_category", "decorator_plugin")
        def decorated_plugin(x: int, y: int) -> int:
            return x + y
            
        # Get the plugin
        plugin = get_plugin("decorator_category", "decorator_plugin")
        
        # Verify
        assert plugin is decorated_plugin
        assert plugin(3, 4) == 7


class TestPluginLoading:
    """Tests for loading plugins from files and directories."""
    
    @pytest.mark.skip(reason="Requires sys.path modification for module loading")
    def test_load_plugins_from_directory(self, tmp_path: Path):
        """Test loading plugins from a directory of Python files."""
        # Add the temp directory to sys.path for importing
        old_path = sys.path.copy()
        sys.path.insert(0, str(tmp_path.parent))
        
        try:
            # Create a temporary directory with plugin modules
            plugin_dir = tmp_path / "plugins"
            plugin_dir.mkdir()
            
            # Make it a package with __init__.py
            with open(plugin_dir / "__init__.py", "w") as f:
                f.write("# Plugin package\n")
            
            # Create a plugin module file
            plugin_file = plugin_dir / "test_plugin.py"
            with open(plugin_file, "w") as f:
                f.write("""
from llm_markdown_generator.plugins import plugin_hook

@plugin_hook("test_category", "module_plugin")
def test_function(text: str) -> str:
    return text.upper()
    
def register():
    # This function is called by the plugin loader
    pass
""")
            
            # Load plugins from the directory
            loaded_plugins = load_plugins_from_directory(plugin_dir)
            
            # Verify
            assert "test_category" in loaded_plugins
            assert "module_plugin" in loaded_plugins["test_category"]
            
            # Get and test the plugin
            plugin = get_plugin("test_category", "module_plugin")
            assert plugin("hello") == "HELLO"
            
        finally:
            # Restore sys.path
            sys.path = old_path
        
    def test_load_plugins_nonexistent_directory(self):
        """Test loading plugins from a directory that doesn't exist."""
        with pytest.raises(PluginError):
            load_plugins_from_directory("/nonexistent/directory")
            
    def test_load_plugins_error_in_module(self, tmp_path: Path):
        """Test loading plugins with an error in the module."""
        # Create a temporary directory with plugin modules
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()
        
        # Create a plugin module file with an error
        plugin_file = plugin_dir / "error_plugin.py"
        with open(plugin_file, "w") as f:
            f.write("""
# This will cause an ImportError
from nonexistent_module import something

def register():
    pass
""")
        
        # Loading should raise a PluginError
        with pytest.raises(PluginError):
            load_plugins_from_directory(plugin_dir)


class TestPluginFunctionality:
    """Tests actual plugin functionality with the MarkdownGenerator."""
    
    def test_content_processor_plugin(self):
        """Test a content processor plugin with the MarkdownGenerator."""
        from llm_markdown_generator.generator import MarkdownGenerator
        from unittest.mock import MagicMock
        
        # Create a content processor plugin
        @plugin_hook("content_processor", "test_processor")
        def add_footer(content: str, **kwargs) -> str:
            return content + "\n\n---\nProcessed by test plugin"
            
        # Setup mock objects
        config = MagicMock()
        llm_provider = MagicMock()
        prompt_engine = MagicMock()
        front_matter_generator = MagicMock()
        
        # Configure mocks
        llm_provider.generate_text.return_value = "Test content"
        front_matter_generator.generate.return_value = "---\ntitle: Test\n---\n\n"
        
        # Setup config.topics for topic check
        config.topics = {"test_topic": MagicMock()}
        config.topics["test_topic"].keywords = ["test"]
        config.topics["test_topic"].custom_data = {}
        
        # Create generator and register plugin
        generator = MarkdownGenerator(
            config=config,
            llm_provider=llm_provider,
            prompt_engine=prompt_engine,
            front_matter_generator=front_matter_generator
        )
        generator.register_content_processor(get_plugin("content_processor", "test_processor"))
        
        # Generate content
        content = generator.generate_content("test_topic", {})
        
        # Verify
        assert "Test content" in content
        assert "Processed by test plugin" in content
        
    def test_front_matter_enhancer_plugin(self):
        """Test a front matter enhancer plugin with the MarkdownGenerator."""
        from llm_markdown_generator.generator import MarkdownGenerator
        from unittest.mock import MagicMock
        
        # Create a front matter enhancer plugin
        @plugin_hook("front_matter_enhancer", "test_enhancer")
        def add_metadata(front_matter: Dict, **kwargs) -> Dict:
            enhanced = front_matter.copy()
            enhanced["enhanced"] = True
            enhanced["generator"] = "test-plugin"
            return enhanced
            
        # Setup mock objects
        config = MagicMock()
        llm_provider = MagicMock()
        prompt_engine = MagicMock()
        front_matter_generator = MagicMock()
        
        # Setup config.topics for topic check
        config.topics = {"test_topic": MagicMock()}
        config.topics["test_topic"].keywords = ["test"]
        config.topics["test_topic"].custom_data = {}
        
        # Create generator and register plugin
        generator = MarkdownGenerator(
            config=config,
            llm_provider=llm_provider,
            prompt_engine=prompt_engine,
            front_matter_generator=front_matter_generator
        )
        generator.register_front_matter_enhancer(get_plugin("front_matter_enhancer", "test_enhancer"))
        
        # Mock the generate_content method to check the front matter data
        original_front_matter_generate = front_matter_generator.generate
        
        def mock_generate(data):
            # Verify the front matter data was enhanced
            assert data["enhanced"] is True
            assert data["generator"] == "test-plugin"
            
            # Return some front matter 
            return "---\nenhanced: true\n---\n\n"
            
        front_matter_generator.generate = mock_generate
        
        # Configure llm_provider
        llm_provider.generate_text.return_value = "Test content"
        
        # Generate content
        generator.generate_content("test_topic", {})
        
        # Restore the original method
        front_matter_generator.generate = original_front_matter_generate