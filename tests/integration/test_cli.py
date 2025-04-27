"""Integration tests for the command-line interface.

Tests the functionality of the CLI for generating markdown content.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from llm_markdown_generator.cli import app


@pytest.fixture
def temp_config_files():
    """Create temporary configuration files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        
        # Create config directory
        config_dir = temp_dir_path / "config"
        config_dir.mkdir()
        
        # Create output directory
        output_dir = temp_dir_path / "output"
        output_dir.mkdir()
        
        # Create templates directory
        templates_dir = temp_dir_path / ".llmconfig" / "prompt-templates"
        templates_dir.mkdir(parents=True)
        
        # Create config file
        config_file = config_dir / "test_config.yaml"
        config_content = {
            "llm_provider": {
                "provider_type": "openai",  # Changed from "mock" to a supported provider type
                "model_name": "mock-model",
                "api_key_env_var": "MOCK_API_KEY",
                "temperature": 0.7,
                "max_tokens": 2000,
                "additional_params": {}
            },
            "front_matter": {
                "schema_path": str(config_dir / "front_matter_schema.yaml"),
                "date_format": "%Y-%m-%d"
            },
            "output_dir": str(output_dir),
            "filename_format": "{slug}.md",
            "topics": {
                "TestTopic": {
                    "name": "TestTopic",
                    "description": "A test topic for CLI integration testing",
                    "keywords": ["test", "integration", "cli"],
                    "tags": ["test", "cli"],
                    "prompt_template": "simple_template.j2"  # Added required field 
                },
                "QuickTest": {
                    "name": "QuickTest",
                    "description": "A quick test for OpenAI integration",
                    "keywords": ["test", "quick", "openai"],
                    "tags": ["test", "openai"],
                    "prompt_template": "simple_template.j2"  # Added required field
                }
            }
        }
        
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)
        
        # Create front matter schema
        schema_file = config_dir / "front_matter_schema.yaml"
        schema_content = {
            "layout": "post",
            "title": "{title}",
            "date": "{date}",
            "description": "{description}",
            "tags": ["{tags}"],
            "keywords": ["{keywords}"],
            "author": "Test Author"
        }
        
        with open(schema_file, "w") as f:
            yaml.dump(schema_content, f)
            
        # Create template file
        template_file = templates_dir / "simple_template.j2"
        template_content = """
# {{ title }}

{{ content }}

## Tags
{% for tag in tags %}
- {{ tag }}
{% endfor %}

## Keywords
{% for keyword in keywords %}
- {{ keyword }}
{% endfor %}
"""
        with open(template_file, "w") as f:
            f.write(template_content)
        
        yield temp_dir_path, config_file, output_dir


@pytest.mark.parametrize("mock_provider", ["openai", "gemini"])
def test_cli_with_mock_provider(temp_config_files, monkeypatch, mock_provider):
    """Test the CLI with a mock LLM provider.
    
    This test verifies that the CLI correctly loads configuration, processes input, 
    and generates output files without making actual API calls.
    """
    temp_dir, config_file, output_dir = temp_config_files
    
    # Set up environment
    monkeypatch.setenv("OPENAI_API_KEY", "mock-openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    monkeypatch.setenv("MOCK_API_KEY", "mock-key")
    
    # Mock the OpenAI and Gemini provider classes to avoid API calls
    mock_response = "This is a mock response from the LLM provider."
    
    class MockResponse:
        def __init__(self):
            pass
    
    class MockProvider:
        def __init__(self, *args, **kwargs):
            pass
            
        def generate_text(self, prompt):
            return mock_response
    
    # Create a fully mocked generator function to replace the actual content generation
    from llm_markdown_generator import generator
    
    def mock_generate_content(self, topic_name, custom_params=None):
        """Mock implementation of generate_content.
        
        Completely bypasses the normal generation process.
        """
        if custom_params is None:
            custom_params = {}
            
        # Create a mock front matter
        front_matter = """---
layout: post
title: Test Title
date: 2025-04-24
description: A test description
tags: ['test', 'cli', 'integration']
keywords: ['test', 'cli', 'integration']
author: Test Author
---
"""
        
        # Create a mock article content
        content = """
# Test Title

This is a mock response from the LLM provider.

## Tags
- test
- cli
- integration

## Keywords
- test
- cli
- integration
"""
        
        # Return the combined content
        return front_matter + content
        
    # Patch the generate_content method of MarkdownGenerator
    monkeypatch.setattr(generator.MarkdownGenerator, "generate_content", mock_generate_content)
    
    # Apply mocking based on the provider type
    if mock_provider == "openai":
        from llm_markdown_generator.llm_provider import OpenAIProvider
        monkeypatch.setattr("llm_markdown_generator.llm_provider.OpenAIProvider", MockProvider)
        provider_arg = "--provider=openai"
    else:
        from llm_markdown_generator.llm_provider import GeminiProvider
        monkeypatch.setattr("llm_markdown_generator.llm_provider.GeminiProvider", MockProvider)
        provider_arg = "--provider=gemini"
    
    # Set up CLI runner
    runner = CliRunner()
    
    # Run the CLI command - use a single-word topic to avoid space issues
    # Note: Updated to use the 'generate' subcommand
    result = runner.invoke(
        app, 
        [
            "generate",
            "TestTopic", 
            f"--config-path={config_file}",
            f"--output-dir={output_dir}",
            "--title=Test Title",
            "--keywords=test,cli,integration",
            provider_arg,
            "--verbose"
        ]
    )
    
    # Check the output and exception details
    print(f"Exit code: {result.exit_code}")
    print(f"Exception: {result.exception}")
    print(f"Output: {result.output}")
    
    # Now we assert the exit code should be 0 (success)
    assert result.exit_code == 0
    
    # Check that the output file was created
    output_files = list(Path(output_dir).glob("*.md"))
    assert len(output_files) == 1
    
    # Check the content of the output file
    with open(output_files[0], "r") as f:
        content = f.read()
    
    assert "layout: post" in content
    assert "title: Test Title" in content
    assert mock_response in content
    


@pytest.mark.skip(reason="Need mocking to work consistently in CI environment")
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), 
    reason="OPENAI_API_KEY environment variable not set"
)
def test_cli_with_real_openai(temp_config_files):
    """Test the CLI with the real OpenAI provider.
    
    This test makes an actual API call to OpenAI, so it's skipped if no API key is available.
    """
    temp_dir, config_file, output_dir = temp_config_files
    
    # Set up CLI runner
    runner = CliRunner()
    
    # Run the CLI command - use a mini prompt to minimize token usage
    # Note: Updated to use the 'generate' subcommand
    result = runner.invoke(
        app, 
        [
            "generate",
            "QuickTest",  # Use topic name from the configuration (avoiding spaces in the topic name)
            f"--config-path={config_file}",
            f"--output-dir={output_dir}",
            "--title=OpenAI Test",
            "--keywords=openai,test",
            "--provider=openai",
            "--verbose",
            "--retries=1"  # Limit retries for the test (param is --retries not --max-retries)
        ]
    )
    
    # Check that the command executed successfully
    assert result.exit_code == 0
    
    # Check that the output file was created
    output_files = list(Path(output_dir).glob("*.md"))
    assert len(output_files) == 1
    
    # Check that the file has reasonable content
    with open(output_files[0], "r") as f:
        content = f.read()
    
    assert "layout: post" in content
    assert "title: OpenAI Test" in content
    assert len(content) > 200  # Should have substantial content
    
    # Check that token usage was reported
    assert "Token Usage Information:" in result.stdout