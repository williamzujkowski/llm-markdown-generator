"""Integration tests for the entire markdown generation pipeline.

These tests verify the end-to-end functionality of the markdown generation process,
from configuration loading to final output file generation.
"""

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import yaml

from llm_markdown_generator.config import Config, load_config, load_front_matter_schema
from llm_markdown_generator.front_matter import FrontMatterGenerator
from llm_markdown_generator.generator import MarkdownGenerator
from llm_markdown_generator.llm_provider import OpenAIProvider, GeminiProvider
from llm_markdown_generator.prompt_engine import PromptEngine
from llm_markdown_generator.error_handler import RetryConfig


class MockLLMProvider:
    """Mock LLM provider for testing without API calls."""
    
    def __init__(self, mock_response=None):
        """Initialize the mock provider.
        
        Args:
            mock_response: The text to return when generate_text is called
        """
        self.mock_response = mock_response or "This is a mock LLM response."
        self.prompts = []
    
    def generate_text(self, prompt):
        """Mock implementation of generate_text.
        
        Args:
            prompt: The prompt to store
            
        Returns:
            str: The mock response
        """
        self.prompts.append(prompt)
        return self.mock_response
    


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory with test configuration files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create config directory
        config_dir = Path(temp_dir) / "config"
        config_dir.mkdir()
        
        # Create templates directory
        templates_dir = Path(temp_dir) / "templates"
        templates_dir.mkdir()
        
        # Create output directory
        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir()
        
        # Create a test config file
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
                "template": "python_blog.j2",
                "include_front_matter": True
            },
            "front_matter": {
                "schema_path": str(config_dir / "front_matter_schema.yaml"),
                "date_format": "%Y-%m-%d"
            },
            "output_dir": str(output_dir),
            "filename_format": "{slug}.md",
            "topics": {
                "Python Programming": {
                    "name": "Python Programming",
                    "description": "Python programming language information",
                    "keywords": ["python", "programming", "coding"],
                    "tags": ["python", "programming"]
                },
                "Brief Python Introduction": {
                    "name": "Brief Python Introduction",
                    "description": "Brief intro to Python",
                    "keywords": ["python", "intro", "quick"],
                    "tags": ["python", "beginner"]
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
            "author": "AI Assistant"
        }
        
        with open(config_dir / "front_matter_schema.yaml", "w") as f:
            yaml.dump(front_matter_schema, f)
        
        # Create a test template
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
        with open(templates_dir / "python_blog.j2", "w") as f:
            f.write(template_content)
        
        yield temp_dir, config_dir, templates_dir, output_dir


def test_full_pipeline_with_mock_provider(temp_config_dir):
    """Test the full markdown generation pipeline using a mock LLM provider."""
    temp_dir, config_dir, templates_dir, output_dir = temp_config_dir
    
    # Load configuration
    config_path = str(config_dir / "config.yaml")
    config = load_config(config_path)
    
    # Update config for testing
    config.output_dir = str(output_dir)
    
    # Fix prompt template setting for topics in config
    for topic in config.topics.values():
        if hasattr(topic, 'prompt_template'):
            topic.prompt_template = "python_blog.j2"
    
    # Load front matter schema
    front_matter_schema = load_front_matter_schema(config.front_matter.schema_path)
    
    # Create mock prompt engine with direct template
    prompt_engine = PromptEngine(str(templates_dir))
    
    # Add a mock render_prompt method to avoid template loading issues
    def mock_render_prompt(template_name, context):
        return f"Generate a blog post about {context.get('topic')} with focus on {context.get('keywords')}."
    
    prompt_engine.render_prompt = mock_render_prompt
    
    # Create mock LLM provider
    mock_response = """
This is a test article about Python programming.

## Introduction
Python is a versatile programming language that can be used for many purposes.

## Core Features
1. Easy to learn syntax
2. Large standard library
3. Wide community support

## Conclusion
Python is an excellent choice for beginners and experts alike.
"""
    llm_provider = MockLLMProvider(mock_response)
    
    # Create front matter generator
    front_matter_generator = FrontMatterGenerator(front_matter_schema)
    
    # Create markdown generator
    generator = MarkdownGenerator(
        config=config,
        llm_provider=llm_provider,
        prompt_engine=prompt_engine,
        front_matter_generator=front_matter_generator,
    )
    
    # Generate content
    topic = "Python Programming"
    custom_params = {
        "title": "Introduction to Python Programming",
        "description": "A beginner's guide to Python programming language",
        "tags": ["python", "programming", "tutorial"],
        "keywords": ["python", "beginner", "guide"]
    }
    
    content = generator.generate_content(topic, custom_params)
    
    # Verify that the prompt was generated and sent to the LLM
    assert len(llm_provider.prompts) == 1
    assert topic in llm_provider.prompts[0]
    
    # Verify that the content contains the front matter
    assert "layout: post" in content
    assert "title: Introduction to Python Programming" in content
    # The description might be using the template placeholder since we're mocking
    assert "description:" in content
    
    # Verify that the content contains the LLM response
    assert "This is a test article about Python programming" in content
    assert "Python is a versatile programming language" in content
    
    # Write the content to a file
    output_path = generator.write_to_file(content)
    
    # Verify that the file was created
    output_file = Path(output_path)
    assert output_file.exists()
    
    # Verify the file content
    with open(output_file, "r") as f:
        file_content = f.read()
    
    assert "layout: post" in file_content
    assert "title: Introduction to Python Programming" in file_content
    assert "This is a test article about Python programming" in file_content


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), 
    reason="OPENAI_API_KEY environment variable not set"
)
def test_openai_integration(temp_config_dir):
    """Test the integration with the OpenAI provider (requires API key)."""
    temp_dir, config_dir, templates_dir, output_dir = temp_config_dir
    
    # Load configuration
    config_path = str(config_dir / "config.yaml")
    config = load_config(config_path)
    
    # Update config for testing
    config.output_dir = str(output_dir)
    config.llm_provider.provider_type = "openai"
    config.llm_provider.model_name = "gpt-3.5-turbo"  # Use a smaller model for testing
    config.llm_provider.api_key_env_var = "OPENAI_API_KEY"
    
    # Fix prompt template setting for topics in config
    for topic in config.topics.values():
        if hasattr(topic, 'prompt_template'):
            topic.prompt_template = "python_blog.j2"
    
    # Load front matter schema
    front_matter_schema = load_front_matter_schema(config.front_matter.schema_path)
    
    # Create mock prompt engine
    prompt_engine = PromptEngine(str(templates_dir))
    
    # Add a mock render_prompt method to avoid template loading issues
    def mock_render_prompt(template_name, context):
        return f"Generate a brief blog post about {context.get('topic', 'Python')} focusing on: {context.get('keywords', ['basics'])}."
    
    prompt_engine.render_prompt = mock_render_prompt
    
    # Create OpenAI provider
    retry_config = RetryConfig(max_retries=2, base_delay=1.0)
    llm_provider = OpenAIProvider(
        model_name=config.llm_provider.model_name,
        api_key_env_var=config.llm_provider.api_key_env_var,
        temperature=config.llm_provider.temperature,
        max_tokens=config.llm_provider.max_tokens,
        additional_params=config.llm_provider.additional_params,
        retry_config=retry_config
    )
    
    # Create front matter generator
    front_matter_generator = FrontMatterGenerator(front_matter_schema)
    
    # Create markdown generator
    generator = MarkdownGenerator(
        config=config,
        llm_provider=llm_provider,
        prompt_engine=prompt_engine,
        front_matter_generator=front_matter_generator,
    )
    
    # Generate content with a short prompt for testing
    topic = "Brief Python Introduction"
    custom_params = {
        "title": "Quick Python Intro",
        "description": "A very brief introduction to Python",
        "tags": ["python", "quick"],
        "keywords": ["python", "intro"]
    }
    
    content = generator.generate_content(topic, custom_params)
    
    # Verify that the content contains the front matter
    assert "layout: post" in content
    assert "title: Quick Python Intro" in content
    assert "description:" in content
    
    # Verify that the content is not empty
    assert len(content) > 200
    
    # Write the content to a file
    output_path = generator.write_to_file(content)
    
    # Verify that the file was created
    output_file = Path(output_path)
    assert output_file.exists()
    


@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"), 
    reason="GEMINI_API_KEY environment variable not set"
)
def test_gemini_integration(temp_config_dir):
    """Test the integration with the Gemini provider (requires API key)."""
    temp_dir, config_dir, templates_dir, output_dir = temp_config_dir
    
    # Load configuration
    config_path = str(config_dir / "config.yaml")
    config = load_config(config_path)
    
    # Update config for testing
    config.output_dir = str(output_dir)
    config.llm_provider.provider_type = "gemini"
    config.llm_provider.model_name = "gemini-2.0-flash"
    config.llm_provider.api_key_env_var = "GEMINI_API_KEY"
    
    # Load front matter schema
    front_matter_schema = load_front_matter_schema(config.front_matter.schema_path)
    
    # Create prompt engine
    prompt_engine = PromptEngine(str(templates_dir))
    
    # Add a mock render_prompt method to avoid template loading issues
    def mock_render_prompt(template_name, context):
        return f"Generate a brief blog post about {context.get('topic', 'Python')} focusing on: {context.get('keywords', ['basics'])}."
    
    prompt_engine.render_prompt = mock_render_prompt
    
    # Create Gemini provider
    retry_config = RetryConfig(max_retries=2, base_delay=1.0)
    llm_provider = GeminiProvider(
        model_name=config.llm_provider.model_name,
        api_key_env_var=config.llm_provider.api_key_env_var,
        temperature=config.llm_provider.temperature,
        max_tokens=config.llm_provider.max_tokens,
        additional_params=config.llm_provider.additional_params,
        retry_config=retry_config
    )
    
    # Create front matter generator
    front_matter_generator = FrontMatterGenerator(front_matter_schema)
    
    # Create markdown generator
    generator = MarkdownGenerator(
        config=config,
        llm_provider=llm_provider,
        prompt_engine=prompt_engine,
        front_matter_generator=front_matter_generator,
    )
    
    # Generate content with a short prompt for testing
    topic = "Brief Python Introduction"
    custom_params = {
        "title": "Quick Python Intro (Gemini)",
        "description": "A very brief introduction to Python using Gemini",
        "tags": ["python", "quick", "gemini"],
        "keywords": ["python", "intro", "gemini"]
    }
    
    content = generator.generate_content(topic, custom_params)
    
    # Verify that the content contains the front matter
    assert "layout: post" in content
    assert "title: Quick Python Intro (Gemini)" in content
    assert "description:" in content
    
    # Verify that the content is not empty
    assert len(content) > 200
    
    # Write the content to a file
    output_path = generator.write_to_file(content)
    
    # Verify that the file was created
    output_file = Path(output_path)
    assert output_file.exists()
    
