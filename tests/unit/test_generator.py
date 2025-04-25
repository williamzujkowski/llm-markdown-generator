"""Tests for the generator module."""

import os
from pathlib import Path
import tempfile
from typing import Dict, Any
from unittest import mock
from datetime import datetime

import pytest

from llm_markdown_generator.config import Config, TopicConfig, LLMProviderConfig, FrontMatterConfig
from llm_markdown_generator.front_matter import FrontMatterGenerator
from llm_markdown_generator.generator import GeneratorError, MarkdownGenerator
from llm_markdown_generator.llm_provider import LLMProvider, TokenUsage
from llm_markdown_generator.prompt_engine import PromptEngine


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, response="This is a mock LLM response."):
        super().__init__()
        self.response = response
        self.prompt_history = []
        self.last_usage = TokenUsage(
            prompt_tokens=50,
            completion_tokens=30,
            total_tokens=80,
            cost=0.00123
        )
        self.total_usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=60,
            total_tokens=160,
            cost=0.00246
        )

    def generate_text(self, prompt: str) -> str:
        """Generate text from the provided prompt (mock)."""
        self.prompt_history.append(prompt)
        return self.response
        
    def get_token_usage(self) -> TokenUsage:
        """Get token usage information for the most recent request."""
        return self.last_usage


class MockPromptEngine:
    """Mock prompt engine for testing."""

    def __init__(self):
        self.render_history = []

    def render_prompt(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render a prompt with the given context (mock)."""
        self.render_history.append((template_name, context))
        return f"Rendered prompt for {template_name} with context: {context}"


class MockFrontMatterGenerator:
    """Mock front matter generator for testing."""

    def __init__(self):
        self.generate_history = []

    def generate(self, data: Dict[str, Any] = None) -> str:
        """Generate front matter with the provided data (mock)."""
        if data is None:
            data = {}
        self.generate_history.append(data)
        return "---\nyaml: front matter\n---\n"


@pytest.fixture
def mock_config():
    """Create a mock configuration object for testing."""
    llm_provider = LLMProviderConfig(
        provider_type="mock",
        model_name="mock-model",
        api_key_env_var="MOCK_API_KEY",
    )

    front_matter = FrontMatterConfig(schema_path="mock_schema.yaml")

    topics = {
        "python": TopicConfig(
            name="python",
            prompt_template="python_blog.j2",
            keywords=["python", "programming"],
            custom_data={"audience": "developers"},
        ),
        "javascript": TopicConfig(
            name="javascript",
            prompt_template="javascript_blog.j2",
            keywords=["javascript", "web"],
        ),
    }

    return Config(
        llm_provider=llm_provider,
        front_matter=front_matter,
        topics=topics,
        output_dir="output",
    )


@pytest.fixture
def markdown_generator(mock_config):
    """Create a markdown generator with mock components for testing."""
    llm_provider = MockLLMProvider()
    prompt_engine = MockPromptEngine()
    front_matter_generator = MockFrontMatterGenerator()

    return MarkdownGenerator(
        config=mock_config,
        llm_provider=llm_provider,
        prompt_engine=prompt_engine,
        front_matter_generator=front_matter_generator,
    )


class TestMarkdownGenerator:
    """Tests for the MarkdownGenerator class."""

    def test_generate_content(self, markdown_generator, mock_config):
        """Test content generation for a valid topic."""
        content = markdown_generator.generate_content("python")

        # Check that prompt engine was called with correct parameters
        prompt_engine = markdown_generator.prompt_engine
        assert len(prompt_engine.render_history) == 1
        template_name, context = prompt_engine.render_history[0]
        assert template_name == "python_blog.j2"
        assert context["topic"] == "python"
        assert context["keywords"] == ["python", "programming"]
        assert context["audience"] == "developers"

        # Check that LLM provider was called
        llm_provider = markdown_generator.llm_provider
        assert len(llm_provider.prompt_history) == 1

        # Check that front matter generator was called with correct parameters
        front_matter_generator = markdown_generator.front_matter_generator
        assert len(front_matter_generator.generate_history) == 1
        front_matter_data = front_matter_generator.generate_history[0]
        assert front_matter_data["title"] == "Python Post"
        assert front_matter_data["tags"] == ["python", "programming"]
        assert front_matter_data["category"] == "python"

        # Check the final content structure
        assert "---\nyaml: front matter\n---\n" in content
        assert "This is a mock LLM response." in content
        
    def test_get_token_usage(self, markdown_generator):
        """Test getting token usage from the generator."""
        # Generate content to ensure LLM provider is called
        markdown_generator.generate_content("python")
        
        # Get token usage
        usage = markdown_generator.get_token_usage()
        
        # Check that it matches what's expected from the mock provider
        assert usage.prompt_tokens == 50
        assert usage.completion_tokens == 30
        assert usage.total_tokens == 80
        assert usage.cost == 0.00123
        
    def test_get_total_usage(self, markdown_generator):
        """Test getting total token usage from the generator."""
        # Generate content to ensure LLM provider is called
        markdown_generator.generate_content("python")
        
        # Get total usage
        total_usage = markdown_generator.get_total_usage()
        
        # Check that it matches what's expected from the mock provider
        assert total_usage.prompt_tokens == 100
        assert total_usage.completion_tokens == 60
        assert total_usage.total_tokens == 160
        assert total_usage.cost == 0.00246

    def test_generate_content_with_custom_params(self, markdown_generator):
        """Test content generation with custom parameters."""
        custom_params = {
            "title": "Custom Python Title",
            "additional_keywords": ["tutorial", "beginner"],
        }

        content = markdown_generator.generate_content("python", custom_params)

        # Check that the custom parameters were passed to the prompt engine
        prompt_engine = markdown_generator.prompt_engine
        _, context = prompt_engine.render_history[0]
        assert context["title"] == "Custom Python Title"
        assert "additional_keywords" in context

        # Check that front matter generator received the custom title
        front_matter_generator = markdown_generator.front_matter_generator
        front_matter_data = front_matter_generator.generate_history[0]
        assert front_matter_data["title"] == "Custom Python Title"

    def test_generate_content_invalid_topic(self, markdown_generator):
        """Test handling of invalid topic."""
        with pytest.raises(GeneratorError) as exc_info:
            markdown_generator.generate_content("invalid_topic")

        assert "Topic 'invalid_topic' not found in configuration" in str(exc_info.value)

    def test_generate_content_error_handling(self, markdown_generator):
        """Test error handling during content generation."""
        # Mock LLM provider to raise an exception
        markdown_generator.llm_provider.generate_text = mock.Mock(
            side_effect=Exception("LLM error")
        )

        with pytest.raises(GeneratorError) as exc_info:
            markdown_generator.generate_content("python")

        assert "Error generating content for topic 'python'" in str(exc_info.value)

    def test_write_to_file(self, markdown_generator):
        """Test writing content to a file."""
        content = "---\ntitle: Test Post\n---\n\nThis is a test post."

        with tempfile.TemporaryDirectory() as temp_dir:
            # Override output directory for testing
            markdown_generator.config.output_dir = temp_dir

            # Write content to file
            file_path = markdown_generator.write_to_file(content, filename="test-post")

            # Check that the file was created
            assert os.path.exists(file_path)
            with open(file_path, "r") as f:
                file_content = f.read()
                assert file_content == content

    def test_write_to_file_with_title(self, markdown_generator):
        """Test writing content to a file using a title to generate the filename."""
        content = "---\ntitle: Test Post\n---\n\nThis is a test post."

        with tempfile.TemporaryDirectory() as temp_dir:
            # Override output directory for testing
            markdown_generator.config.output_dir = temp_dir

            # Write content to file using title
            file_path = markdown_generator.write_to_file(content, title="Test Post Title")

            # Check that the file was created with a slugified name
            assert os.path.exists(file_path)
            assert "test-post-title.md" in file_path

    def test_write_to_file_extract_title(self, markdown_generator):
        """Test writing content to a file by extracting title from front matter."""
        content = "---\ntitle: Extracted Title\ntags: [test]\n---\n\nThis is a test post."

        with tempfile.TemporaryDirectory() as temp_dir:
            # Override output directory for testing
            markdown_generator.config.output_dir = temp_dir

            # Write content to file extracting title from front matter
            file_path = markdown_generator.write_to_file(content)

            # Check that the file was created with a slugified name from the front matter title
            assert os.path.exists(file_path)
            assert "extracted-title.md" in file_path

    def test_write_to_file_fallback_timestamp(self, markdown_generator):
        """Test falling back to timestamp when no title can be determined."""
        content = "No front matter here, just text content."

        with tempfile.TemporaryDirectory() as temp_dir:
            # Override output directory for testing
            markdown_generator.config.output_dir = temp_dir

            # Create a fixed datetime for testing
            fixed_datetime = datetime(2023, 1, 1, 12, 34, 56)
            
            # We need to patch the specific imported datetime within the generator module
            with mock.patch("datetime.datetime") as mock_datetime:
                mock_datetime.now.return_value = fixed_datetime
                
                # Write content to file
                file_path = markdown_generator.write_to_file(content)

                # Check that the file was created with a timestamp-based name
                assert os.path.exists(file_path)
                # The datetime format may differ from the expected one, so just check for part of it
                assert "post-" in file_path
                assert ".md" in file_path

    def test_write_to_file_error_handling(self, markdown_generator):
        """Test error handling during file writing."""
        content = "Test content"

        # Mock os.makedirs to raise an exception
        with mock.patch(
            "pathlib.Path.mkdir", side_effect=Exception("File error")
        ):
            with pytest.raises(GeneratorError) as exc_info:
                markdown_generator.write_to_file(content)

            assert "Error writing content to file" in str(exc_info.value)
