"""Tests for the config module."""

import os
from pathlib import Path
import tempfile
from unittest import mock

import pytest
import yaml

from llm_markdown_generator.config import (
    Config,
    ConfigError,
    FrontMatterConfig,
    LLMProviderConfig,
    TopicConfig,
    load_config,
    load_front_matter_schema,
)


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary for testing."""
    return {
        "llm_provider": {
            "provider_type": "openai",
            "model_name": "gpt-4",
            "api_key_env_var": "OPENAI_API_KEY",
            "temperature": 0.7,
            "max_tokens": 500,
            "additional_params": {"top_p": 0.9},
        },
        "front_matter": {"schema_path": "config/front_matter_schema.yaml"},
        "topics": {
            "python": {
                "prompt_template": "python_blog.j2",
                "keywords": ["python", "programming", "coding"],
                "custom_data": {"audience": "developers"},
            },
            "javascript": {
                "prompt_template": "javascript_blog.j2",
                "keywords": ["javascript", "web development", "frontend"],
            },
        },
        "output_dir": "output/blog",
    }


@pytest.fixture
def sample_config_file(sample_config_dict):
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as temp:
        yaml.dump(sample_config_dict, temp)
        temp_path = temp.name

    yield temp_path

    # Cleanup after test
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_front_matter_schema():
    """Sample front matter schema for testing."""
    return {
        "title": "Default Title",
        "date": None,
        "tags": [],
        "category": "general",
        "layout": "post",
        "author": "Author Name",
        "description": "",
    }


@pytest.fixture
def sample_schema_file(sample_front_matter_schema):
    """Create a temporary schema file for testing."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as temp:
        yaml.dump(sample_front_matter_schema, temp)
        temp_path = temp.name

    yield temp_path

    # Cleanup after test
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestConfig:
    """Tests for the Config class."""

    def test_load_config(self, sample_config_file, sample_config_dict):
        """Test loading configuration from a file."""
        config = load_config(sample_config_file)

        assert isinstance(config, Config)
        assert isinstance(config.llm_provider, LLMProviderConfig)
        assert isinstance(config.front_matter, FrontMatterConfig)

        # Check LLM provider config
        assert config.llm_provider.provider_type == "openai"
        assert config.llm_provider.model_name == "gpt-4"
        assert config.llm_provider.api_key_env_var == "OPENAI_API_KEY"
        assert config.llm_provider.temperature == 0.7
        assert config.llm_provider.max_tokens == 500
        assert config.llm_provider.additional_params == {"top_p": 0.9}

        # Check front matter config
        assert config.front_matter.schema_path == "config/front_matter_schema.yaml"

        # Check topics
        assert len(config.topics) == 2
        assert isinstance(config.topics["python"], TopicConfig)
        assert config.topics["python"].name == "python"
        assert config.topics["python"].prompt_template == "python_blog.j2"
        assert config.topics["python"].keywords == ["python", "programming", "coding"]
        assert config.topics["python"].custom_data == {"audience": "developers"}

        assert isinstance(config.topics["javascript"], TopicConfig)
        assert config.topics["javascript"].custom_data == {}

        # Check output directory
        assert config.output_dir == "output/blog"

    def test_load_config_invalid_file(self):
        """Test loading configuration from a non-existent file."""
        with pytest.raises(ConfigError):
            load_config("non_existent_file.yaml")

    def test_load_config_invalid_yaml(self):
        """Test loading configuration from an invalid YAML file."""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as temp:
            temp.write("invalid: yaml: : : :")
            temp_path = temp.name

        try:
            with pytest.raises(ConfigError):
                load_config(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_front_matter_schema(self, sample_schema_file, sample_front_matter_schema):
        """Test loading front matter schema from a file."""
        schema = load_front_matter_schema(sample_schema_file)

        assert isinstance(schema, dict)
        assert schema == sample_front_matter_schema

    def test_load_front_matter_schema_invalid_file(self):
        """Test loading front matter schema from a non-existent file."""
        with pytest.raises(ConfigError):
            load_front_matter_schema("non_existent_schema.yaml")

    def test_dataclass_post_init(self):
        """Test __post_init__ for dataclasses with default None values."""
        # Test LLMProviderConfig.__post_init__
        llm_config = LLMProviderConfig(
            provider_type="test",
            model_name="test-model",
            api_key_env_var="TEST_API_KEY",
            additional_params=None,
        )
        assert llm_config.additional_params == {}

        # Test TopicConfig.__post_init__
        topic_config = TopicConfig(
            name="test",
            prompt_template="test.j2",
            keywords=["test"],
            custom_data=None,
        )
        assert topic_config.custom_data == {}
