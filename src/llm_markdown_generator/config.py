"""Configuration management for the LLM Markdown Generator framework.

Handles loading and validation of configuration files for the framework,
including main configuration, topic-specific settings, and front matter schemas.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv


class ConfigError(Exception):
    """Raised when there is an error loading or validating configuration."""

    pass


@dataclass
class LLMProviderConfig:
    """Configuration for an LLM provider."""

    provider_type: str
    model_name: str
    api_key_env_var: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    additional_params: Dict[str, Any] = None

    def __post_init__(self) -> None:
        """Initialize with default empty dict for additional_params if None."""
        if self.additional_params is None:
            self.additional_params = {}


@dataclass
class FrontMatterConfig:
    """Configuration for front matter generation."""

    schema_path: str
    # Schema will be loaded from the specified path


@dataclass
class TopicConfig:
    """Configuration for a specific content topic."""

    name: str
    prompt_template: str
    keywords: List[str]
    custom_data: Dict[str, Any] = None

    def __post_init__(self) -> None:
        """Initialize with default empty dict for custom_data if None."""
        if self.custom_data is None:
            self.custom_data = {}


@dataclass
class Config:
    """Main configuration for the LLM Markdown Generator framework."""

    llm_provider: LLMProviderConfig
    front_matter: FrontMatterConfig
    topics: Dict[str, TopicConfig]
    output_dir: str


def load_config(config_path: str) -> Config:
    """Load and validate the main configuration file.

    Args:
        config_path: Path to the main configuration YAML file.

    Returns:
        Config: A validated Config object.

    Raises:
        ConfigError: If the configuration file cannot be loaded or is invalid.
    """
    # Load environment variables from .env file if it exists
    load_dotenv()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        # Extract LLM provider configuration
        llm_config_data = config_data.get("llm_provider", {})
        llm_provider = LLMProviderConfig(
            provider_type=llm_config_data.get("provider_type", ""),
            model_name=llm_config_data.get("model_name", ""),
            api_key_env_var=llm_config_data.get("api_key_env_var", ""),
            temperature=llm_config_data.get("temperature", 0.7),
            max_tokens=llm_config_data.get("max_tokens"),
            additional_params=llm_config_data.get("additional_params", {}),
        )

        # Extract front matter configuration
        front_matter_data = config_data.get("front_matter", {})
        front_matter = FrontMatterConfig(
            schema_path=front_matter_data.get("schema_path", ""),
        )

        # Extract topic configurations
        topics_data = config_data.get("topics", {})
        topics = {}
        for topic_name, topic_data in topics_data.items():
            topics[topic_name] = TopicConfig(
                name=topic_name,
                prompt_template=topic_data.get("prompt_template", ""),
                keywords=topic_data.get("keywords", []),
                custom_data=topic_data.get("custom_data", {}),
            )

        # Create and return the main config object
        return Config(
            llm_provider=llm_provider,
            front_matter=front_matter,
            topics=topics,
            output_dir=config_data.get("output_dir", "output"),
        )

    except (yaml.YAMLError, IOError) as e:
        raise ConfigError(f"Error loading configuration from {config_path}: {str(e)}")


def load_front_matter_schema(schema_path: str) -> Dict[str, Any]:
    """Load the front matter schema from a YAML file.

    Args:
        schema_path: Path to the front matter schema YAML file.

    Returns:
        Dict[str, Any]: The front matter schema as a dictionary.

    Raises:
        ConfigError: If the schema file cannot be loaded or is invalid.
    """
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = yaml.safe_load(f)
        return schema
    except (yaml.YAMLError, IOError) as e:
        raise ConfigError(f"Error loading front matter schema from {schema_path}: {str(e)}")
