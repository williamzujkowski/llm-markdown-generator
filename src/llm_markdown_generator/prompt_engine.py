"""Prompt engine for generating LLM prompts from templates.

Handles loading and rendering of prompt templates using Jinja2.
"""

from pathlib import Path
from typing import Any, Dict

import jinja2
from jinja2 import Environment, FileSystemLoader, Template


class PromptError(Exception):
    """Raised when there is an error with prompt template loading or rendering."""

    pass


class PromptEngine:
    """Engine for loading and rendering prompt templates."""

    def __init__(self, templates_dir: str) -> None:
        """Initialize the prompt engine.

        Args:
            templates_dir: Path to the directory containing prompt templates.

        Raises:
            PromptError: If the templates directory does not exist or is not readable.
        """
        templates_path = Path(templates_dir)
        if not templates_path.exists() or not templates_path.is_dir():
            raise PromptError(f"Templates directory not found: {templates_dir}")

        # Create environment with strict_variables enabled to raise errors on missing variables
        self.env = Environment(
            loader=FileSystemLoader(templates_path),
            undefined=jinja2.StrictUndefined
        )

    def load_template(self, template_name: str) -> Template:
        """Load a prompt template by name.

        Args:
            template_name: The name of the template file (with extension).

        Returns:
            Template: The loaded Jinja2 template.

        Raises:
            PromptError: If the template cannot be loaded.
        """
        try:
            return self.env.get_template(template_name)
        except Exception as e:
            raise PromptError(f"Error loading template '{template_name}': {str(e)}")

    def render_prompt(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render a prompt template with the given context.

        Args:
            template_name: The name of the template file (with extension).
            context: Dictionary containing variables to inject into the template.

        Returns:
            str: The rendered prompt.

        Raises:
            PromptError: If the template cannot be rendered.
        """
        try:
            template = self.load_template(template_name)
            return template.render(**context)
        except Exception as e:
            raise PromptError(f"Error rendering template '{template_name}': {str(e)}")
