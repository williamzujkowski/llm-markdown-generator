"""Front matter generation for markdown files.

Handles generation of YAML front matter for markdown files
based on a defined schema.
"""

from datetime import datetime
from typing import Any, Dict, List

import yaml


class FrontMatterError(Exception):
    """Raised when there is an error generating front matter."""

    pass


class FrontMatterGenerator:
    """Generator for YAML front matter in markdown files."""

    def __init__(self, schema: Dict[str, Any]) -> None:
        """Initialize the front matter generator with a schema.

        Args:
            schema: A dictionary defining the structure and default values
                   for the front matter.
        """
        self.schema = schema

    def generate(self, data: Dict[str, Any] = None) -> str:
        """Generate YAML front matter based on the schema and provided data.

        Args:
            data: A dictionary of values to use in the front matter,
                 overriding schema defaults where provided.

        Returns:
            str: The generated YAML front matter, including the opening and
                 closing '---' delimiters.

        Raises:
            FrontMatterError: If there is an error generating the front matter.
        """
        if data is None:
            data = {}

        try:
            # Start with the schema defaults
            front_matter = self.schema.copy()

            # Override with provided data
            for key, value in data.items():
                if key in front_matter:
                    front_matter[key] = value

            # Add current date if not provided
            if "date" in front_matter and not data.get("date"):
                front_matter["date"] = datetime.now().strftime("%Y-%m-%d")

            # Convert to YAML
            yaml_str = yaml.dump(
                front_matter, default_flow_style=False, sort_keys=False
            )

            # Return with delimiters
            return f"---\n{yaml_str}---\n"

        except Exception as e:
            raise FrontMatterError(f"Error generating front matter: {str(e)}")


def slugify(title: str) -> str:
    """Generate a URL-friendly slug from a title.

    Args:
        title: The title to convert to a slug.

    Returns:
        str: The generated slug.
    """
    # Convert to lowercase
    slug = title.lower()

    # Replace non-alphanumeric characters with hyphens
    import re

    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "-", slug)
    slug = re.sub(r"^-+|-+$", "", slug)

    return slug
