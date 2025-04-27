"""Front matter generation for markdown files.

Handles generation of YAML front matter for markdown files
based on a defined schema.
"""

from datetime import datetime
from typing import Any, Dict, List, Union

import yaml
from pydantic import BaseModel


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
        # Schema might still be useful for default values or validation if not using Pydantic model directly
        self.schema = schema

    def generate(self, data: Union[Dict[str, Any], BaseModel]) -> str:
        """Generate YAML front matter from a dictionary or Pydantic model.

        Args:
            data: A dictionary or Pydantic BaseModel containing the front matter fields.

        Returns:
            str: The generated YAML front matter, including the opening and
                 closing '---' delimiters.

        Raises:
            FrontMatterError: If there is an error generating the front matter.
        """
        if data is None:
            front_matter_dict = {}
        elif isinstance(data, BaseModel):
            # Convert Pydantic model to dict, excluding unset fields for cleaner YAML
            front_matter_dict = data.model_dump(exclude_unset=True, mode='python')
        elif isinstance(data, dict):
            front_matter_dict = data
        else:
            raise FrontMatterError("Input data must be a dictionary or Pydantic BaseModel")

        try:
            # Apply schema defaults if necessary (optional, Pydantic model might handle defaults)
            # Example: Merge with schema defaults, giving priority to provided data
            # final_dict = {**self.schema, **front_matter_dict}

            # For now, just use the provided data directly
            final_dict = front_matter_dict

            # Ensure date is present if expected (can be handled by Pydantic model default)
            if "publishDate" in final_dict and not final_dict.get("publishDate"):
                 final_dict["publishDate"] = datetime.now().strftime("%Y-%m-%d")
            elif "date" in final_dict and not final_dict.get("date"):
                 final_dict["date"] = datetime.now().strftime("%Y-%m-%d")


            # Convert to YAML
            # Use safe_dump and allow unicode for broader character support
            yaml_str = yaml.safe_dump(
                final_dict,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                explicit_start=False,
                explicit_end=False
            )

            # Add diagnostic output when debugging
            try:
                import os
                if os.environ.get('DEBUG_FRONT_MATTER') == '1':
                    print(f"DEBUG: Front matter data before YAML dump: {front_matter}")
                    print(f"DEBUG: YAML dump output: {yaml_str}")
            except:
                pass
                
            # Return with delimiters (add a newline after opening delimiter for better readability)
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

    # Normalize unicode characters to ASCII for better URL compatibility
    import unicodedata
    # First normalize the string to NFKD form (decomposed form)
    slug = unicodedata.normalize('NFKD', slug)
    # Then remove all non-ASCII characters (accents, etc.)
    slug = ''.join([c for c in slug if not unicodedata.combining(c)])
    
    # Replace non-alphanumeric characters with hyphens
    import re
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "-", slug)
    slug = re.sub(r"^-+|-+$", "", slug)

    return slug
