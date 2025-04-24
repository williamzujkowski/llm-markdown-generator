"""Tests for the prompt_engine module."""

import os
import tempfile
from pathlib import Path

import pytest
from jinja2 import Template

from llm_markdown_generator.prompt_engine import PromptEngine, PromptError


@pytest.fixture
def temp_templates_dir():
    """Create a temporary directory with some template files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a basic template file
        basic_template = Path(temp_dir) / "basic.j2"
        basic_template.write_text("Hello, {{ name }}!")

        # Create a template with more complex logic
        advanced_template = Path(temp_dir) / "advanced.j2"
        advanced_template.write_text(
            """# {{ title }}

Write a blog post about {{ topic }}.

Keywords: {% for kw in keywords %}{{ kw }}{% if not loop.last %}, {% endif %}{% endfor %}

{% if audience %}Target audience: {{ audience }}{% endif %}
"""
        )

        # Create an invalid template
        invalid_template = Path(temp_dir) / "invalid.j2"
        invalid_template.write_text("{{ unclosed }")  # Missing closing brace

        yield temp_dir


class TestPromptEngine:
    """Tests for the PromptEngine class."""

    def test_init_with_valid_dir(self, temp_templates_dir):
        """Test initialization with a valid templates directory."""
        engine = PromptEngine(temp_templates_dir)
        assert engine.env is not None

    def test_init_with_invalid_dir(self):
        """Test initialization with a non-existent directory."""
        with pytest.raises(PromptError):
            PromptEngine("/non/existent/directory")

    def test_load_template_success(self, temp_templates_dir):
        """Test successful template loading."""
        engine = PromptEngine(temp_templates_dir)
        template = engine.load_template("basic.j2")
        assert isinstance(template, Template)

    def test_load_template_not_found(self, temp_templates_dir):
        """Test handling of template not found error."""
        engine = PromptEngine(temp_templates_dir)
        with pytest.raises(PromptError):
            engine.load_template("non_existent.j2")

    def test_render_prompt_basic(self, temp_templates_dir):
        """Test rendering a basic template."""
        engine = PromptEngine(temp_templates_dir)
        result = engine.render_prompt("basic.j2", {"name": "World"})
        assert result == "Hello, World!"

    def test_render_prompt_advanced(self, temp_templates_dir):
        """Test rendering a more complex template."""
        engine = PromptEngine(temp_templates_dir)
        context = {
            "title": "Python Programming",
            "topic": "Python 3.9 features",
            "keywords": ["python", "programming", "features"],
            "audience": "developers",
        }
        result = engine.render_prompt("advanced.j2", context)
        assert "# Python Programming" in result
        assert "Write a blog post about Python 3.9 features." in result
        assert "Keywords: python, programming, features" in result
        assert "Target audience: developers" in result

    def test_render_prompt_missing_var(self, temp_templates_dir):
        """Test handling of missing variables in templates."""
        engine = PromptEngine(temp_templates_dir)
        # Missing 'audience' should be fine (conditional)
        result = engine.render_prompt(
            "advanced.j2",
            {
                "title": "Python Programming",
                "topic": "Python 3.9 features",
                "keywords": ["python", "programming", "features"],
            },
        )
        assert "Target audience:" not in result

        # Missing required variable should cause an error
        with pytest.raises(PromptError):
            engine.render_prompt(
                "advanced.j2",
                {"title": "Python Programming", "keywords": ["python"]},
            )

    def test_render_prompt_invalid_template(self, temp_templates_dir):
        """Test handling of syntax errors in templates."""
        engine = PromptEngine(temp_templates_dir)
        with pytest.raises(PromptError):
            engine.render_prompt("invalid.j2", {"name": "World"})
