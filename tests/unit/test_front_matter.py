"""Tests for the front_matter module."""

import re
from datetime import datetime
from unittest import mock

import pytest

from llm_markdown_generator.front_matter import FrontMatterError, FrontMatterGenerator, slugify


@pytest.fixture
def sample_schema():
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


class TestFrontMatterGenerator:
    """Tests for the FrontMatterGenerator class."""

    def test_init(self, sample_schema):
        """Test initialization of the front matter generator."""
        generator = FrontMatterGenerator(sample_schema)
        assert generator.schema == sample_schema

    def test_generate_with_defaults(self, sample_schema):
        """Test generation with default values from schema."""
        generator = FrontMatterGenerator(sample_schema)

        # Mock datetime.now() to return a fixed date
        with mock.patch(
            "llm_markdown_generator.front_matter.datetime",
            mock.Mock(wraps=datetime),
        ) as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 1, 1)
            front_matter = generator.generate()

            # Check the result contains expected values
            assert "---\n" in front_matter
            assert "title: Default Title" in front_matter
            assert "date: '2023-01-01'" in front_matter
            assert "tags: []" in front_matter
            assert "category: general" in front_matter
            assert "layout: post" in front_matter
            assert "author: Author Name" in front_matter
            assert "description: ''" in front_matter
            assert front_matter.endswith("---\n")

    def test_generate_with_overrides(self, sample_schema):
        """Test generation with overridden values."""
        generator = FrontMatterGenerator(sample_schema)

        overrides = {
            "title": "Custom Title",
            "tags": ["tag1", "tag2"],
            "description": "Custom description",
            "date": "2023-02-15",  # Explicitly provided date
        }

        front_matter = generator.generate(overrides)

        # Check the result contains expected values
        assert "---\n" in front_matter
        assert "title: Custom Title" in front_matter
        assert "date: '2023-02-15'" in front_matter
        assert "tags:\n- tag1\n- tag2" in front_matter
        assert "category: general" in front_matter
        assert "layout: post" in front_matter
        assert "author: Author Name" in front_matter
        assert "description: Custom description" in front_matter
        assert front_matter.endswith("---\n")

    def test_generate_error_handling(self, sample_schema):
        """Test error handling during generation."""
        generator = FrontMatterGenerator(sample_schema)

        # Mock yaml.dump to raise an exception
        with mock.patch(
            "llm_markdown_generator.front_matter.yaml.dump",
            side_effect=Exception("YAML error"),
        ):
            with pytest.raises(FrontMatterError) as exc_info:
                generator.generate()

            assert "Error generating front matter" in str(exc_info.value)


class TestSlugify:
    """Tests for the slugify function."""

    def test_slugify_basic(self):
        """Test basic slugification."""
        assert slugify("Hello World") == "hello-world"

    def test_slugify_special_chars(self):
        """Test slugification with special characters."""
        assert slugify("Hello, World! How are you?") == "hello-world-how-are-you"

    def test_slugify_extra_spaces(self):
        """Test slugification with extra spaces."""
        assert slugify("  Hello   World  ") == "hello-world"

    def test_slugify_non_ascii(self):
        """Test slugification with non-ASCII characters."""
        assert slugify("Café & Résumé") == "caf-rsum"

    def test_slugify_leading_trailing_hyphens(self):
        """Test slugification with leading and trailing hyphens."""
        assert slugify("-Hello World-") == "hello-world"

    def test_slugify_empty_string(self):
        """Test slugification with empty string."""
        assert slugify("") == ""
