"""Main markdown generator module.

Combines front matter and LLM-generated content into a final markdown file.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from llm_markdown_generator.config import Config, TopicConfig
from llm_markdown_generator.front_matter import FrontMatterGenerator, slugify
from llm_markdown_generator.llm_provider import LLMProvider, TokenUsage
from llm_markdown_generator.prompt_engine import PromptEngine


class GeneratorError(Exception):
    """Raised when there is an error generating markdown content."""

    pass


class MarkdownGenerator:
    """Main generator for markdown content using LLMs."""

    def __init__(
        self,
        config: Config,
        llm_provider: LLMProvider,
        prompt_engine: PromptEngine,
        front_matter_generator: FrontMatterGenerator,
    ) -> None:
        """Initialize the markdown generator.

        Args:
            config: The main configuration object.
            llm_provider: The LLM provider to use for content generation.
            prompt_engine: The prompt engine for rendering templates.
            front_matter_generator: The generator for front matter.
        """
        self.config = config
        self.llm_provider = llm_provider
        self.prompt_engine = prompt_engine
        self.front_matter_generator = front_matter_generator

    def generate_content(self, topic_name: str, custom_params: Dict[str, Any] = None) -> str:
        """Generate markdown content for a specific topic.

        Args:
            topic_name: The name of the topic to generate content for.
            custom_params: Optional custom parameters to include in the prompt context.

        Returns:
            str: The generated markdown content (front matter + body).

        Raises:
            GeneratorError: If there is an error generating the content.
        """
        if topic_name not in self.config.topics:
            raise GeneratorError(f"Topic '{topic_name}' not found in configuration")

        topic_config = self.config.topics[topic_name]
        custom_params = custom_params or {}

        try:
            # Prepare prompt context
            prompt_context = {
                "topic": topic_name,
                "keywords": topic_config.keywords,
                **topic_config.custom_data,
                **custom_params,
            }

            # Render prompt from template
            prompt = self.prompt_engine.render_prompt(
                topic_config.prompt_template, prompt_context
            )

            # Generate content using LLM provider
            content = self.llm_provider.generate_text(prompt)

            # Prepare front matter data
            # For simplicity, we're using basic front matter data derived from the topic
            front_matter_data = {
                "title": custom_params.get("title", f"{topic_name.capitalize()} Post"),
                "tags": topic_config.keywords,
                "category": topic_name,
            }

            # Generate front matter
            front_matter = self.front_matter_generator.generate(front_matter_data)

            # Combine front matter and content
            return f"{front_matter}\n{content}"

        except Exception as e:
            raise GeneratorError(f"Error generating content for topic '{topic_name}': {str(e)}")
    
    def get_token_usage(self) -> TokenUsage:
        """Get token usage information for the most recent generation.

        Returns:
            TokenUsage: Token usage information from the LLM provider.
        """
        return self.llm_provider.get_token_usage()
    
    def get_total_usage(self) -> TokenUsage:
        """Get total token usage information across all generations.

        Returns:
            TokenUsage: Accumulated token usage information.
        """
        return self.llm_provider.total_usage

    def write_to_file(self, content: str, filename: Optional[str] = None, title: Optional[str] = None) -> str:
        """Write generated content to a file.

        Args:
            content: The markdown content to write.
            filename: Optional custom filename (without extension).
            title: Optional title to use for generating the filename if not provided.

        Returns:
            str: The path to the written file.

        Raises:
            GeneratorError: If there is an error writing the file.
        """
        try:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Determine filename
            if not filename:
                if title:
                    filename = slugify(title)
                else:
                    # Extract title from front matter if possible
                    import yaml

                    try:
                        front_matter_end = content.find("---", 4)
                        if front_matter_end > 0:
                            front_matter_str = content[4:front_matter_end].strip()
                            front_matter_data = yaml.safe_load(front_matter_str)
                            if "title" in front_matter_data:
                                filename = slugify(front_matter_data["title"])
                    except:
                        pass

                # Fallback to timestamp if we still don't have a filename
                if not filename:
                    from datetime import datetime

                    filename = f"post-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

            # Ensure .md extension
            if not filename.endswith(".md"):
                filename = f"{filename}.md"

            # Write to file
            file_path = output_dir / filename
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            return str(file_path)

        except Exception as e:
            raise GeneratorError(f"Error writing content to file: {str(e)}")
