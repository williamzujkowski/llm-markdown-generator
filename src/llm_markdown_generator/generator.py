"""Main markdown generator module.

Combines front matter and LLM-generated content into a final markdown file.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import datetime

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
        
        # Initialize plugin registry
        self._content_processors = []
        self._front_matter_enhancers = []
        
    def register_content_processor(self, processor_func) -> None:
        """Register a content processor plugin.
        
        Args:
            processor_func: A function that takes content and returns modified content
        """
        self._content_processors.append(processor_func)
        
    def register_front_matter_enhancer(self, enhancer_func) -> None:
        """Register a front matter enhancer plugin.
        
        Args:
            enhancer_func: A function that takes front matter and returns enhanced front matter
        """
        self._front_matter_enhancers.append(enhancer_func)
        
    def clear_plugins(self) -> None:
        """Clear all registered plugins."""
        self._content_processors = []
        self._front_matter_enhancers = []
        
    def load_plugins(self) -> Dict[str, int]:
        """Load plugins from the plugins directory.
        
        Returns:
            A dictionary with plugin categories and count of loaded plugins
        """
        from llm_markdown_generator.plugins import (
            get_plugin, list_plugins, load_plugins_from_directory
        )
        
        # Import from the package's plugins directory
        try:
            from importlib.util import find_spec
            from importlib import import_module
            
            package_plugins = find_spec('llm_markdown_generator.plugins')
            if package_plugins:
                # Import built-in plugins
                import_module('llm_markdown_generator.plugins.content_processor')
                import_module('llm_markdown_generator.plugins.front_matter_enhancer')
            
            # Load any custom plugins from config directory if specified
            if hasattr(self.config, 'plugins_dir') and self.config.plugins_dir:
                plugins_dir = Path(self.config.plugins_dir)
                if plugins_dir.exists() and plugins_dir.is_dir():
                    load_plugins_from_directory(plugins_dir)
        except Exception as e:
            raise GeneratorError(f"Error loading plugins: {str(e)}")
            
        # Register plugins based on their categories
        plugins_loaded = {'content_processor': 0, 'front_matter_enhancer': 0}
        
        plugin_list = list_plugins()
        
        # Load content processors
        if 'content_processor' in plugin_list:
            for name in plugin_list['content_processor']:
                plugin_func = get_plugin('content_processor', name)
                self.register_content_processor(plugin_func)
                plugins_loaded['content_processor'] += 1
                
        # Load front matter enhancers
        if 'front_matter_enhancer' in plugin_list:
            for name in plugin_list['front_matter_enhancer']:
                plugin_func = get_plugin('front_matter_enhancer', name)
                self.register_front_matter_enhancer(plugin_func)
                plugins_loaded['front_matter_enhancer'] += 1
                
        return plugins_loaded

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
            
            # Apply front matter enhancer plugins
            for enhancer in self._front_matter_enhancers:
                try:
                    if hasattr(enhancer, "__name__"):
                        plugin_name = enhancer.__name__
                    else:
                        plugin_name = str(enhancer)
                        
                    # Apply the enhancer
                    enhanced_data = enhancer(
                        front_matter=front_matter_data,
                        content=content,
                        topic=topic_name,
                        **custom_params
                    )
                    
                    # Update front matter data
                    if enhanced_data and isinstance(enhanced_data, dict):
                        front_matter_data = enhanced_data
                    
                except Exception as e:
                    # Log error but continue with other plugins
                    print(f"Error in front matter enhancer plugin {plugin_name}: {str(e)}")

            # Generate front matter
            front_matter = self.front_matter_generator.generate(front_matter_data)

            # Combine front matter and content
            full_content = f"{front_matter}\n{content}"
            
            # Apply content processor plugins
            for processor in self._content_processors:
                try:
                    full_content = processor(
                        content=full_content,
                        topic=topic_name,
                        front_matter_data=front_matter_data,
                        **custom_params
                    )
                except Exception as e:
                    # Log error but continue with other plugins
                    print(f"Error in content processor plugin: {str(e)}")

            return full_content

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

