"""Main markdown generator module.

Combines front matter and LLM-generated content into a final markdown file.
"""

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

# LangChain components for parsing and prompting
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate # Keep for template formatting

from llm_markdown_generator.config import Config, TopicConfig
from llm_markdown_generator.front_matter import FrontMatterGenerator, slugify
from llm_markdown_generator.llm_provider import LLMProvider # Import the base provider class


class GeneratorError(Exception):
    """Raised when there is an error generating markdown content."""
    pass


# Define a Pydantic model for the structured output (front matter + content)
# Align this more closely with common front matter needs and potential schema fields
class GeneratedPost(BaseModel):
    """Pydantic model for the generated post structure."""
    title: str = Field(description="The main title of the blog post.")
    # slug: Optional[str] = Field(None, description="URL-friendly slug for the post (optional, can be generated later).") # Slug generated later
    author: Optional[str] = Field(None, description="Author of the post.")
    publishDate: Optional[str] = Field(None, description="Date the post was published (YYYY-MM-DD).")
    tags: List[str] = Field(default_factory=list, description="Relevant tags for the post.")
    category: Optional[str] = Field(None, description="The primary category of the post.")
    description: Optional[str] = Field(None, description="A short summary/description for the post.")
    # Add other common fields as needed
    # Example: featuredImage: Optional[str] = Field(None, description="Path to a featured image.")

    # The main content body
    content_body: str = Field(description="The full markdown content body of the blog post.")


class MarkdownGenerator:
    """Main generator for markdown content using LLMs and LangChain."""

    def __init__(
        self,
        config: Config,
        llm_provider: Optional[LLMProvider], # Use our custom provider type
        front_matter_generator: FrontMatterGenerator,
    ) -> None:
        """Initialize the markdown generator.

        Args:
            config: The main configuration object.
            llm_provider: The LLM provider instance (e.g., OpenAIProvider, GeminiProvider).
            front_matter_generator: The generator for front matter.
        """
        self.config = config
        self.llm_provider = llm_provider # Use consistent naming
        self.front_matter_generator = front_matter_generator
        self.dry_run = False # Initialize dry_run flag

        # Initialize plugin registry
        self._content_processors = []
        self._front_matter_enhancers = []

    def set_dry_run(self, dry_run: bool) -> None:
        """Set the dry run mode."""
        self.dry_run = dry_run

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

    def generate_content(
        self,
        topic_name: str,
        custom_params: Dict[str, Any] = None,
        prompt_template_override: Optional[str] = None
    ) -> str:
        """Generate markdown content for a specific topic.

        Args:
            topic_name: The name of the topic to generate content for (used for config lookup if no override).
            custom_params: Optional custom parameters to include in the prompt context.
            prompt_template_override: Optional name of the prompt template file to use instead of the one from topic config.
            custom_params: Optional custom parameters to include in the prompt context.

        Returns:
            str: The generated markdown content (front matter + body).

        Raises:
            GeneratorError: If there is an error generating the content.
        """
        # Check for provider existence unless in dry run mode
        if not self.dry_run and self.llm_provider is None:
             raise GeneratorError("LLM provider not initialized. Cannot generate content.")

        if topic_name not in self.config.topics and not prompt_template_override:
            # If no specific topic config and no template override, we can't proceed
            raise GeneratorError(f"Topic '{topic_name}' not found in configuration")

        topic_config = self.config.topics[topic_name]
        custom_params = custom_params or {}

        try:
            # 1. Define Pydantic Output Parser
            parser = PydanticOutputParser(pydantic_object=GeneratedPost)

            # 2. Load Prompt Template Content from File
            # Determine the template file name
            template_filename = prompt_template_override or topic_config.prompt_template
            if not template_filename:
                 raise GeneratorError(f"No prompt template specified for topic '{topic_name}' and no override provided.")

            # Templates are expected in '.llmconfig/prompt-templates/' relative to project root
            template_file_path = Path(".llmconfig/prompt-templates") / template_filename
            if not template_file_path.exists():
                raise GeneratorError(f"Prompt template file not found: {template_file_path}")

            try:
                prompt_template_str = template_file_path.read_text(encoding="utf-8")
            except Exception as e:
                raise GeneratorError(f"Error reading prompt template file {template_file_path}: {e}")

            # Create LangChain PromptTemplate object (still useful for formatting)
            # Note: We are NOT including format_instructions here, it needs to be in the template file itself.
            prompt_template = ChatPromptTemplate.from_template(template=prompt_template_str)

            # 3. Prepare Prompt Context / Input for the template rendering
            prompt_context = {
                "topic": custom_params.get("topic", topic_name), # Use specific topic if provided (e.g., CVE ID)
                "keywords": custom_params.get("keywords", topic_config.keywords),
                "audience": custom_params.get("audience", "general audience"),
                "title": custom_params.get("title", None), # Pass title if provided
                **topic_config.custom_data,
                **custom_params, # Allow overriding any context via custom_params
            }

            # 4. Render the prompt string
            # Use format_map for simple substitution, as ChatPromptTemplate expects ChatMessage objects for full rendering
            rendered_prompt = prompt_template.template.format_map(prompt_context)
            # Add format instructions manually if they weren't part of the context
            # (They should be in the template file now, but double-check)
            if "{format_instructions}" in rendered_prompt:
                 rendered_prompt = rendered_prompt.replace("{format_instructions}", parser.get_format_instructions())


            # 5. Call the LLM Provider (or simulate for dry run)
            if self.dry_run:
                print("[yellow]DRY RUN: Skipping LLM API call.[/yellow]")
                # Create mock GeneratedPost data for dry run
                generated_data = GeneratedPost(
                    title=custom_params.get("title", f"Mock Title for {topic_name}"),
                    tags=custom_params.get("keywords", topic_config.keywords if topic_config else []),
                    category=topic_name,
                    author="Dry Run Author",
                    publishDate=datetime.datetime.now().strftime("%Y-%m-%d"),
                    description="This is a mock description generated during a dry run.",
                    content_body=f"# Mock Content for {topic_name}\n\nThis content is generated because the `--dry-run` flag was used.\n\nRendered Prompt (partial):\n```\n{rendered_prompt[:500]}...\n```"
                )
            elif self.llm_provider is None: # Should not happen due to check at start, but defensive check
                 raise GeneratorError("LLM provider not initialized. Cannot generate content.")
            else:
                # Call the provider's generate_text method (handles retries internally)
                llm_response_text = self.llm_provider.generate_text(rendered_prompt)

                # Parse the LLM response using the Pydantic parser
                try:
                    generated_data: GeneratedPost = parser.parse(llm_response_text)
                except Exception as parse_error:
                    raise GeneratorError(f"Failed to parse LLM response into Pydantic model: {parse_error}\nLLM Response:\n{llm_response_text}")


            # 6. Apply Front Matter Enhancer Plugins
            # Enhancers operate on the Pydantic model before YAML generation
            enhanced_front_matter_data = generated_data # Start with the parsed/mock data
            content_body = generated_data.content_body # Keep content body separate for now

            for enhancer in self._front_matter_enhancers:
                try:
                    # Prepare kwargs for the plugin, passing the Pydantic model and content
                    plugin_name = getattr(enhancer, "__name__", str(enhancer))
                    plugin_kwargs = {
                        "front_matter": enhanced_front_matter_data, # Pass the current Pydantic model
                        "content": content_body, # Pass the raw content body
                        "topic": topic_name,
                        **custom_params # Pass all custom params
                    }

                    # Call the enhancer plugin
                    result = enhancer(**plugin_kwargs)

                    # Update front matter data if enhancer returns a valid Pydantic model
                    if result and isinstance(result, GeneratedPost):
                        enhanced_front_matter_data = result # Replace with the enhanced version
                    elif result:
                         # Attempt to update fields if it's a different Pydantic model or dict
                         try:
                             if isinstance(result, BaseModel):
                                 update_dict = result.model_dump(exclude_unset=True)
                             elif isinstance(result, dict):
                                 update_dict = result
                             else:
                                 update_dict = None
                                 print(f"Warning: Enhancer plugin {plugin_name} returned unexpected type {type(result)}. Skipping update.")

                             if update_dict:
                                 # Create a new model instance with updated fields
                                 current_dict = enhanced_front_matter_data.model_dump()
                                 current_dict.update(update_dict)
                                 enhanced_front_matter_data = GeneratedPost(**current_dict)

                         except Exception as update_err:
                             print(f"Warning: Could not merge enhanced data from plugin {plugin_name}: {update_err}")

                except Exception as e:
                    print(f"Error in front matter enhancer plugin {plugin_name}: {str(e)}")


            # 7. Generate YAML Front Matter from the *final* Pydantic object
            front_matter_yaml = self.front_matter_generator.generate(enhanced_front_matter_data)

            # 8. Combine Final Front Matter and Content Body
            full_content = f"{front_matter_yaml}\n{content_body}" # Use original content body

            # 9. Apply Content Processor Plugins to the *combined* content string
            processed_content = full_content
            for processor in self._content_processors:
                try:
                    # Prepare kwargs for the plugin
                    plugin_name = getattr(processor, "__name__", str(processor))
                    plugin_kwargs = {
                        "content": processed_content, # Pass the current combined content string
                        "topic": topic_name,
                        "front_matter_data": enhanced_front_matter_data, # Pass the final Pydantic model
                        **custom_params # Pass all custom params
                    }

                    # Call the processor plugin
                    result = processor(**plugin_kwargs)

                    # Update content if processor returns a string
                    if isinstance(result, str):
                        processed_content = result
                    elif result is not None:
                        print(f"Warning: Content processor plugin {plugin_name} did not return a string. Skipping update.")

                except Exception as e:
                    print(f"Error in content processor plugin {plugin_name}: {str(e)}")

            return processed_content # Return the final processed content

        except Exception as e:
            raise GeneratorError(f"Error generating content for topic '{topic_name}': {str(e)}")
    
# Token tracking functionality has been removed

    def write_to_file(
        self,
        content: str,
        filename: Optional[str] = None,
        title: Optional[str] = None,
        output_dir_override: Optional[str] = None
    ) -> str:
        """Write generated content to a file.

        Args:
            content: The markdown content to write.
            filename: Optional custom filename (without extension). If provided, it's used directly.
            title: Optional title to use for generating the filename if `filename` is not provided.
            output_dir_override: Optional path to override the default output directory from config.

        Returns:
            str: The path to the written file.

        Raises:
            GeneratorError: If there is an error writing the file.
        """
        try:
            # Determine output directory
            output_dir_path = Path(output_dir_override) if output_dir_override else Path(self.config.output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)

            # Determine filename (use provided filename directly if available)
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

            # Write to file using the determined path and filename
            file_path = output_dir_path / filename
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            return str(file_path)

        except Exception as e:
            raise GeneratorError(f"Error writing content to file: {str(e)}")

