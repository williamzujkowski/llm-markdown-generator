"""Example custom plugins for the LLM Markdown Generator.

This file demonstrates how to create custom plugins for the LLM Markdown Generator.
"""

from datetime import datetime
import re
from typing import Dict, Any

# Import plugin_hook directly to use with decorators
from llm_markdown_generator.plugins import plugin_hook


def register():
    """Register the plugins in this module.
    
    This function is required by the plugin system and will be called when
    loading plugins from a directory.
    """
    # Since we're using decorators, the plugins are registered when the module is loaded
    # so there's nothing more to do here
    pass


@plugin_hook("content_processor", "add_copyright")
def add_copyright(content: str, **kwargs) -> str:
    """Add a copyright notice to the generated content.
    
    Args:
        content: The markdown content to process
        **kwargs: Additional keyword arguments from the generator
        
    Returns:
        The modified content with copyright notice added
    """
    year = datetime.now().year
    author = kwargs.get("author", "Your Name")
    
    copyright_notice = f"\n\n---\n\n© {year} {author}. All rights reserved."
    
    return content + copyright_notice


@plugin_hook("content_processor", "syntax_highlighter")
def syntax_highlighter(content: str, **kwargs) -> str:
    """Enhance code blocks with language-specific syntax highlighting.
    
    This plugin finds code blocks without language specification and attempts
    to identify the language based on the code content.
    
    Args:
        content: The markdown content to process
        **kwargs: Additional keyword arguments from the generator
        
    Returns:
        The modified content with enhanced code blocks
    """
    # Regex to find code blocks without language specification
    code_block_pattern = re.compile(r'```\s*\n(.*?)\n```', re.DOTALL)
    
    # Function to guess language based on code content
    def guess_language(code: str) -> str:
        if re.search(r'(def|import|class|if\s+__name__\s*==\s*[\'"]__main__[\'"])', code):
            return 'python'
        elif re.search(r'(function|const|let|var|=>)', code):
            return 'javascript'
        elif re.search(r'(<html|<body|<div|<span)', code):
            return 'html'
        elif re.search(r'(public class|void main|System\.out)', code):
            return 'java'
        else:
            return ''  # No language detected
    
    # Process each code block
    def replace_code_block(match):
        code = match.group(1)
        language = guess_language(code)
        
        if language:
            return f'```{language}\n{code}\n```'
        else:
            return match.group(0)  # Return unchanged if language not detected
    
    # Apply the replacements
    enhanced_content = code_block_pattern.sub(replace_code_block, content)
    
    return enhanced_content


@plugin_hook("front_matter_enhancer", "add_license")
def add_license(front_matter: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Add license information to the front matter.
    
    Args:
        front_matter: The front matter dictionary to enhance
        **kwargs: Additional keyword arguments from the generator
        
    Returns:
        The enhanced front matter dictionary
    """
    license_type = kwargs.get("license", "CC BY-SA 4.0")
    license_url = kwargs.get("license_url", "https://creativecommons.org/licenses/by-sa/4.0/")
    
    enhanced = front_matter.copy()
    
    # Add license information
    enhanced["license"] = license_type
    enhanced["license_url"] = license_url
    
    # Print debug information
    print(f"License plugin: Adding license '{license_type}' to front matter")
    
    return enhanced


@plugin_hook("front_matter_enhancer", "add_custom_fields")
def add_custom_fields(front_matter: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Add custom fields to the front matter.
    
    This plugin takes any keyword arguments prefixed with 'custom_' and adds them
    to the front matter.
    
    Args:
        front_matter: The front matter dictionary to enhance
        **kwargs: Additional keyword arguments from the generator
        
    Returns:
        The enhanced front matter dictionary
    """
    enhanced = front_matter.copy()
    
    # Extract custom fields from kwargs
    custom_fields = {
        k.replace('custom_', ''): v for k, v in kwargs.items()
        if k.startswith('custom_')
    }
    
    # Add them to front matter
    if custom_fields:
        if 'custom' not in enhanced:
            enhanced['custom'] = {}
            
        enhanced['custom'].update(custom_fields)
    
    return enhanced