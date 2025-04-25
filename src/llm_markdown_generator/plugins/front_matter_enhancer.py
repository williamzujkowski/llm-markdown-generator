"""Front matter enhancer plugins for the LLM Markdown Generator.

This module defines plugins that enhance front matter with additional metadata.
"""

import datetime
import os
import platform
from typing import Any, Dict

from llm_markdown_generator.plugins import plugin_hook


@plugin_hook('front_matter_enhancer', 'add_metadata')
def add_metadata(front_matter: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Add additional metadata to the front matter.
    
    Enhances the front matter with information such as:
    - Date and time of generation
    - Generator version
    - Operating system
    
    Args:
        front_matter: The existing front matter dictionary
        **kwargs: Additional arguments from the generator
        
    Returns:
        The enhanced front matter dictionary
    """
    enhanced = front_matter.copy()
    
    # Add generation timestamp if not present
    if 'date' not in enhanced:
        enhanced['date'] = datetime.datetime.now().strftime("%Y-%m-%d")
        
    # Add generation metadata under a dedicated section
    if 'metadata' not in enhanced:
        enhanced['metadata'] = {}
        
    # Add generator info
    enhanced['metadata']['generator'] = "llm-markdown-generator"
    
    # Try to get version info
    try:
        import importlib.metadata
        version = importlib.metadata.version("llm-markdown-generator")
        enhanced['metadata']['generator_version'] = version
    except (ImportError, importlib.metadata.PackageNotFoundError):
        enhanced['metadata']['generator_version'] = "unknown"
    
    # Add timestamp with time
    enhanced['metadata']['generated_at'] = datetime.datetime.now().isoformat()
    
    # Add system info
    enhanced['metadata']['system'] = platform.system()
    enhanced['metadata']['python_version'] = platform.python_version()
    
    return enhanced


@plugin_hook('front_matter_enhancer', 'seo_enhancer')
def seo_enhancer(front_matter: Dict[str, Any], content: str = "", **kwargs) -> Dict[str, Any]:
    """Enhance front matter with SEO-friendly attributes.
    
    Adds or improves SEO-related front matter fields like:
    - description (if not present)
    - canonical_url (if base URL is provided)
    - og:image (if image path is provided)
    
    Args:
        front_matter: The existing front matter dictionary
        content: Optional content to use for generating description
        **kwargs: Additional arguments from the generator
        
    Returns:
        The enhanced front matter dictionary with SEO attributes
    """
    enhanced = front_matter.copy()
    
    # Generate description from content if not present
    if 'description' not in enhanced and content:
        # Strip markdown formatting and get first 160 chars
        import re
        clean_content = re.sub(r'#+ ', '', content)  # Remove headings
        clean_content = re.sub(r'!\[.*?\]\(.*?\)', '', clean_content)  # Remove images
        clean_content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean_content)  # Replace links with text
        clean_content = re.sub(r'[*_`~]', '', clean_content)  # Remove formatting
        
        # Get first 150-160 chars, ending at a word boundary
        if len(clean_content) > 160:
            description = clean_content[:157].strip()
            last_space = description.rfind(' ')
            if last_space > 100:  # Ensure we don't cut too short
                description = description[:last_space].strip() + '...'
            else:
                description = description + '...'
        else:
            description = clean_content.strip()
            
        enhanced['description'] = description
    
    # Add base URL if provided in kwargs
    base_url = kwargs.get('base_url')
    if base_url and 'slug' in enhanced:
        slug = enhanced['slug']
        if not base_url.endswith('/'):
            base_url += '/'
        enhanced['canonical_url'] = f"{base_url}{slug}"
        
    # Add open graph image if provided
    image_path = kwargs.get('og_image')
    if image_path:
        if 'og' not in enhanced:
            enhanced['og'] = {}
        enhanced['og']['image'] = image_path
        
    # Set article type if not present
    if 'type' not in enhanced:
        enhanced['type'] = 'article'
        
    return enhanced


@plugin_hook('front_matter_enhancer', 'add_series')
def add_series(front_matter: Dict[str, Any], series_name: str = None, part: int = None, **kwargs) -> Dict[str, Any]:
    """Add blog post series information to front matter.
    
    Enhances the front matter with series information for multi-part blog posts.
    
    Args:
        front_matter: The existing front matter dictionary
        series_name: The name of the series
        part: The part number in the series
        **kwargs: Additional arguments from the generator
        
    Returns:
        The enhanced front matter dictionary with series information
    """
    if not series_name:
        return front_matter
        
    enhanced = front_matter.copy()
    
    # Add series information
    if 'series' not in enhanced:
        enhanced['series'] = {}
        
    enhanced['series']['name'] = series_name
    
    if part is not None:
        enhanced['series']['part'] = part
    
    # Add series to tags if tags exist
    series_tag = f"series:{series_name}"
    if 'tags' in enhanced:
        if isinstance(enhanced['tags'], list) and series_tag not in enhanced['tags']:
            enhanced['tags'].append(series_tag)
    else:
        enhanced['tags'] = [series_tag]
        
    return enhanced


@plugin_hook('front_matter_enhancer', 'add_readability_stats')
def add_readability_stats(front_matter: Dict[str, Any], content: str = "", **kwargs) -> Dict[str, Any]:
    """Add readability statistics to front matter.
    
    Calculates and adds readability metrics to the front matter:
    - Reading time estimate
    - Word count
    - Flesch reading ease score (if textstat is available)
    
    Args:
        front_matter: The existing front matter dictionary
        content: The content to analyze
        **kwargs: Additional arguments from the generator
        
    Returns:
        The enhanced front matter dictionary with readability stats
    """
    if not content:
        return front_matter
        
    enhanced = front_matter.copy()
    
    # Initialize readability section
    if 'readability' not in enhanced:
        enhanced['readability'] = {}
    
    # Calculate word count
    words = len(content.split())
    enhanced['readability']['word_count'] = words
    
    # Calculate reading time (avg 200-250 words per minute)
    reading_minutes = max(1, round(words / 200))
    enhanced['readability']['reading_time'] = f"{reading_minutes} min"
    
    # Try to calculate Flesch reading ease if textstat is available
    try:
        import textstat
        flesch_score = textstat.flesch_reading_ease(content)
        enhanced['readability']['flesch_score'] = round(flesch_score, 1)
        
        # Add readability level based on Flesch score
        if flesch_score >= 90:
            level = "Very Easy"
        elif flesch_score >= 80:
            level = "Easy"
        elif flesch_score >= 70:
            level = "Fairly Easy"
        elif flesch_score >= 60:
            level = "Standard"
        elif flesch_score >= 50:
            level = "Fairly Difficult"
        elif flesch_score >= 30:
            level = "Difficult"
        else:
            level = "Very Difficult"
            
        enhanced['readability']['level'] = level
    except ImportError:
        # textstat not available, skip Flesch score
        pass
        
    return enhanced


def register() -> None:
    """Function called by the plugin system to register plugins in this module."""
    # All plugins are registered through decorators, so this function
    # is effectively a no-op, but it's required by the plugin loading system.
    pass