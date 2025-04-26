"""Content processor plugins for the LLM Markdown Generator.

This module defines the interface and provides built-in plugins for
modifying generated content before it is written to the output file.
"""

from typing import Any, Dict, List, Optional, Union

from llm_markdown_generator.plugins import plugin_hook


@plugin_hook('content_processor', 'add_timestamp')
def add_timestamp(content: str, **kwargs: Any) -> str:
    """Add a timestamp to the content.
    
    Appends a timestamp to the end of the generated content.
    This is useful for tracking when content was generated.
    
    Args:
        content: The markdown content to process
        **kwargs: Additional arguments passed from the generator
        
    Returns:
        The modified content with a timestamp added
    """
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_section = f"\n\n---\n\n*Generated on: {timestamp}*\n"
    
    return content + timestamp_section


@plugin_hook('content_processor', 'add_reading_time')
def add_reading_time(content: str, **kwargs: Any) -> str:
    """Add an estimated reading time to the content.
    
    Calculates and adds an estimated reading time based on word count.
    Average reading speed is assumed to be 200-250 words per minute.
    
    Args:
        content: The markdown content to process
        **kwargs: Additional arguments passed from the generator
        
    Returns:
        The modified content with reading time added
    """
    # Strip front matter if present
    content_text = content
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            content_text = parts[2]
    
    # Calculate word count and reading time
    words = len(content_text.split())
    reading_time_minutes = max(1, round(words / 200))
    reading_time_text = f"{reading_time_minutes} min read"
    
    # If front matter exists, try to add to it
    if content.startswith('---') and '---' in content[3:]:
        parts = content.split('---', 2)
        front_matter = parts[1]
        body = parts[2] if len(parts) > 2 else ""
        
        if 'reading_time:' not in front_matter:
            # Add to front matter
            updated_front_matter = front_matter.rstrip() + f"\nreading_time: {reading_time_text}\n"
            return f"---{updated_front_matter}---{body}"
    
    # Otherwise just add it as text at the top of the content
    reading_info = f"*{reading_time_text}*\n\n"
    if content.startswith('---') and '---' in content[3:]:
        parts = content.split('---', 2)
        return f"---{parts[1]}---\n{reading_info}{parts[2].lstrip()}"
    else:
        return reading_info + content


@plugin_hook('content_processor', 'add_table_of_contents')
def add_table_of_contents(content: str, **kwargs: Any) -> str:
    """Generate and add a table of contents based on markdown headings.
    
    Scans the markdown content for headings and generates a table of contents
    at the top of the document. Only works with ATX-style headings (# Heading).
    
    Args:
        content: The markdown content to process
        **kwargs: Additional arguments passed from the generator
        
    Returns:
        The modified content with a table of contents added
    """
    import re
    
    # Parse headings
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+?)(?:\s+#+)?$', re.MULTILINE)
    headings: List[Dict[str, str]] = []
    
    for match in heading_pattern.finditer(content):
        level = len(match.group(1))
        text = match.group(2).strip()
        
        # Skip h1 headings (title usually comes from front matter)
        if level == 1:
            continue
            
        # Skip headings in code blocks (simplistic approach)
        pos = match.start()
        code_block_start = content.rfind('```', 0, pos)
        code_block_end = content.rfind('```', 0, pos)
        if code_block_start > code_block_end:
            continue
            
        # Create slug for the heading
        slug = text.lower().replace(' ', '-')
        slug = re.sub(r'[^\w\-]', '', slug)
        
        # Use a consistent dictionary with string keys and string values
        heading_info: Dict[str, str] = {
            'level': str(level),
            'text': text,
            'slug': str(slug)
        }
        headings.append(heading_info)
    
    # If no headings, return original content
    if not headings:
        return content
    
    # Generate TOC
    toc_lines = ["## Table of Contents", ""]
    for heading in headings:
        # Indent based on heading level
        level_num = int(heading['level'])
        indent = "  " * (level_num - 2)
        toc_lines.append(f"{indent}- [{heading['text']}](#{heading['slug']})")
    
    toc = "\n".join(toc_lines) + "\n\n---\n\n"
    
    # Add TOC after front matter or at the beginning
    if content.startswith('---') and '---' in content[3:]:
        parts = content.split('---', 2)
        result = f"---{parts[1]}---\n\n{toc}{parts[2].lstrip()}"
    else:
        result = toc + content
    
    return result


@plugin_hook('content_processor', 'add_tag_links')
def add_tag_links(content: str, base_url: Optional[str] = None, **kwargs: Any) -> str:
    """Convert tags in front matter to clickable links at the bottom of the content.
    
    Extracts tags from the front matter and adds a list of clickable links at
    the bottom of the content. Useful for blogs where tags are used for navigation.
    
    Args:
        content: The markdown content to process
        base_url: Optional base URL for tag links (e.g., '/tags/')
        **kwargs: Additional arguments passed from the generator
        
    Returns:
        The modified content with tag links added
    """
    import re
    import yaml
    
    if not content.startswith('---'):
        return content
    
    # Extract front matter
    parts = content.split('---', 2)
    if len(parts) < 3:
        return content
    
    front_matter_raw = parts[1]
    body = parts[2]
    
    # Parse tags from front matter
    tags = []
    try:
        # Add spaces to empty front matter if needed
        if not front_matter_raw.strip():
            front_matter_raw = " "
            
        front_matter = yaml.safe_load(front_matter_raw)
        if isinstance(front_matter, dict) and 'tags' in front_matter:
            tags_data = front_matter['tags']
            if isinstance(tags_data, list):
                tags = [str(tag) for tag in tags_data]
            elif isinstance(tags_data, str):
                tags = [tag.strip() for tag in tags_data.split(',')]
    except Exception:
        # If we can't parse the front matter, try regex
        tag_match = re.search(r'tags:\s*\[(.*?)\]', front_matter_raw)
        if tag_match:
            tags_str = tag_match.group(1)
            tags = [tag.strip().strip("'\"") for tag in tags_str.split(',')]
    
    # If no tags found or couldn't parse, return original content
    if not tags:
        return content
    
    # Create tag links section
    base = base_url or '/tags/'
    tag_links = [f"[#{tag}]({base}{tag.replace(' ', '-').lower()})" for tag in tags]
    tag_section = "\n\n---\n\n### Tags\n\n" + " ".join(tag_links) + "\n"
    
    # Add tag links to the end of the content
    return f"---{front_matter_raw}---{body.rstrip()}{tag_section}"


def register() -> None:
    """Function called by the plugin system to register plugins in this module."""
    # All plugins are registered through decorators, so this function
    # is effectively a no-op, but it's required by the plugin loading system.
    pass