"""Plugin system for the LLM Markdown Generator.

Provides a framework for extending the generator with custom logic.
"""

from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Type, TypeVar, Union

# Define types for plugin functions and classes
T = TypeVar('T')
PluginFunction = Callable[..., Any]
PluginClass = Type[Any]
Plugin = Union[PluginFunction, PluginClass]

# Registry to store loaded plugins
_plugins: Dict[str, Dict[str, Plugin]] = {}


class PluginError(Exception):
    """Base exception for plugin-related errors."""
    pass


class PluginHook(Protocol):
    """Protocol defining the structure of a plugin hook function."""
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the plugin hook."""
        ...


def register_plugin(category: str, name: str, plugin: Plugin) -> None:
    """Register a plugin in the specified category.
    
    Args:
        category: The category of the plugin (e.g., 'post_process', 'formatter')
        name: The name of the plugin
        plugin: The plugin function or class
        
    Raises:
        PluginError: If a plugin with the same name already exists in the category
    """
    if category not in _plugins:
        _plugins[category] = {}
        
    if name in _plugins[category]:
        raise PluginError(f"Plugin '{name}' already registered in category '{category}'")
        
    _plugins[category][name] = plugin


def get_plugin(category: str, name: str) -> Plugin:
    """Get a plugin by category and name.
    
    Args:
        category: The category of the plugin
        name: The name of the plugin
        
    Returns:
        The plugin function or class
        
    Raises:
        PluginError: If the category or plugin name doesn't exist
    """
    if category not in _plugins:
        raise PluginError(f"Plugin category '{category}' does not exist")
        
    if name not in _plugins[category]:
        raise PluginError(f"Plugin '{name}' not found in category '{category}'")
        
    return _plugins[category][name]


def list_plugins(category: Optional[str] = None) -> Dict[str, List[str]]:
    """List available plugins, optionally filtered by category.
    
    Args:
        category: Optional category to filter by
        
    Returns:
        A dictionary mapping categories to lists of plugin names
    """
    result: Dict[str, List[str]] = {}
    
    if category:
        if category in _plugins:
            result[category] = list(_plugins[category].keys())
    else:
        for cat in _plugins:
            result[cat] = list(_plugins[cat].keys())
            
    return result


def load_plugins_from_directory(plugins_dir: Union[str, Path]) -> Dict[str, List[str]]:
    """Load plugins from Python modules in the specified directory.
    
    This function loads plugins from all Python files in the given directory.
    It expects each module to have a `register` function that registers its plugins.
    
    Args:
        plugins_dir: The directory containing Python plugin modules
        
    Returns:
        A dictionary mapping categories to lists of loaded plugin names
        
    Raises:
        PluginError: If there was an error loading a plugin
    """
    plugins_path = Path(plugins_dir).resolve()
    
    if not plugins_path.exists() or not plugins_path.is_dir():
        raise PluginError(f"Plugin directory '{plugins_path}' does not exist or is not a directory")
    
    # Store initial state to compare after loading
    initial_state = {
        category: list(plugins.keys()) for category, plugins in _plugins.items()
    }
    
    # Find all Python files in the plugins directory
    plugin_modules = [
        f.stem for f in plugins_path.glob("*.py") 
        if f.is_file() and f.name != "__init__.py"
    ]
    
    # Import each module
    for module_name in plugin_modules:
        try:
            # Calculate the import path for the module
            module_path = str(plugins_path.name) + "." + module_name
            module = import_module(module_path)
            
            # Call the register function if it exists
            if hasattr(module, "register") and callable(module.register):
                module.register()
        except Exception as e:
            raise PluginError(f"Error loading plugin module '{module_name}': {str(e)}")
    
    # Calculate which plugins were added
    result: Dict[str, List[str]] = {}
    for category, plugins in _plugins.items():
        # If this is a new category, add all plugins
        if category not in initial_state:
            result[category] = list(plugins.keys())
        else:
            # Otherwise, find new plugins in existing categories
            new_plugins = [
                name for name in plugins.keys() 
                if name not in initial_state[category]
            ]
            if new_plugins:
                result[category] = new_plugins
    
    return result


# Decorator for registering plugin hooks
def plugin_hook(category: str, name: str) -> Callable[[PluginFunction], PluginFunction]:
    """Decorator for registering a function as a plugin hook.
    
    Args:
        category: The category of the plugin
        name: The name of the plugin
        
    Returns:
        A decorator function that registers the decorated function as a plugin
        
    Example:
        @plugin_hook('post_process', 'add_timestamp')
        def add_timestamp_to_content(content: str) -> str:
            # Add timestamp logic here
            return modified_content
    """
    def decorator(func: PluginFunction) -> PluginFunction:
        register_plugin(category, name, func)
        return func
    return decorator