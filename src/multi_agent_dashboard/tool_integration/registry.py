"""
Tool registry for LangChain tools.

Provides a central registry for tool definitions with decorator-based registration.
Each tool is a LangChain BaseTool subclass with a JSON Schema description.
"""

import logging
from typing import Dict, Any, Optional, Type, Callable, Union, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Try to import LangChain BaseTool (optional)
try:
    from langchain.tools import BaseTool
    _LANGCHAIN_TOOLS_AVAILABLE = True
except ImportError:
    # Fallback for environments without LangChain
    class BaseTool:
        pass
    _LANGCHAIN_TOOLS_AVAILABLE = False


@dataclass
class ToolMetadata:
    """Metadata for a registered tool."""
    name: str
    description: str
    schema: Dict[str, Any]
    tool_class: Optional[Type[BaseTool]] = None
    tool_instance: Optional[BaseTool] = None
    tags: List[str] = field(default_factory=list)


class ToolRegistry:
    """
    Singleton registry for LangChain tools.
    
    Maintains mapping from tool name to ToolMetadata.
    Supports both class-based registration (decorator) and instance registration.
    """
    
    _instance: Optional["ToolRegistry"] = None
    _registry: Dict[str, ToolMetadata]
    
    def __new__(cls) -> "ToolRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry = {}
        return cls._instance
    
    def register(
        self,
        name: str,
        description: str,
        schema: Dict[str, Any],
        tool_class: Optional[Type[BaseTool]] = None,
        tool_instance: Optional[BaseTool] = None,
        tags: Optional[List[str]] = None,
    ) -> Callable:
        """
        Register a tool with the registry.
        
        Can be used as a decorator on a BaseTool subclass or called directly.
        
        Args:
            name: Unique identifier for the tool (e.g., "web_search")
            description: Human-readable description for the LLM
            schema: JSON Schema dict describing the tool's parameters
            tool_class: BaseTool subclass (optional, for class registration)
            tool_instance: Instantiated tool (optional, for instance registration)
            tags: Optional list of tags for categorization
        
        Returns:
            Decorator function if used as decorator, otherwise None.
        """
        if name in self._registry:
            logger.warning(f"Tool '{name}' already registered, overwriting")
        
        # Return decorator if tool_class is not provided (meaning we're being used as decorator)
        if tool_class is None and tool_instance is None:
            def decorator(cls: Type[BaseTool]) -> Type[BaseTool]:
                # Register the class when decorator is applied
                self.register(
                    name=name,
                    description=description,
                    schema=schema,
                    tool_class=cls,
                    tags=tags,
                )
                return cls
            return decorator
        
        # Direct registration: create metadata now
        metadata = ToolMetadata(
            name=name,
            description=description,
            schema=schema,
            tool_class=tool_class,
            tool_instance=tool_instance,
            tags=tags or [],
        )
        self._registry[name] = metadata
        return None
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool instance by name.
        
        If a tool_instance is registered, returns it.
        Otherwise, instantiates the tool_class with default constructor.
        
        Returns:
            BaseTool instance or None if not found.
        """
        if name not in self._registry:
            return None
        
        metadata = self._registry[name]
        if metadata.tool_instance is not None:
            return metadata.tool_instance
        
        if metadata.tool_class is not None:
            try:
                # Instantiate with default constructor
                instance = metadata.tool_class()
                # Cache the instance for future calls
                metadata.tool_instance = instance
                return instance
            except Exception as e:
                logger.error(f"Failed to instantiate tool '{name}': {e}")
                return None
        
        return None
    
    def get_metadata(self, name: str) -> Optional[ToolMetadata]:
        """Get tool metadata by name."""
        return self._registry.get(name)
    
    def list_tools(self) -> List[str]:
        """Return list of registered tool names."""
        return list(self._registry.keys())
    
    def list_tools_with_metadata(self) -> List[ToolMetadata]:
        """Return list of all tool metadata."""
        return list(self._registry.values())
    
    def clear(self) -> None:
        """Clear the registry (mainly for testing)."""
        self._registry.clear()


# Global registry instance
_registry_instance: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ToolRegistry()
    return _registry_instance


def register_tool(
    name: str,
    description: str,
    schema: Dict[str, Any],
    tags: Optional[List[str]] = None,
) -> Callable:
    """
    Decorator to register a BaseTool subclass.
    
    Usage:
        @register_tool("web_search", "Search the web", schema={...})
        class WebSearchTool(BaseTool):
            ...
    """
    registry = get_registry()
    return registry.register(name=name, description=description, schema=schema, tags=tags)


# Convenience function for direct registration
def register_tool_instance(
    name: str,
    description: str,
    schema: Dict[str, Any],
    tool_instance: BaseTool,
    tags: Optional[List[str]] = None,
) -> None:
    """Register an already instantiated tool."""
    registry = get_registry()
    registry.register(
        name=name,
        description=description,
        schema=schema,
        tool_instance=tool_instance,
        tags=tags,
    )