"""Base class for RS-Agent tools."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    output: Any
    message: str
    tool_name: str


class BaseTool(ABC):
    """Base class for all RS-Agent tools."""

    name: str = ""
    description: str = ""
    input_schema: dict = {}

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        raise NotImplementedError

    def to_anthropic_tool(self) -> dict:
        """Convert to Anthropic API tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
