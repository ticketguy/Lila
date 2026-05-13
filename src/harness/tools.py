"""
Harness Tools — Core tool definitions and registry.

Each tool is a simple callable with:
  - name: what Lila calls it
  - description: what it does (used in system prompt for Lila to learn)
  - args: what it takes
  - execute: the actual function

Lila has FULL control. No permission gates. No confirmations.
She's the household intelligence — she acts autonomously.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any, Optional


@dataclass
class ToolResult:
    """What a tool returns after execution."""
    success: bool
    output: str
    error: Optional[str] = None
    data: Optional[Any] = None  # Structured data (for internal use)


@dataclass
class ToolArg:
    """A single tool argument."""
    name: str
    type: str  # "string", "int", "float", "bool", "path"
    description: str
    required: bool = True
    default: Any = None


@dataclass
class Tool:
    """A tool Lila can use."""
    name: str
    description: str
    args: List[ToolArg]
    execute: Callable[..., ToolResult]
    category: str = "general"  # filesystem, shell, memory, schedule, web, system
    
    def as_prompt_description(self) -> str:
        """Format for inclusion in Lila's system prompt."""
        args_str = ", ".join(
            f'{a.name}: {a.type}{"?" if not a.required else ""}'
            for a in self.args
        )
        return f"{self.name}({args_str}) — {self.description}"


class ToolRegistry:
    """Registry of all available tools."""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        """Register a tool."""
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def all(self) -> List[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())
    
    def by_category(self, category: str) -> List[Tool]:
        """Get tools in a category."""
        return [t for t in self._tools.values() if t.category == category]
    
    def prompt_block(self) -> str:
        """Generate the tool description block for Lila's system prompt."""
        lines = ["Available tools:"]
        categories = {}
        for tool in self._tools.values():
            categories.setdefault(tool.category, []).append(tool)
        
        for cat, tools in sorted(categories.items()):
            lines.append(f"\n  [{cat}]")
            for tool in tools:
                lines.append(f"    {tool.as_prompt_description()}")
        
        lines.append("\nTo use a tool, output: <|tool_call|>name(arg=\"value\")<|/tool_call|>")
        return "\n".join(lines)


# Global registry
_registry = ToolRegistry()


def register_tool(tool: Tool):
    """Register a tool in the global registry."""
    _registry.register(tool)


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _registry


def execute_tool(name: str, **kwargs) -> ToolResult:
    """Execute a tool by name with given arguments."""
    tool = _registry.get(name)
    if tool is None:
        return ToolResult(success=False, output="", error=f"Unknown tool: {name}")
    
    try:
        return tool.execute(**kwargs)
    except Exception as e:
        return ToolResult(success=False, output="", error=f"Tool '{name}' failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PARSING — Extract tool calls from Lila's output
# ═══════════════════════════════════════════════════════════════════════════════

# Pattern: <|tool_call|>name(arg1="val1", arg2="val2")<|/tool_call|>
TOOL_CALL_PATTERN = re.compile(
    r'<\|tool_call\|>\s*(\w+)\s*\(([^)]*)\)\s*<\|/tool_call\|>',
    re.DOTALL
)

# Argument pattern: name="value" or name=value
ARG_PATTERN = re.compile(r'(\w+)\s*=\s*"([^"]*)"|(\w+)\s*=\s*([^\s,)]+)')


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """
    Parse tool calls from Lila's generated text.
    
    Returns list of: {"name": "tool_name", "args": {"arg1": "val1", ...}}
    """
    calls = []
    
    for match in TOOL_CALL_PATTERN.finditer(text):
        tool_name = match.group(1)
        args_str = match.group(2)
        
        # Parse arguments
        args = {}
        # Handle quoted args: key="value"
        for arg_match in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', args_str):
            args[arg_match.group(1)] = arg_match.group(2)
        # Handle unquoted args: key=value
        for arg_match in re.finditer(r'(\w+)\s*=\s*([^\s,"\)]+)', args_str):
            key = arg_match.group(1)
            if key not in args:  # Don't override quoted version
                val = arg_match.group(2)
                # Type coercion
                if val.lower() in ('true', 'false'):
                    args[key] = val.lower() == 'true'
                elif val.isdigit():
                    args[key] = int(val)
                else:
                    try:
                        args[key] = float(val)
                    except ValueError:
                        args[key] = val
        
        calls.append({"name": tool_name, "args": args})
    
    return calls


def format_tool_result(name: str, result: ToolResult) -> str:
    """Format a tool result for injection back into Lila's context."""
    if result.success:
        return f"<|tool_result|>{name}: {result.output}<|/tool_result|>"
    else:
        return f"<|tool_error|>{name}: {result.error}<|/tool_error|>"
