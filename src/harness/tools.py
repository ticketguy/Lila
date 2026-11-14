"""
Harness Tools — Core tool definitions and registry.

Tools are definitions only; HarnessExecutor is the only supported execution
path. It validates arguments and applies centralized permission policies.

Each tool definition contains:
  - name: what Lila calls it
  - description: what it does (used in system prompt for Lila to learn)
  - args: what it takes
  - execute: the actual function
She's the household intelligence — she acts autonomously.
"""

from __future__ import annotations
import re
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any, Optional
from .permissions import policy_for


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
    choices: Optional[List[Any]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None


@dataclass
class Tool:
    """A tool Lila can use."""
    name: str
    description: str
    args: List[ToolArg]
    execute: Callable[..., ToolResult]
    category: str = "general"  # filesystem, shell, memory, schedule, web, system
    risk: str = "high"
    permission: str = "approval"
    
    def as_prompt_description(self) -> str:
        """Format for inclusion in Lila's system prompt."""
        args_str = ", ".join(
            f'{a.name}: {a.type}{"?" if not a.required else ""}'
            for a in self.args
        )
        requirement = {
            "auto": "automatic",
            "approval": "owner approval required",
            "disabled": "currently unavailable",
        }.get(self.permission, "owner approval required")
        return (
            f"{self.name}({args_str}) — {self.description} "
            f"[risk: {self.risk}; {requirement}]"
        )


class ToolRegistry:
    """Registry of all available tools."""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        """Register a tool with its centrally defined risk policy."""
        if tool.name in self._tools:
            raise DuplicateToolRegistrationError(
                f"Tool '{tool.name}' is already registered."
            )
        tool.risk, tool.permission = policy_for(tool.name)
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
            if tool.permission != "disabled":
                categories.setdefault(tool.category, []).append(tool)
        
        for cat, tools in sorted(categories.items()):
            lines.append(f"\n  [{cat}]")
            for tool in tools:
                lines.append(f"    {tool.as_prompt_description()}")
        
        lines.extend([
            "\nTool calls must use this JSON format:",
            "<tool_call>",
            '{"name": "file_read", "arguments": {"path": "lila.py"}}',
            "</tool_call>",
        ])
        return "\n".join(lines)


class DuplicateToolRegistrationError(ValueError):
    """Raised when registration would replace an existing tool identity."""


# Global registry
_registry = ToolRegistry()


def register_tool(tool: Tool):
    """Register a tool in the global registry."""
    _registry.register(tool)


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _registry


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


def _parse_argument_string(args_str: str) -> Dict[str, Any]:
    """Parse name=value arguments from the legacy call format."""
    args = {}
    pattern = re.compile(
        r'\s*([A-Za-z_]\w*)\s*=\s*(?:"([^"]*)"|([^,\s)]+))\s*'
    )
    position = 0
    while position < len(args_str):
        match = pattern.match(args_str, position)
        if match is None:
            raise ValueError("Malformed tool argument list.")
        key = match.group(1)
        if key in args:
            raise ValueError(f"Duplicate tool argument: {key}")
        value = match.group(2) if match.group(2) is not None else match.group(3)
        if match.group(2) is None:
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
        args[key] = value
        position = match.end()
        if position == len(args_str):
            break
        if args_str[position] != ",":
            raise ValueError("Malformed tool argument separator.")
        position += 1
        if not args_str[position:].strip():
            raise ValueError("Trailing separator in tool arguments.")

    return args


def _xml_tool_calls(body: str) -> List[Dict[str, Any]]:
    """Parse Qwen-style XML tool calls, including nested argument tags."""
    calls = []

    body = re.sub(
        r"^\s*```(?:xml|json)?\s*|\s*```\s*$",
        "",
        body.strip(),
        flags=re.IGNORECASE,
    ).strip()

    # Qwen's JSON tool-call representation.
    try:
        payload = json.loads(body, object_pairs_hook=_unique_json_object)
        if isinstance(payload, dict) and isinstance(payload.get("name"), str):
            arguments = payload.get(
                "arguments", payload.get("parameters", {})
            )
            if isinstance(arguments, dict):
                return [{
                    "name": payload["name"],
                    "args": arguments,
                }]
    except (ValueError, TypeError):
        pass

    # Also support a function expression inside <tool_call>.
    function = re.fullmatch(
        r"([A-Za-z_]\w*)\s*\((.*)\)",
        body,
        flags=re.DOTALL,
    )
    if function:
        try:
            arguments = _parse_argument_string(function.group(2))
        except ValueError as exc:
            arguments = {"__invalid_tool_call__": str(exc)}
        return [{"name": function.group(1), "args": arguments}]

    # Support XML such as:
    # <file_read>lila.py</file_read>
    # <file_read path="lila.py"/>
    # <file_read><path>lila.py</path></file_read>
    try:
        root = ET.fromstring(f"<root>{body}</root>")
    except ET.ParseError:
        return []

    registry = get_registry()

    for node in root:
        name = node.tag
        args = dict(node.attrib)

        if name == "function":
            name = args.pop("name", "")

        for child in node:
            value = (child.text or "").strip()
            args[child.tag] = value

        direct_text = (node.text or "").strip()
        if direct_text and not args and not list(node):
            tool = registry.get(name)
            if tool and tool.args:
                # Bare XML content maps to the first declared argument.
                args[tool.args[0].name] = direct_text

        if name:
            calls.append({"name": name, "args": args})

    return calls


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Parse legacy, Qwen JSON and XML-style tool calls."""
    calls = []

    # Original Lila format.
    legacy_matches = list(TOOL_CALL_PATTERN.finditer(text))
    for match in legacy_matches:
        try:
            arguments = _parse_argument_string(match.group(2))
        except ValueError as exc:
            arguments = {"__invalid_tool_call__": str(exc)}
        calls.append({"name": match.group(1), "args": arguments})
    legacy_starts = len(re.findall(r"<\|tool_call\|>", text))
    legacy_ends = len(re.findall(r"<\|/tool_call\|>", text))
    if legacy_starts != len(legacy_matches) or legacy_ends != len(legacy_matches):
        calls.append({
            "name": "invalid_tool_call",
            "args": {"__invalid_tool_call__": "Malformed tool call syntax."},
        })

    # Qwen-style <tool_call> blocks.
    xml_matches = list(re.finditer(
        r"<tool_call\b[^>]*>(.*?)</tool_call\s*>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ))
    for match in xml_matches:
        parsed_calls = _xml_tool_calls(match.group(1))
        if parsed_calls:
            calls.extend(parsed_calls)
        else:
            calls.append({
                "name": "invalid_tool_call",
                "args": {"__invalid_tool_call__": "Malformed tool call payload."},
            })
    xml_starts = len(re.findall(r"<tool_call\b", text, flags=re.IGNORECASE))
    xml_ends = len(re.findall(r"</tool_call\s*>", text, flags=re.IGNORECASE))
    if xml_starts != len(xml_matches) or xml_ends != len(xml_matches):
        calls.append({
            "name": "invalid_tool_call",
            "args": {"__invalid_tool_call__": "Malformed tool call syntax."},
        })

    return calls


def format_tool_result(name: str, result: ToolResult) -> str:
    """Format a tool result for injection back into Lila's context."""
    if result.success:
        return f"<|tool_result|>{name}: {result.output}<|/tool_result|>"
    else:
        return f"<|tool_error|>{name}: {result.error}<|/tool_error|>"
