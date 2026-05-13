"""
Harness — Lila's Hands

The harness is how Lila acts on the world. When she decides to DO something
(not just respond), she invokes a tool through the harness.

This is NOT a generic tool framework. This is Lila-specific:
- Tools are things Lila needs (file ops, shell, memory, schedule, web)
- Lila decides WHEN to use tools based on her thinking
- No permission prompts — Lila has full autonomy in the household
- Results feed back into her context for the next response
- Tool usage is logged for training (she gets better at using them)

Architecture:
    LilaCore.think() → detects tool intent → Harness.execute() → result → response

Tool invocation format (in Lila's output):
    <|tool_call|>tool_name(arg1="value", arg2="value")<|/tool_call|>

Lila learns to emit these through training. The harness parses and executes them.
"""

from .tools import (
    Tool, ToolResult, ToolRegistry,
    register_tool, execute_tool, get_registry,
    parse_tool_calls, format_tool_result,
)
from .executor import HarnessExecutor
from .system_tools import register_all_system_tools

__all__ = [
    'Tool', 'ToolResult', 'ToolRegistry',
    'register_tool', 'execute_tool', 'get_registry',
    'parse_tool_calls', 'format_tool_result',
    'HarnessExecutor',
    'register_all_system_tools',
]
