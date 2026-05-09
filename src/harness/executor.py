"""
Harness Executor — Runs tool calls and feeds results back.

This is the bridge between Lila's output (which may contain tool calls)
and the actual execution of those tools. It:
  1. Parses tool calls from Lila's generated text
  2. Executes them in sequence
  3. Injects results back into context
  4. Lets Lila continue generating with the results

Lila can chain multiple tool calls in a single response.
She has full autonomy — no human approval needed.
"""

from __future__ import annotations
import time
import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .tools import (
    parse_tool_calls, execute_tool, format_tool_result,
    ToolResult, get_registry,
)


@dataclass
class ExecutionRecord:
    """Record of a single tool execution (for training data)."""
    timestamp: str
    tool_name: str
    args: Dict[str, Any]
    success: bool
    output: str
    error: Optional[str]
    duration_ms: float


class HarnessExecutor:
    """
    Executes tool calls from Lila's output.
    
    Usage:
        executor = HarnessExecutor()
        
        # After Lila generates text:
        response_text = lila.think("what time is it?")
        
        # Check for tool calls and execute them:
        results = executor.process(response_text)
        
        # If tools were called, feed results back:
        if results:
            augmented = executor.augment_context(response_text, results)
            # Lila can now continue with the results
    """
    
    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = log_dir or os.path.expanduser("~/.lila/harness_logs/")
        self.history: List[ExecutionRecord] = []
        os.makedirs(self.log_dir, exist_ok=True)
    
    def process(self, lila_output: str) -> List[Dict[str, Any]]:
        """
        Parse and execute any tool calls in Lila's output.
        
        Returns list of execution results:
            [{"name": "tool", "result": ToolResult, "record": ExecutionRecord}]
        """
        calls = parse_tool_calls(lila_output)
        if not calls:
            return []
        
        results = []
        for call in calls:
            name = call["name"]
            args = call["args"]
            
            # Execute
            start = time.time()
            result = execute_tool(name, **args)
            duration = (time.time() - start) * 1000
            
            # Record
            record = ExecutionRecord(
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                tool_name=name,
                args=args,
                success=result.success,
                output=result.output[:1000],  # Truncate for logging
                error=result.error,
                duration_ms=duration,
            )
            self.history.append(record)
            self._log(record)
            
            results.append({
                "name": name,
                "result": result,
                "record": record,
            })
        
        return results
    
    def augment_context(self, original_output: str, results: List[Dict]) -> str:
        """
        Augment Lila's output with tool results.
        
        Strips the tool_call markers and appends results so Lila
        can incorporate them in her next generation pass.
        """
        context_parts = [original_output]
        
        for r in results:
            formatted = format_tool_result(r["name"], r["result"])
            context_parts.append(formatted)
        
        return "\n".join(context_parts)
    
    def has_pending_tools(self, text: str) -> bool:
        """Check if text contains unexecuted tool calls."""
        return bool(parse_tool_calls(text))
    
    def get_tool_prompt(self) -> str:
        """Get the tool description block to include in Lila's system prompt."""
        return get_registry().prompt_block()
    
    def recent_history(self, n: int = 10) -> List[ExecutionRecord]:
        """Get recent execution history."""
        return self.history[-n:]
    
    def _log(self, record: ExecutionRecord):
        """Log execution for training data collection."""
        log_file = os.path.join(self.log_dir, "executions.jsonl")
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps({
                    "timestamp": record.timestamp,
                    "tool": record.tool_name,
                    "args": record.args,
                    "success": record.success,
                    "output": record.output,
                    "error": record.error,
                    "duration_ms": record.duration_ms,
                }) + "\n")
        except OSError:
            pass  # Don't crash if logging fails
