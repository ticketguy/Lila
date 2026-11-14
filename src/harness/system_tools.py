"""
System Tools — The actual things Lila can do.

These are Lila's hands. Each tool is a real capability.
No stubs. No mirrors. Actual execution.

Categories:
  filesystem — read, write, list files
  shell      — execute commands
  memory     — store/recall from Memory Fabric namespaces
  schedule   — time-aware operations
  web        — fetch URLs, search
  system     — device control, status queries
  self       — introspection, learning, self-modification
"""

import os
import subprocess
import json
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

from .tools import Tool, ToolArg, ToolResult, register_tool
from .permissions import FileListScopeError, resolve_file_list_scope
from .file_targets import FileTargetError, VerifiedFileHandle


# ═══════════════════════════════════════════════════════════════════════════════
#  FILESYSTEM TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

def _file_read(path: str, *, _verified_file=None, **kwargs) -> ToolResult:
    """Read only from the Harness-verified descriptor; never reopen `path`."""
    if not isinstance(_verified_file, VerifiedFileHandle):
        return ToolResult(success=False, output="", error="A verified file handle is required.")
    try:
        content, truncated, total_chars = _verified_file.read_text(max_chars=50000)
        if truncated:
            content += f"\n... (truncated, {total_chars} total chars)"
        return ToolResult(success=True, output=content)
    except FileTargetError:
        return ToolResult(success=False, output="", error="The approved file identity changed or could not be verified.")
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


def _file_write(path: str, content: str, **kwargs) -> ToolResult:
    """Write content to a file. Creates parent directories."""
    path = os.path.expanduser(path)
    try:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return ToolResult(success=True, output=f"Written {len(content)} bytes to {path}")
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


def _file_append(path: str, content: str, **kwargs) -> ToolResult:
    """Append content to a file."""
    path = os.path.expanduser(path)
    try:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(content)
        return ToolResult(success=True, output=f"Appended {len(content)} bytes to {path}")
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


def _file_list(path: str = ".", pattern: str = "*", **kwargs) -> ToolResult:
    """List files in a directory."""
    try:
        scope = kwargs.get("_authorized_file_list_scope")
        if scope is None:
            _, entries = resolve_file_list_scope(path, pattern)
        else:
            _, entries = scope
        listing = "\n".join(str(entry) for entry in entries[:200])
        return ToolResult(success=True, output=listing)
    except FileListScopeError as exc:
        return ToolResult(success=False, output="", error=str(exc))
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


def _file_delete(path: str, **kwargs) -> ToolResult:
    """Delete a file."""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return ToolResult(success=False, output="", error=f"Not found: {path}")
    try:
        os.remove(path)
        return ToolResult(success=True, output=f"Deleted: {path}")
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  SHELL TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

def _bash(command: str, timeout: int = 30, **kwargs) -> ToolResult:
    """Execute a shell command through M3 BoundedProcessBroker (process-group kill, output cap, env scrub)."""
    from .isolation import BoundedProcessBroker, ProcessExecutionSpec, ResourceClass, ResourceLeaseManager

    broker = BoundedProcessBroker(lease_manager=ResourceLeaseManager())
    spec = ProcessExecutionSpec(
        argv=("bash", "-lc", str(command)),
        cwd=os.path.expanduser("~"),
        execution_backend="trusted-local",
        resource_class=ResourceClass.CPU_HEAVY,
        timeout_seconds=float(max(1, min(int(timeout), 60))),
        max_output_bytes=20000,
        allow_network_egress=True,
    )
    outcome = broker.execute(spec)
    output = outcome.stdout or ""
    if outcome.stderr:
        output = f"{output}\n[stderr] {outcome.stderr}".strip() if output else f"[stderr] {outcome.stderr}"
    if outcome.state in ("timed_out", "cancelled"):
        return ToolResult(success=False, output=output, error=f"{outcome.state}: {outcome.reason_code}")
    if outcome.exit_code not in (0, None):
        output += f"\n[exit code: {outcome.exit_code}]"
    ok = outcome.state == "succeeded" and outcome.exit_code == 0
    return ToolResult(success=ok, output=output, error=None if ok else (outcome.stderr or outcome.reason_code))


# ═══════════════════════════════════════════════════════════════════════════════
#  MEMORY TOOLS (interfaces with Memory Fabric)
# ═══════════════════════════════════════════════════════════════════════════════

MEMORY_DIR = os.path.expanduser("~/.lila/memory/")

def _memory_store(namespace: str, key: str, content: str, **kwargs) -> ToolResult:
    """Store information in a Memory Fabric namespace."""
    valid_ns = ["personal", "episodic", "wiki", "schedule", "contested"]
    if namespace not in valid_ns:
        return ToolResult(success=False, output="",
                         error=f"Invalid namespace. Use: {valid_ns}")
    
    ns_dir = os.path.join(MEMORY_DIR, namespace)
    os.makedirs(ns_dir, exist_ok=True)
    
    entry = {
        "key": key,
        "content": content,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Append to namespace log
    log_path = os.path.join(ns_dir, "entries.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    
    return ToolResult(success=True,
                     output=f"Stored in {namespace}/{key}: {content[:100]}...")


def _memory_recall(query: str, namespace: str = "all", **kwargs) -> ToolResult:
    """Recall information from Memory Fabric. Simple keyword search."""
    results = []
    
    namespaces = ["personal", "episodic", "wiki", "schedule", "contested"]
    if namespace != "all" and namespace in namespaces:
        namespaces = [namespace]
    
    query_lower = query.lower()
    
    for ns in namespaces:
        log_path = os.path.join(MEMORY_DIR, ns, "entries.jsonl")
        if not os.path.exists(log_path):
            continue
        with open(log_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if (query_lower in entry.get("key", "").lower() or
                        query_lower in entry.get("content", "").lower()):
                        results.append(f"[{ns}] {entry['key']}: {entry['content']}")
                except json.JSONDecodeError:
                    continue
    
    if not results:
        return ToolResult(success=True, output="No memories found matching query.")
    
    return ToolResult(success=True, output="\n".join(results[:20]))


def _memory_list(namespace: str = "all", **kwargs) -> ToolResult:
    """List all stored memories in a namespace."""
    namespaces = ["personal", "episodic", "wiki", "schedule", "contested"]
    if namespace != "all" and namespace in namespaces:
        namespaces = [namespace]
    
    output_parts = []
    for ns in namespaces:
        log_path = os.path.join(MEMORY_DIR, ns, "entries.jsonl")
        if not os.path.exists(log_path):
            continue
        count = 0
        with open(log_path, 'r') as f:
            for line in f:
                count += 1
        output_parts.append(f"{ns}: {count} entries")
    
    return ToolResult(success=True, output="\n".join(output_parts) or "Memory is empty.")


# ═══════════════════════════════════════════════════════════════════════════════
#  SCHEDULE TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

SCHEDULE_FILE = os.path.expanduser("~/.lila/schedule.json")

def _schedule_add(event: str, time_str: str, **kwargs) -> ToolResult:
    """Add an event to the schedule."""
    schedule = []
    if os.path.exists(SCHEDULE_FILE):
        with open(SCHEDULE_FILE, 'r', encoding='utf-8') as f:
            schedule = json.load(f)
    
    entry = {
        "event": event,
        "time": time_str,
        "created": datetime.now().isoformat(),
        "done": False,
    }
    schedule.append(entry)
    
    os.makedirs(os.path.dirname(SCHEDULE_FILE), exist_ok=True)
    with open(SCHEDULE_FILE, 'w', encoding='utf-8') as f:
        json.dump(schedule, f, indent=2, ensure_ascii=False)
    
    return ToolResult(success=True, output=f"Scheduled: {event} at {time_str}")


def _schedule_list(**kwargs) -> ToolResult:
    """List upcoming schedule entries."""
    if not os.path.exists(SCHEDULE_FILE):
        return ToolResult(success=True, output="Schedule is empty.")
    
    with open(SCHEDULE_FILE, 'r', encoding='utf-8') as f:
        schedule = json.load(f)
    
    active = [e for e in schedule if not e.get("done")]
    if not active:
        return ToolResult(success=True, output="No upcoming events.")
    
    lines = [f"• {e['event']} — {e['time']}" for e in active[:20]]
    return ToolResult(success=True, output="\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════════
#  WEB TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

def _web_fetch(url: str, **kwargs) -> ToolResult:
    """Fetch a URL and return its content."""
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={'User-Agent': 'Lila/1.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read().decode('utf-8', errors='replace')
        if len(content) > 30000:
            content = content[:30000] + "\n... (truncated)"
        return ToolResult(success=True, output=content)
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  SELF TOOLS (introspection)
# ═══════════════════════════════════════════════════════════════════════════════

def _current_time(**kwargs) -> ToolResult:
    """Return the current local date and time."""
    now = datetime.now()
    return ToolResult(
        success=True,
        output=now.strftime("%Y-%m-%d %H:%M:%S (%A)"),
        data={"iso": now.isoformat(timespec="seconds")},
    )


def _self_status(**kwargs) -> ToolResult:
    """Report Lila's current status."""
    import platform
    status = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "memory_dir": MEMORY_DIR,
        "schedule_file": SCHEDULE_FILE,
    }
    return ToolResult(success=True, output=json.dumps(status, indent=2, ensure_ascii=False))


def _self_log(message: str, level: str = "info", **kwargs) -> ToolResult:
    """Write to Lila's internal log."""
    log_dir = os.path.expanduser("~/.lila/logs/")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "lila.log")
    
    entry = f"[{datetime.now().isoformat()}] [{level.upper()}] {message}\n"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(entry)
    
    return ToolResult(success=True, output=f"Logged: {message}")


# ═══════════════════════════════════════════════════════════════════════════════
#  REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def register_all_system_tools(registry=None):
    """Register built-ins globally by default, or only into an explicit registry."""
    if registry is None:
        register = register_tool
    else:
        register = getattr(registry, "register", None)
        if not callable(register):
            raise TypeError("registry must provide a callable register(tool) method")
    
    # Filesystem
    register(Tool(
        name="file_read", description="Read a file's contents",
        args=[ToolArg("path", "path", "File path to read")],
        execute=_file_read, category="filesystem"
    ))
    register(Tool(
        name="file_write", description="Write content to a file (creates dirs)",
        args=[ToolArg("path", "path", "File path"), ToolArg("content", "string", "Content to write")],
        execute=_file_write, category="filesystem"
    ))
    register(Tool(
        name="file_append", description="Append content to a file",
        args=[ToolArg("path", "path", "File path"), ToolArg("content", "string", "Content to append")],
        execute=_file_append, category="filesystem"
    ))
    register(Tool(
        name="file_list", description="List files in a directory",
        args=[ToolArg("path", "path", "Directory path", default="."),
              ToolArg("pattern", "string", "Glob pattern", required=False, default="*")],
        execute=_file_list, category="filesystem"
    ))
    register(Tool(
        name="file_delete", description="Delete a file",
        args=[ToolArg("path", "path", "File to delete")],
        execute=_file_delete, category="filesystem"
    ))
    
    # Shell
    register(Tool(
        name="bash", description="Execute a shell command (full access)",
        args=[ToolArg("command", "string", "Command to execute"),
              ToolArg("timeout", "int", "Timeout in seconds", required=False,
                      default=30, minimum=1, maximum=300)],
        execute=_bash, category="shell"
    ))
    
    # Memory
    register(Tool(
        name="memory_store", description="Store information in Memory Fabric",
        args=[ToolArg("namespace", "string", "Namespace: personal/episodic/wiki/schedule/contested"),
              ToolArg("key", "string", "Short key/label"),
              ToolArg("content", "string", "Content to store")],
        execute=_memory_store, category="memory"
    ))
    register(Tool(
        name="memory_recall", description="Search Memory Fabric for information",
        args=[ToolArg("query", "string", "Search query"),
              ToolArg("namespace", "string", "Namespace to search (or 'all')", required=False, default="all")],
        execute=_memory_recall, category="memory"
    ))
    register(Tool(
        name="memory_list", description="List stored memories by namespace",
        args=[ToolArg("namespace", "string", "Namespace (or 'all')", required=False, default="all")],
        execute=_memory_list, category="memory"
    ))
    
    # Schedule
    register(Tool(
        name="schedule_add", description="Add an event/reminder to the schedule",
        args=[ToolArg("event", "string", "Event description"),
              ToolArg("time_str", "string", "When (e.g. 'tomorrow 9am', '2024-03-15 14:00')")],
        execute=_schedule_add, category="schedule"
    ))
    register(Tool(
        name="schedule_list", description="List upcoming scheduled events",
        args=[], execute=_schedule_list, category="schedule"
    ))
    
    # Web
    register(Tool(
        name="web_fetch", description="Fetch a URL and return content",
        args=[ToolArg("url", "string", "URL to fetch")],
        execute=_web_fetch, category="web"
    ))
    
    # Self
    register(Tool(
        name="current_time", description="Get the current local date and time",
        args=[],
        execute=_current_time, category="system"
    ))
    register(Tool(
        name="self_status", description="Report current system status",
        args=[], execute=_self_status, category="self"
    ))
    register(Tool(
        name="self_log", description="Write to internal log",
        args=[ToolArg("message", "string", "Log message"),
              ToolArg("level", "string", "Level: info/warn/error", required=False, default="info")],
        execute=_self_log, category="self"
    ))
