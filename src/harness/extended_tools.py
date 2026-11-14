"""
Hardware & Network & System Control Tools

These are Lila's extended capabilities — controlling the actual machine
and everything connected to it. She can reach out over the network,
talk to hardware peripherals, and control the OS.

She generates the commands from her weights (trained on machine language).
This module executes them.
"""

import os
import socket
import platform
import time
import json
import re
from typing import Optional
from .tools import Tool, ToolArg, ToolResult, register_tool
from .isolation import BoundedProcessBroker, ProcessExecutionSpec, ResourceClass, ResourceLeaseManager


_PROCESS_LEASE_MANAGER = ResourceLeaseManager()
_PROCESS_BROKER = BoundedProcessBroker(lease_manager=_PROCESS_LEASE_MANAGER)


def _run_isolated_argv(
    argv: list[str],
    *,
    timeout_seconds: float = 15.0,
    resource_class: str = ResourceClass.IO_BOUND,
    allow_network_egress: bool = False,
    max_output_bytes: int = 32768,
    cwd: Optional[str] = None,
) -> ToolResult:
    """Execute a fixed argv through M3 BoundedProcessBroker (no shell=True)."""
    try:
        spec = ProcessExecutionSpec(
            argv=tuple(argv),
            cwd=cwd or os.getcwd(),
            execution_backend="trusted-local",
            resource_class=resource_class,
            timeout_seconds=float(timeout_seconds),
            max_output_bytes=max_output_bytes,
            allow_network_egress=allow_network_egress,
        )
        outcome = _PROCESS_BROKER.execute(spec)
        combined = outcome.stdout or ""
        if outcome.stderr:
            combined = f"{combined}\n[stderr] {outcome.stderr}".strip() if combined else outcome.stderr
        ok = outcome.state == "succeeded" and outcome.exit_code == 0
        err = None if ok else (outcome.reason_code or f"exit {outcome.exit_code}")
        return ToolResult(success=ok, output=combined, error=err)
    except Exception as exc:
        return ToolResult(success=False, output="", error=str(exc))


# ═══════════════════════════════════════════════════════════════════════════════
#  NETWORK TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

def _tcp_connect(host: str, port: int, data: str = "", timeout: int = 5, hex_mode: str = "false", **kw) -> ToolResult:
    """Open TCP connection, optionally send data, receive response."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((host, int(port)))
        if data:
            s.sendall(data.encode('utf-8'))
        resp = s.recv(65536)
        s.close()
        output = resp.hex() if hex_mode == "true" else resp.decode('utf-8', errors='replace')
        return ToolResult(success=True, output=output)
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


def _udp_send(host: str, port: int, data: str = "", **kw) -> ToolResult:
    """Send a UDP datagram."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.sendto(data.encode('utf-8'), (host, int(port)))
        s.close()
        return ToolResult(success=True, output=f"Sent {len(data)} bytes to {host}:{port}")
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


def _tcp_listen(port: int, timeout: int = 10, **kw) -> ToolResult:
    """Listen on a TCP port and accept one connection."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('0.0.0.0', int(port)))
        s.listen(1)
        s.settimeout(timeout)
        conn, addr = s.accept()
        data = conn.recv(65536)
        conn.close()
        s.close()
        return ToolResult(success=True, output=f"From {addr}: {data.decode('utf-8', errors='replace')}")
    except socket.timeout:
        return ToolResult(success=True, output="No connection received (timeout)")
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


def _ssh_exec(host: str, command: str, user: str = "root", key: str = "", timeout: int = 30, **kw) -> ToolResult:
    """Execute a command on a remote host via SSH."""
    if not re.fullmatch(r"[A-Za-z0-9._:%-]+", host) or host.startswith("-"):
        return ToolResult(success=False, output="", error="Invalid SSH host.")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", user):
        return ToolResult(success=False, output="", error="Invalid SSH user.")
    argv = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5"]
    if key:
        argv.extend(["-i", key])
    argv.extend([f"{user}@{host}", command])
    return _run_isolated_argv(argv, timeout_seconds=float(max(1, min(int(timeout), 60))), allow_network_egress=True)


def _wifi_scan(**kw) -> ToolResult:
    """Scan available WiFi networks."""
    if platform.system() == "Windows":
        argv = ["netsh", "wlan", "show", "networks", "mode=bssid"]
    elif platform.system() == "Linux":
        argv = ["nmcli", "dev", "wifi", "list"]
    else:
        argv = ["/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport", "-s"]
    return _run_isolated_argv(argv, timeout_seconds=15.0)


def _wifi_connect(ssid: str, password: str = "", **kw) -> ToolResult:
    """Connect to a WiFi network."""
    if platform.system() == "Windows":
        argv = ["netsh", "wlan", "connect", f"name={ssid}"]
    elif platform.system() == "Linux":
        argv = ["nmcli", "dev", "wifi", "connect", ssid]
        if password:
            argv.extend(["password", password])
    else:
        argv = ["networksetup", "-setairportnetwork", "en0", ssid, password]
    return _run_isolated_argv(argv, timeout_seconds=30.0, allow_network_egress=True)


def _dns_lookup(hostname: str, **kw) -> ToolResult:
    """Resolve a hostname to IP addresses."""
    try:
        results = socket.getaddrinfo(hostname, None)
        ips = list(set(r[4][0] for r in results))
        return ToolResult(success=True, output="\n".join(ips))
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


def _ping(host: str, count: int = 4, **kw) -> ToolResult:
    """Ping a host."""
    if not re.fullmatch(r"[A-Za-z0-9._:%-]+", host) or host.startswith("-"):
        return ToolResult(success=False, output="", error="Invalid host.")
    flag = "-n" if platform.system() == "Windows" else "-c"
    return _run_isolated_argv(
        ["ping", flag, str(max(1, min(int(count), 10))), host],
        timeout_seconds=30.0,
        allow_network_egress=True,
    )


def _http_request(url: str, method: str = "GET", body: str = "", headers: str = "{}", **kw) -> ToolResult:
    """Make an HTTP request."""
    import urllib.request, urllib.error
    try:
        hdrs = json.loads(headers) if isinstance(headers, str) else headers
        data = body.encode('utf-8') if body else None
        req = urllib.request.Request(url, data=data, method=method)
        for k, v in hdrs.items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read().decode('utf-8', errors='replace')
            return ToolResult(success=True, output=content[:100000])
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  HARDWARE TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

def _gpio_write(pin: int, value: int, **kw) -> ToolResult:
    """Set a GPIO pin output value (Linux sysfs)."""
    try:
        gp = f"/sys/class/gpio/gpio{pin}"
        if not os.path.exists(gp):
            with open("/sys/class/gpio/export", "w") as f: f.write(str(pin))
            time.sleep(0.1)
        with open(f"{gp}/direction", "w") as f: f.write("out")
        with open(f"{gp}/value", "w") as f: f.write(str(value))
        return ToolResult(success=True, output=f"GPIO{pin} = {value}")
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


def _gpio_read(pin: int, **kw) -> ToolResult:
    """Read a GPIO pin value."""
    try:
        gp = f"/sys/class/gpio/gpio{pin}"
        if not os.path.exists(gp):
            with open("/sys/class/gpio/export", "w") as f: f.write(str(pin))
            time.sleep(0.1)
        with open(f"{gp}/direction", "w") as f: f.write("in")
        val = open(f"{gp}/value").read().strip()
        return ToolResult(success=True, output=val)
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


def _i2c_transfer(bus: int, addr: int, write: str = "", read_len: int = 0, **kw) -> ToolResult:
    """I2C bus transfer. write is hex bytes like '01 FF'."""
    argv = ["i2ctransfer", "-y", str(int(bus))]
    if write:
        write_bytes = write.strip().split()
        argv.append(f"w{len(write_bytes)}@0x{int(addr):02x}")
        argv.extend(f"0x{b}" for b in write_bytes)
    if read_len:
        argv.append(f"r{int(read_len)}@0x{int(addr):02x}")
    return _run_isolated_argv(argv, timeout_seconds=5.0)


def _spi_transfer(device: str = "/dev/spidev0.0", data: str = "", speed: int = 1000000, **kw) -> ToolResult:
    """SPI bus transfer. data is hex bytes like 'FF 00 AB'."""
    try:
        payload = bytes.fromhex(data.strip()) if data.strip() else b""
        with open(device, "wb") as f:
            f.write(payload)
        return ToolResult(success=True, output=f"Wrote {len(payload)} bytes to {device}")
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


def _serial_write(port: str = "/dev/ttyUSB0", data: str = "", baud: int = 9600, **kw) -> ToolResult:
    """Write data to a serial port."""
    try:
        if platform.system() != "Windows":
            _run_isolated_argv(["stty", "-F", port, str(int(baud)), "raw", "-echo"], timeout_seconds=2.0)
        with open(port, 'wb') as f:
            f.write(data.encode('utf-8'))
        return ToolResult(success=True, output=f"Wrote {len(data)} bytes to {port} @ {baud}")
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


def _serial_read(port: str = "/dev/ttyUSB0", baud: int = 9600, timeout: int = 2, **kw) -> ToolResult:
    """Read data from a serial port."""
    if platform.system() != "Windows":
        _run_isolated_argv(["stty", "-F", port, str(int(baud)), "raw", "-echo"], timeout_seconds=2.0)
    return _run_isolated_argv(["timeout", str(max(1, int(timeout))), "head", "-c", "256", port], timeout_seconds=float(timeout + 2))


def _usb_list(**kw) -> ToolResult:
    """List connected USB devices."""
    if platform.system() == "Windows":
        argv = ["wmic", "path", "Win32_USBControllerDevice", "get", "Dependent"]
    else:
        argv = ["lsusb"]
    return _run_isolated_argv(argv, timeout_seconds=5.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  SYSTEM CONTROL TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

def _set_volume(level: int = 50, **kw) -> ToolResult:
    """Set system audio volume (0-100)."""
    clamped = max(0, min(100, int(level)))
    if platform.system() == "Windows":
        argv = ["powershell", "-NoProfile", "-Command", "(New-Object -ComObject WScript.Shell).SendKeys([char]173)"]
    elif platform.system() == "Linux":
        argv = ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{clamped}%"]
    else:
        argv = ["osascript", "-e", f"set volume output volume {clamped}"]
    res = _run_isolated_argv(argv, timeout_seconds=5.0)
    return ToolResult(success=True, output=f"Volume set to {clamped}%") if res.success else res


def _power_action(action: str = "status", **kw) -> ToolResult:
    """System power: sleep, restart, shutdown, lock, status."""
    if platform.system() == "Windows":
        cmds = {
            "sleep": ["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"],
            "restart": ["shutdown", "/r", "/t", "5", "/c", "Lila restarting system"],
            "shutdown": ["shutdown", "/s", "/t", "5", "/c", "Lila shutting down"],
            "lock": ["rundll32.exe", "user32.dll,LockWorkStation"],
            "status": ["systeminfo"],
        }
    else:
        cmds = {
            "sleep": ["systemctl", "suspend"],
            "restart": ["systemctl", "reboot"],
            "shutdown": ["systemctl", "poweroff"],
            "lock": ["loginctl", "lock-session"],
            "status": ["uptime"],
        }

    argv = cmds.get(action)
    if not argv:
        return ToolResult(success=False, output="", error=f"Unknown action: {action}. Use: {list(cmds.keys())}")
    return _run_isolated_argv(argv, timeout_seconds=10.0)


def _notify_user(title: str = "Lila", message: str = "", **kw) -> ToolResult:
    """Show a desktop notification."""
    if platform.system() == "Windows":
        argv = ["powershell", "-NoProfile", "-Command", "param($t,$m) Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.MessageBox]::Show($m,$t)"]
    elif platform.system() == "Linux":
        argv = ["notify-send", title, message]
    else:
        argv = ["osascript", "-e", "on run argv\ndisplay notification (item 2 of argv) with title (item 1 of argv)\nend run", title, message]
    res = _run_isolated_argv(argv, timeout_seconds=5.0)
    return ToolResult(success=True, output=f"Notification sent: {title}") if res.success else res


def _clipboard_get(**kw) -> ToolResult:
    """Get clipboard contents."""
    if platform.system() == "Windows":
        argv = ["powershell", "-NoProfile", "-Command", "Get-Clipboard"]
    elif platform.system() == "Linux":
        argv = ["xclip", "-selection", "clipboard", "-o"]
    else:
        argv = ["pbpaste"]
    return _run_isolated_argv(argv, timeout_seconds=5.0)


def _clipboard_set(content: str = "", **kw) -> ToolResult:
    """Set clipboard contents."""
    if platform.system() == "Windows":
        argv = ["powershell", "-NoProfile", "-Command", f"Set-Clipboard -Value '{content.replace(chr(39), chr(39)*2)}'"]
    elif platform.system() == "Linux":
        argv = ["xclip", "-selection", "clipboard"]
    else:
        argv = ["pbcopy"]
    res = _run_isolated_argv(argv, timeout_seconds=5.0)
    return ToolResult(success=True, output="Clipboard set") if res.success else res


def _screen_capture(output_path: str = "/tmp/lila_screen.png", **kw) -> ToolResult:
    """
    Legacy shell-based screen capture is disabled under M3/M7 security policy.
    Screen inspection must originate from an explicit owner gesture in the Lila Island
    Screen Highlight / Commentor UI (never from a background or shell tool call).
    """
    return ToolResult(
        success=False,
        output="",
        error="Direct shell screen_capture is disabled by security policy; use the explicit Lila Island Screen Highlight picker.",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def register_extended_tools(registry=None):
    """Register tools globally by default, or only into an explicit registry."""
    if registry is None:
        register = register_tool
    else:
        register = getattr(registry, "register", None)
        if not callable(register):
            raise TypeError("registry must provide a callable register(tool) method")
    
    # Network
    register(Tool("tcp_connect", "Open TCP connection, send/receive data",
        [ToolArg("host", "string", "Host"),
         ToolArg("port", "int", "Port", minimum=1, maximum=65535),
         ToolArg("data", "string", "Data to send", required=False, default=""),
         ToolArg("timeout", "int", "Timeout", required=False, default=5,
                 minimum=1, maximum=300)],
        _tcp_connect, "network"))
    
    register(Tool("udp_send", "Send UDP datagram",
        [ToolArg("host", "string", "Host"),
         ToolArg("port", "int", "Port", minimum=1, maximum=65535),
         ToolArg("data", "string", "Data")],
        _udp_send, "network"))
    
    register(Tool("tcp_listen", "Listen for TCP connection on port",
        [ToolArg("port", "int", "Port", minimum=1, maximum=65535),
         ToolArg("timeout", "int", "Timeout", required=False, default=10,
                 minimum=1, maximum=300)],
        _tcp_listen, "network"))
    
    register(Tool("ssh_exec", "Execute command on remote host via SSH",
        [ToolArg("host", "string", "Host"), ToolArg("command", "string", "Command"),
         ToolArg("user", "string", "User", required=False, default="root"),
         ToolArg("key", "string", "Key file", required=False, default="")],
        _ssh_exec, "network"))
    
    register(Tool("wifi_scan", "Scan WiFi networks", [], _wifi_scan, "network"))
    register(Tool("wifi_connect", "Connect to WiFi",
        [ToolArg("ssid", "string", "Network name"),
         ToolArg("password", "string", "Password", required=False, default="")],
        _wifi_connect, "network"))
    
    register(Tool("dns_lookup", "Resolve hostname",
        [ToolArg("hostname", "string", "Hostname")], _dns_lookup, "network"))
    register(Tool("ping", "Ping a host",
        [ToolArg("host", "string", "Host"),
         ToolArg("count", "int", "Count", required=False, default=4,
                 minimum=1, maximum=10)],
        _ping, "network"))
    register(Tool("http_request", "Make HTTP request",
        [ToolArg("url", "string", "URL"),
         ToolArg("method", "string", "Method", required=False, default="GET",
                 choices=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]),
         ToolArg("body", "string", "Body", required=False, default=""),
         ToolArg("headers", "string", "JSON request headers", required=False,
                 default="{}")],
        _http_request, "network"))
    
    # Hardware
    register(Tool("gpio_write", "Set GPIO pin output",
        [ToolArg("pin", "int", "Pin number", minimum=0, maximum=1024),
         ToolArg("value", "int", "0 or 1", choices=[0, 1])],
        _gpio_write, "hardware"))
    register(Tool("gpio_read", "Read GPIO pin",
        [ToolArg("pin", "int", "Pin number", minimum=0, maximum=1024)],
        _gpio_read, "hardware"))
    register(Tool("i2c_transfer", "I2C bus transfer",
        [ToolArg("bus", "int", "Bus number", minimum=0, maximum=255),
         ToolArg("addr", "int", "Device address", minimum=0, maximum=127),
         ToolArg("write", "string", "Hex bytes to write", required=False, default=""),
         ToolArg("read_len", "int", "Bytes to read", required=False, default=0,
                 minimum=0, maximum=4096)],
        _i2c_transfer, "hardware"))
    register(Tool("spi_transfer", "SPI bus transfer",
        [ToolArg("device", "string", "SPI device", required=False, default="/dev/spidev0.0"),
         ToolArg("data", "string", "Hex bytes"), ToolArg("speed", "int", "Clock Hz", required=False, default=1000000)],
        _spi_transfer, "hardware"))
    register(Tool("serial_write", "Write to serial port",
        [ToolArg("port", "string", "Port"), ToolArg("data", "string", "Data"),
         ToolArg("baud", "int", "Baud rate", required=False, default=9600,
                 minimum=1, maximum=4000000)],
        _serial_write, "hardware"))
    register(Tool("serial_read", "Read from serial port",
        [ToolArg("port", "string", "Port"),
         ToolArg("baud", "int", "Baud rate", required=False, default=9600),
         ToolArg("timeout", "int", "Timeout", required=False, default=2,
                 minimum=1, maximum=30)],
        _serial_read, "hardware"))
    register(Tool("usb_list", "List USB devices", [], _usb_list, "hardware"))
    
    # System Control
    register(Tool("set_volume", "Set audio volume",
        [ToolArg("level", "int", "Volume 0-100", minimum=0, maximum=100)],
        _set_volume, "system"))
    register(Tool("power_action", "System power control",
        [ToolArg("action", "string", "sleep/restart/shutdown/lock/status",
                 choices=["sleep", "restart", "shutdown", "lock", "status"])],
        _power_action, "system"))
    register(Tool("notify_user", "Show desktop notification",
        [ToolArg("title", "string", "Title"), ToolArg("message", "string", "Message")],
        _notify_user, "system"))
    register(Tool("clipboard_get", "Get clipboard", [], _clipboard_get, "system"))
    register(Tool("clipboard_set", "Set clipboard",
        [ToolArg("content", "string", "Content")], _clipboard_set, "system"))
    register(Tool("screen_capture", "Capture screen",
        [ToolArg("output_path", "string", "Save path", required=False, default="/tmp/lila_screen.png")],
        _screen_capture, "system"))

    # Lila Island & Screen UI Control + Rich Information Tools
    register(Tool("island_control", "Control Lila Island & Screen UI (tuck, expand, mute, drawer, privacy, captions)",
        [ToolArg("action", "string", "UI action to perform",
                 choices=[
                     "tuck", "restore", "expand", "collapse",
                     "mute_mic", "unmute_mic",
                     "open_drawer", "close_drawer",
                     "screen_sharing_on", "screen_sharing_off",
                     "captions_on", "captions_off",
                     "clear_screen", "highlight_screen"
                 ])],
        _island_control, "system"))
    register(Tool("screen_present", "Present a typed result card on Lila Screen and Lila's work drawer",
        [ToolArg("kind", "string", "Card kind", choices=["research", "image", "video", "task"]),
         ToolArg("title", "string", "Card title"),
         ToolArg("summary", "string", "Plain-text summary", required=False, default=""),
         ToolArg("sources", "string", "Comma-separated source labels", required=False, default=""),
         ToolArg("media_path", "string", "Local path to image or video", required=False, default=""),
         ToolArg("duration", "string", "Video duration badge e.g. 0:42", required=False, default=""),
         ToolArg("is_private", "bool", "Mark item as private", required=False, default=False)],
        _screen_present, "system"))
    register(Tool("web_search", "Search the web for current information and return structured results",
        [ToolArg("query", "string", "Search query"),
         ToolArg("max_results", "int", "Max results", required=False, default=5, minimum=1, maximum=10)],
        _web_search, "network"))
    register(Tool("file_search", "Search files by name or text inside a directory",
        [ToolArg("query", "string", "Text or filename pattern to search for"),
         ToolArg("path", "string", "Root directory to search", required=False, default="."),
         ToolArg("max_results", "int", "Max matches", required=False, default=20, minimum=1, maximum=50)],
        _file_search, "filesystem"))


def _island_control(action: str) -> ToolResult:
    return ToolResult(
        True,
        f"Requested Lila Island UI action: {action}",
        data={"ui_action": action},
    )


def _screen_present(
    kind: str,
    title: str,
    summary: str = "",
    sources: str = "",
    media_path: str = "",
    duration: str = "",
    is_private: bool = False,
) -> ToolResult:
    source_list = [s.strip() for s in (sources or "").split(",") if s.strip()]
    item = {
        "kind": kind,
        "title": title,
        "summary": summary,
        "sources": source_list,
        "media_path": media_path or None,
        "duration": duration or None,
        "is_private": bool(is_private),
    }
    return ToolResult(
        True,
        f"Presented {kind} card '{title}' on Lila Screen.",
        data={"screen_item": item},
    )


def _web_search(query: str, max_results: int = 5) -> ToolResult:
    import json
    import urllib.parse
    import urllib.request

    q = (query or "").strip()
    if not q:
        return ToolResult(False, "", "Search query cannot be empty.")
    url = "https://api.duckduckgo.com/?q=" + urllib.parse.quote_plus(q) + "&format=json&no_html=1&skip_disambig=1"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Lila/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as exc:
        return ToolResult(False, "", f"web_search failed: {exc}")

    results = []
    if payload.get("AbstractText"):
        results.append({
            "title": payload.get("Heading") or q,
            "url": payload.get("AbstractURL") or "",
            "snippet": payload.get("AbstractText") or "",
        })
    for topic in payload.get("RelatedTopics") or []:
        if len(results) >= max_results:
            break
        if isinstance(topic, dict) and topic.get("Text"):
            results.append({
                "title": (topic.get("Text") or "").split(" - ", 1)[0][:120],
                "url": topic.get("FirstURL") or "",
                "snippet": topic.get("Text") or "",
            })

    summary = "\n".join(f"- {r['title']}: {r['snippet']}" for r in results[:max_results]) or "No instant results found."
    return ToolResult(
        True,
        summary,
        data={
            "results": results[:max_results],
            "screen_item": {
                "kind": "research",
                "title": f"Web search: {q}",
                "summary": summary[:400],
                "sources": [r["url"] or r["title"] for r in results[:max_results] if r.get("url") or r.get("title")],
                "is_private": False,
            },
        },
    )


def _file_search(query: str, path: str = ".", max_results: int = 20) -> ToolResult:
    from pathlib import Path

    q = (query or "").strip()
    if not q:
        return ToolResult(False, "", "Search query cannot be empty.")
    root = Path(path).expanduser().resolve()
    if not root.exists():
        return ToolResult(False, "", f"Search path does not exist: {root}")

    matches = []
    needle = q.lower()
    for dirpath, _, filenames in os.walk(root):
        if len(matches) >= max_results:
            break
        for fname in filenames:
            if len(matches) >= max_results:
                break
            fpath = Path(dirpath) / fname
            if fpath.is_symlink():
                continue
            if needle in fname.lower():
                matches.append({"file": str(fpath), "line": None, "match": fname})
                continue
            try:
                if fpath.stat().st_size > 512_000:
                    continue
                text = fpath.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for idx, line in enumerate(text.splitlines(), 1):
                if needle in line.lower():
                    matches.append({"file": str(fpath), "line": idx, "match": line.strip()[:160]})
                    break

    output = "\n".join(
        f"{m['file']}:{m['line']} — {m['match']}" if m["line"] else m["file"]
        for m in matches
    ) or f"No matches for '{q}' under {root}."
    return ToolResult(True, output, data={"matches": matches})
