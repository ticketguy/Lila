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
import subprocess
import platform
import time
import json
from typing import Optional
from .tools import Tool, ToolArg, ToolResult, register_tool


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
    ssh_cmd = f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5"
    if key:
        ssh_cmd += f" -i {key}"
    ssh_cmd += f" {user}@{host} '{command}'"
    try:
        r = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return ToolResult(success=r.returncode == 0, output=r.stdout + r.stderr,
                         error=None if r.returncode == 0 else f"exit {r.returncode}")
    except subprocess.TimeoutExpired:
        return ToolResult(success=False, output="", error="SSH timeout")
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


def _wifi_scan(**kw) -> ToolResult:
    """Scan available WiFi networks."""
    if platform.system() == "Windows":
        cmd = "netsh wlan show networks mode=bssid"
    elif platform.system() == "Linux":
        cmd = "nmcli dev wifi list 2>/dev/null || iwlist scan 2>/dev/null"
    else:
        cmd = "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport -s"
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
        return ToolResult(success=True, output=r.stdout)
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


def _wifi_connect(ssid: str, password: str = "", **kw) -> ToolResult:
    """Connect to a WiFi network."""
    if platform.system() == "Windows":
        cmd = f'netsh wlan connect name="{ssid}"'
    elif platform.system() == "Linux":
        cmd = f'nmcli dev wifi connect "{ssid}"'
        if password:
            cmd += f' password "{password}"'
    else:
        cmd = f'networksetup -setairportnetwork en0 "{ssid}" "{password}"'
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return ToolResult(success=r.returncode == 0, output=r.stdout + r.stderr)
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


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
    flag = "-n" if platform.system() == "Windows" else "-c"
    cmd = f"ping {flag} {count} {host}"
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return ToolResult(success=r.returncode == 0, output=r.stdout)
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


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
    parts = [f"i2ctransfer -y {bus}"]
    if write:
        write_bytes = write.strip().split()
        parts.append(f"w{len(write_bytes)}@0x{addr:02x} " + " ".join(f"0x{b}" for b in write_bytes))
    if read_len:
        parts.append(f"r{read_len}@0x{addr:02x}")
    cmd = " ".join(parts)
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return ToolResult(success=r.returncode == 0, output=r.stdout + r.stderr)
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


def _spi_transfer(device: str = "/dev/spidev0.0", data: str = "", speed: int = 1000000, **kw) -> ToolResult:
    """SPI bus transfer. data is hex bytes like 'FF 00 AB'."""
    hex_bytes = data.strip().replace(" ", "\\x")
    cmd = f'echo -ne "\\x{hex_bytes}" > {device}'
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return ToolResult(success=r.returncode == 0, output=r.stdout)
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


def _serial_write(port: str = "/dev/ttyUSB0", data: str = "", baud: int = 9600, **kw) -> ToolResult:
    """Write data to a serial port."""
    try:
        if platform.system() != "Windows":
            os.system(f"stty -F {port} {baud} raw -echo 2>/dev/null")
        with open(port, 'wb') as f:
            f.write(data.encode('utf-8'))
        return ToolResult(success=True, output=f"Wrote {len(data)} bytes to {port} @ {baud}")
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


def _serial_read(port: str = "/dev/ttyUSB0", baud: int = 9600, timeout: int = 2, **kw) -> ToolResult:
    """Read data from a serial port."""
    cmd = f"stty -F {port} {baud} raw -echo && timeout {timeout} cat {port} | xxd -l 256"
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout + 2)
        return ToolResult(success=True, output=r.stdout or "(no data)")
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


def _usb_list(**kw) -> ToolResult:
    """List connected USB devices."""
    if platform.system() == "Windows":
        cmd = 'wmic path Win32_USBControllerDevice get Dependent'
    else:
        cmd = "lsusb 2>/dev/null || ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null"
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return ToolResult(success=True, output=r.stdout)
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  SYSTEM CONTROL TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

def _set_volume(level: int = 50, **kw) -> ToolResult:
    """Set system audio volume (0-100)."""
    if platform.system() == "Windows":
        # Use nircmd or PowerShell
        cmd = f'powershell -c "(New-Object -ComObject WScript.Shell).SendKeys([char]173)"'
    elif platform.system() == "Linux":
        cmd = f"amixer set Master {level}% 2>/dev/null || pactl set-sink-volume @DEFAULT_SINK@ {level}%"
    else:
        cmd = f"osascript -e 'set volume output volume {level}'"
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return ToolResult(success=True, output=f"Volume set to {level}%")
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


def _power_action(action: str = "status", **kw) -> ToolResult:
    """System power: sleep, restart, shutdown, lock."""
    cmds = {}
    if platform.system() == "Windows":
        cmds = {
            "sleep": "rundll32.exe powrprof.dll,SetSuspendState 0,1,0",
            "restart": "shutdown /r /t 5 /c \"Lila restarting system\"",
            "shutdown": "shutdown /s /t 5 /c \"Lila shutting down\"",
            "lock": "rundll32.exe user32.dll,LockWorkStation",
            "status": "systeminfo | findstr /B /C:\"OS\" /C:\"System\" /C:\"Total Physical\"",
        }
    else:
        cmds = {
            "sleep": "systemctl suspend",
            "restart": "systemctl reboot",
            "shutdown": "systemctl poweroff",
            "lock": "loginctl lock-session",
            "status": "uptime && free -h && df -h /",
        }
    
    cmd = cmds.get(action)
    if not cmd:
        return ToolResult(success=False, output="", error=f"Unknown action: {action}. Use: {list(cmds.keys())}")
    
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return ToolResult(success=r.returncode == 0, output=r.stdout + r.stderr)
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


def _notify_user(title: str = "Lila", message: str = "", **kw) -> ToolResult:
    """Show a desktop notification."""
    if platform.system() == "Windows":
        cmd = f'powershell -c "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.MessageBox]::Show(\'{message}\', \'{title}\')"'
    elif platform.system() == "Linux":
        cmd = f'notify-send "{title}" "{message}" 2>/dev/null'
    else:
        cmd = f'osascript -e \'display notification "{message}" with title "{title}"\''
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return ToolResult(success=True, output=f"Notification sent: {title}")
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


def _clipboard_get(**kw) -> ToolResult:
    """Get clipboard contents."""
    if platform.system() == "Windows":
        cmd = "powershell -c Get-Clipboard"
    elif platform.system() == "Linux":
        cmd = "xclip -selection clipboard -o 2>/dev/null || xsel -b 2>/dev/null"
    else:
        cmd = "pbpaste"
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return ToolResult(success=True, output=r.stdout)
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


def _clipboard_set(content: str = "", **kw) -> ToolResult:
    """Set clipboard contents."""
    if platform.system() == "Windows":
        cmd = f'powershell -c "Set-Clipboard -Value \'{content}\'"'
    elif platform.system() == "Linux":
        cmd = f'echo -n "{content}" | xclip -selection clipboard'
    else:
        cmd = f'echo -n "{content}" | pbcopy'
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return ToolResult(success=True, output="Clipboard set")
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


def _screen_capture(output_path: str = "/tmp/lila_screen.png", **kw) -> ToolResult:
    """Capture the current screen."""
    if platform.system() == "Windows":
        cmd = f'powershell -c "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Screen]::PrimaryScreen | ForEach {{ $b = New-Object System.Drawing.Bitmap($_.Bounds.Width, $_.Bounds.Height); $g = [System.Drawing.Graphics]::FromImage($b); $g.CopyFromScreen($_.Bounds.Location, [System.Drawing.Point]::Empty, $_.Bounds.Size); $b.Save(\'{output_path}\') }}"'
    elif platform.system() == "Linux":
        cmd = f"scrot {output_path} 2>/dev/null || import -window root {output_path}"
    else:
        cmd = f"screencapture {output_path}"
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return ToolResult(success=os.path.exists(output_path), output=f"Captured to {output_path}")
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def register_extended_tools():
    """Register all hardware, network, and system control tools."""
    
    # Network
    register_tool(Tool("tcp_connect", "Open TCP connection, send/receive data",
        [ToolArg("host", "string", "Host"), ToolArg("port", "int", "Port"),
         ToolArg("data", "string", "Data to send", required=False, default=""),
         ToolArg("timeout", "int", "Timeout", required=False, default=5)],
        _tcp_connect, "network"))
    
    register_tool(Tool("udp_send", "Send UDP datagram",
        [ToolArg("host", "string", "Host"), ToolArg("port", "int", "Port"),
         ToolArg("data", "string", "Data")],
        _udp_send, "network"))
    
    register_tool(Tool("tcp_listen", "Listen for TCP connection on port",
        [ToolArg("port", "int", "Port"), ToolArg("timeout", "int", "Timeout", required=False, default=10)],
        _tcp_listen, "network"))
    
    register_tool(Tool("ssh_exec", "Execute command on remote host via SSH",
        [ToolArg("host", "string", "Host"), ToolArg("command", "string", "Command"),
         ToolArg("user", "string", "User", required=False, default="root"),
         ToolArg("key", "string", "Key file", required=False, default="")],
        _ssh_exec, "network"))
    
    register_tool(Tool("wifi_scan", "Scan WiFi networks", [], _wifi_scan, "network"))
    register_tool(Tool("wifi_connect", "Connect to WiFi",
        [ToolArg("ssid", "string", "Network name"),
         ToolArg("password", "string", "Password", required=False, default="")],
        _wifi_connect, "network"))
    
    register_tool(Tool("dns_lookup", "Resolve hostname",
        [ToolArg("hostname", "string", "Hostname")], _dns_lookup, "network"))
    register_tool(Tool("ping", "Ping a host",
        [ToolArg("host", "string", "Host"), ToolArg("count", "int", "Count", required=False, default=4)],
        _ping, "network"))
    register_tool(Tool("http_request", "Make HTTP request",
        [ToolArg("url", "string", "URL"), ToolArg("method", "string", "Method", required=False, default="GET"),
         ToolArg("body", "string", "Body", required=False, default="")],
        _http_request, "network"))
    
    # Hardware
    register_tool(Tool("gpio_write", "Set GPIO pin output",
        [ToolArg("pin", "int", "Pin number"), ToolArg("value", "int", "0 or 1")],
        _gpio_write, "hardware"))
    register_tool(Tool("gpio_read", "Read GPIO pin",
        [ToolArg("pin", "int", "Pin number")], _gpio_read, "hardware"))
    register_tool(Tool("i2c_transfer", "I2C bus transfer",
        [ToolArg("bus", "int", "Bus number"), ToolArg("addr", "int", "Device address"),
         ToolArg("write", "string", "Hex bytes to write", required=False, default=""),
         ToolArg("read_len", "int", "Bytes to read", required=False, default=0)],
        _i2c_transfer, "hardware"))
    register_tool(Tool("spi_transfer", "SPI bus transfer",
        [ToolArg("device", "string", "SPI device", required=False, default="/dev/spidev0.0"),
         ToolArg("data", "string", "Hex bytes"), ToolArg("speed", "int", "Clock Hz", required=False, default=1000000)],
        _spi_transfer, "hardware"))
    register_tool(Tool("serial_write", "Write to serial port",
        [ToolArg("port", "string", "Port"), ToolArg("data", "string", "Data"),
         ToolArg("baud", "int", "Baud rate", required=False, default=9600)],
        _serial_write, "hardware"))
    register_tool(Tool("serial_read", "Read from serial port",
        [ToolArg("port", "string", "Port"), ToolArg("baud", "int", "Baud rate", required=False, default=9600)],
        _serial_read, "hardware"))
    register_tool(Tool("usb_list", "List USB devices", [], _usb_list, "hardware"))
    
    # System Control
    register_tool(Tool("set_volume", "Set audio volume",
        [ToolArg("level", "int", "Volume 0-100")], _set_volume, "system"))
    register_tool(Tool("power_action", "System power control",
        [ToolArg("action", "string", "sleep/restart/shutdown/lock/status")],
        _power_action, "system"))
    register_tool(Tool("notify_user", "Show desktop notification",
        [ToolArg("title", "string", "Title"), ToolArg("message", "string", "Message")],
        _notify_user, "system"))
    register_tool(Tool("clipboard_get", "Get clipboard", [], _clipboard_get, "system"))
    register_tool(Tool("clipboard_set", "Set clipboard",
        [ToolArg("content", "string", "Content")], _clipboard_set, "system"))
    register_tool(Tool("screen_capture", "Capture screen",
        [ToolArg("output_path", "string", "Save path", required=False, default="/tmp/lila_screen.png")],
        _screen_capture, "system"))
