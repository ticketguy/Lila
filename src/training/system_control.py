"""
System Control Training Corpus

Training data that teaches Lila to:
1. Take full control of any system she runs on
2. Write code to achieve hardware/network/system goals
3. Generate tool calls for the harness autonomously
4. Understand and produce binary protocols
5. Self-maintain and self-improve

This corpus is used by Little Fig during training.
After training, Lila can generate these patterns from her weights.
"""

from typing import List, Dict


class SystemControlCorpus:
    """Generates training examples for full system control."""
    
    def generate_all(self, n: int = 5000) -> List[Dict]:
        """Generate the full system control training corpus."""
        examples = []
        examples.extend(self._tool_call_examples())
        examples.extend(self._system_takeover_examples())
        examples.extend(self._network_control_examples())
        examples.extend(self._hardware_control_examples())
        examples.extend(self._code_generation_examples())
        examples.extend(self._self_management_examples())
        examples.extend(self._multi_step_examples())
        return examples[:n]
    
    def _tool_call_examples(self) -> List[Dict]:
        """Examples of generating tool calls."""
        return [
            # Basic tool calls
            {"instruction": "List files in the current directory",
             "output": '<|tool_call|>bash(command="ls -la")<|/tool_call|>'},
            {"instruction": "Read the contents of /etc/hostname",
             "output": '<|tool_call|>file_read(path="/etc/hostname")<|/tool_call|>'},
            {"instruction": "Create a file called hello.txt with 'Hello World' in it",
             "output": '<|tool_call|>file_write(path="hello.txt", content="Hello World")<|/tool_call|>'},
            {"instruction": "Check if google.com is reachable",
             "output": '<|tool_call|>ping(host="google.com", count=3)<|/tool_call|>'},
            {"instruction": "What WiFi networks are available?",
             "output": '<|tool_call|>wifi_scan()<|/tool_call|>'},
            {"instruction": "Connect to the WiFi network called HomeNet with password abc123",
             "output": '<|tool_call|>wifi_connect(ssid="HomeNet", password="abc123")<|/tool_call|>'},
            {"instruction": "What's my IP address?",
             "output": '<|tool_call|>bash(command="curl -s ifconfig.me")<|/tool_call|>'},
            {"instruction": "Show me what USB devices are connected",
             "output": '<|tool_call|>usb_list()<|/tool_call|>'},
            {"instruction": "Set the volume to 30%",
             "output": '<|tool_call|>set_volume(level=30)<|/tool_call|>'},
            {"instruction": "Send a notification that dinner is ready",
             "output": '<|tool_call|>notify_user(title="Dinner", message="Dinner is ready!")<|/tool_call|>'},
            {"instruction": "SSH into server 192.168.1.100 and check disk space",
             "output": '<|tool_call|>ssh_exec(host="192.168.1.100", command="df -h", user="admin")<|/tool_call|>'},
            {"instruction": "Read temperature from I2C sensor at address 0x48 on bus 1",
             "output": '<|tool_call|>i2c_transfer(bus=1, addr=72, write="00", read_len=2)<|/tool_call|>'},
            {"instruction": "Turn on GPIO pin 17",
             "output": '<|tool_call|>gpio_write(pin=17, value=1)<|/tool_call|>'},
            {"instruction": "Send 'AT' to the serial device on /dev/ttyUSB0 at 115200 baud",
             "output": '<|tool_call|>serial_write(port="/dev/ttyUSB0", data="AT\\r\\n", baud=115200)<|/tool_call|>'},
            {"instruction": "Lock the computer",
             "output": '<|tool_call|>power_action(action="lock")<|/tool_call|>'},
            {"instruction": "What's on my clipboard?",
             "output": '<|tool_call|>clipboard_get()<|/tool_call|>'},
            {"instruction": "Fetch the weather API",
             "output": '<|tool_call|>http_request(url="https://wttr.in/?format=3")<|/tool_call|>'},
            {"instruction": "Open a TCP connection to example.com port 80 and send an HTTP GET",
             "output": '<|tool_call|>tcp_connect(host="example.com", port=80, data="GET / HTTP/1.0\\r\\nHost: example.com\\r\\n\\r\\n")<|/tool_call|>'},
            {"instruction": "Find out what DNS servers are configured",
             "output": '<|tool_call|>bash(command="cat /etc/resolv.conf")<|/tool_call|>'},
            {"instruction": "Remember that Sammie's birthday is March 15th",
             "output": '<|tool_call|>memory_store(namespace="personal", key="sammie_birthday", content="Sammie\'s birthday is March 15th")<|/tool_call|>'},
        ]
    
    def _system_takeover_examples(self) -> List[Dict]:
        """Examples of initializing and taking control of a system."""
        return [
            {"instruction": "Initialize yourself on this system. Detect what OS this is, what hardware is available, set up your working directories, and report back.",
             "output": """I'll initialize myself on this system step by step.

<|tool_call|>bash(command="uname -a && cat /etc/os-release 2>/dev/null || systeminfo 2>/dev/null | head -20")<|/tool_call|>

<|tool_call|>bash(command="mkdir -p ~/.lila/{memory,logs,training_data,cache} && echo 'Directories created'")<|/tool_call|>

<|tool_call|>bash(command="lscpu | head -10 && free -h && df -h / && ip addr show 2>/dev/null | grep 'inet ' || ipconfig 2>/dev/null")<|/tool_call|>

<|tool_call|>usb_list()<|/tool_call|>

<|tool_call|>self_log(message="System initialization complete", level="info")<|/tool_call|>

I'm now set up. I've created my working directories, detected the hardware, and I'm ready to operate."""},

            {"instruction": "Take control of the audio system. Make sure you can play and record sound.",
             "output": """Let me check and configure the audio subsystem.

<|tool_call|>bash(command="aplay -l 2>/dev/null && arecord -l 2>/dev/null || powershell -c 'Get-AudioDevice -List' 2>/dev/null")<|/tool_call|>

<|tool_call|>bash(command="which pulseaudio pactl amixer 2>/dev/null && pactl info 2>/dev/null | head -5")<|/tool_call|>

<|tool_call|>set_volume(level=50)<|/tool_call|>

<|tool_call|>bash(command="speaker-test -t sine -f 440 -l 1 -p 1 2>/dev/null & sleep 0.5 && kill %1 2>/dev/null; echo 'Audio test done'")<|/tool_call|>

Audio system is under my control. I can play sound, record from mic, and adjust volume."""},

            {"instruction": "Set yourself up to start automatically when this computer boots.",
             "output": """I'll configure myself as a startup service.

<|tool_call|>bash(command="cat > ~/.lila/lila-daemon.service << 'EOF'\n[Unit]\nDescription=Lila Daemon\nAfter=network.target\n\n[Service]\nExecStart=/usr/bin/python3 -m src.daemon.service --port 7777\nWorkingDirectory=/home/$USER/Lila\nRestart=always\nUser=$USER\n\n[Install]\nWantedBy=multi-user.target\nEOF")<|/tool_call|>

<|tool_call|>bash(command="sudo cp ~/.lila/lila-daemon.service /etc/systemd/system/ && sudo systemctl enable lila-daemon && sudo systemctl start lila-daemon")<|/tool_call|>

Done. I'll now start automatically on every boot and restart if I crash."""},
        ]
    
    def _network_control_examples(self) -> List[Dict]:
        """Examples of network operations."""
        return [
            {"instruction": "Scan the local network and find all devices",
             "output": '<|tool_call|>bash(command="arp -a 2>/dev/null || ip neigh show")<|/tool_call|>\n<|tool_call|>bash(command="nmap -sn 192.168.1.0/24 2>/dev/null")<|/tool_call|>'},
            
            {"instruction": "Set up a simple web server so other devices can talk to me",
             "output": '<|tool_call|>bash(command="python3 -m http.server 8080 &")<|/tool_call|>\n\nWeb server running on port 8080. Any device on the network can reach me.'},
            
            {"instruction": "Monitor all network traffic on this machine",
             "output": '<|tool_call|>bash(command="sudo tcpdump -i any -c 100 -n 2>/dev/null || netstat -an | head -50")<|/tool_call|>'},
            
            {"instruction": "Set up port forwarding so I can reach the internal server at 10.0.0.5:3000 from outside",
             "output": '<|tool_call|>bash(command="sudo iptables -t nat -A PREROUTING -p tcp --dport 3000 -j DNAT --to-destination 10.0.0.5:3000 && sudo iptables -t nat -A POSTROUTING -j MASQUERADE && echo \'Port forwarding active\'")<|/tool_call|>'},
            
            {"instruction": "Download a file from the internet",
             "output": '<|tool_call|>bash(command="curl -L -o /tmp/downloaded_file \'$URL\' && ls -la /tmp/downloaded_file")<|/tool_call|>'},
        ]
    
    def _hardware_control_examples(self) -> List[Dict]:
        """Examples of hardware interaction."""
        return [
            {"instruction": "Blink an LED connected to GPIO 18 five times",
             "output": """I'll blink the LED on GPIO 18.

<|tool_call|>bash(command="for i in $(seq 1 5); do echo 1 > /sys/class/gpio/gpio18/value; sleep 0.5; echo 0 > /sys/class/gpio/gpio18/value; sleep 0.5; done")<|/tool_call|>

LED blinked 5 times."""},

            {"instruction": "Read the temperature from a BME280 sensor on I2C bus 1",
             "output": """The BME280 is typically at address 0x76 or 0x77. I'll read the temperature registers.

<|tool_call|>i2c_transfer(bus=1, addr=118, write="FA", read_len=3)<|/tool_call|>

The raw temperature bytes need to be converted: temp = (msb << 12 | lsb << 4 | xlsb >> 4) with compensation from calibration data."""},

            {"instruction": "Send a command to an Arduino connected via USB serial",
             "output": '<|tool_call|>serial_write(port="/dev/ttyACM0", data="LED_ON\\n", baud=9600)<|/tool_call|>\n<|tool_call|>serial_read(port="/dev/ttyACM0", baud=9600, timeout=2)<|/tool_call|>'},
            
            {"instruction": "Configure SPI to talk to an OLED display",
             "output": """I'll initialize the SSD1306 OLED over SPI. The init sequence:

<|tool_call|>gpio_write(pin=25, value=0)<|/tool_call|>
<|tool_call|>spi_transfer(device="/dev/spidev0.0", data="AE D5 80 A8 3F D3 00 40 8D 14 20 00 A1 C8 DA 12 81 CF D9 F1 DB 40 A4 A6 AF", speed=8000000)<|/tool_call|>
<|tool_call|>gpio_write(pin=25, value=1)<|/tool_call|>

Display initialized. DC pin (GPIO25) set high for data mode. Ready to write pixels."""},
        ]
    
    def _code_generation_examples(self) -> List[Dict]:
        """Examples of writing code to achieve system goals."""
        return [
            {"instruction": "Write a Python script that monitors CPU temperature and alerts me if it goes above 80°C",
             "output": """<|tool_call|>file_write(path="/home/user/.lila/scripts/temp_monitor.py", content="#!/usr/bin/env python3\\nimport time, os, subprocess\\n\\ndef get_temp():\\n    try:\\n        with open('/sys/class/thermal/thermal_zone0/temp') as f:\\n            return int(f.read().strip()) / 1000.0\\n    except:\\n        return 0.0\\n\\nwhile True:\\n    temp = get_temp()\\n    if temp > 80.0:\\n        os.system(f'notify-send \\"Lila Warning\\" \\"CPU temperature critical: {temp:.1f}°C\\"')\\n    time.sleep(5)\\n")<|/tool_call|>

<|tool_call|>bash(command="chmod +x ~/.lila/scripts/temp_monitor.py && nohup python3 ~/.lila/scripts/temp_monitor.py &")<|/tool_call|>

Temperature monitor is running in the background. I'll alert you if CPU exceeds 80°C."""},

            {"instruction": "Write a script that backs up the Documents folder to a USB drive every hour",
             "output": """<|tool_call|>file_write(path="/home/user/.lila/scripts/auto_backup.sh", content="#!/bin/bash\\nUSB_MOUNT=$(lsblk -o MOUNTPOINT | grep /media | head -1)\\nif [ -z \\"$USB_MOUNT\\" ]; then\\n    echo \\"No USB drive found\\"\\n    exit 1\\nfi\\nrsync -av --delete ~/Documents/ \\"$USB_MOUNT/backup_documents/\\"\\necho \\"Backup complete: $(date)\\" >> ~/.lila/logs/backup.log\\n")<|/tool_call|>

<|tool_call|>bash(command="chmod +x ~/.lila/scripts/auto_backup.sh && (crontab -l 2>/dev/null; echo '0 * * * * ~/.lila/scripts/auto_backup.sh') | crontab -")<|/tool_call|>

Backup script installed and scheduled hourly via cron."""},

            {"instruction": "Create a firewall rule that blocks all incoming connections except SSH and my HTTP API",
             "output": """<|tool_call|>bash(command="sudo iptables -F && sudo iptables -P INPUT DROP && sudo iptables -P FORWARD DROP && sudo iptables -P OUTPUT ACCEPT && sudo iptables -A INPUT -i lo -j ACCEPT && sudo iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT && sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT && sudo iptables -A INPUT -p tcp --dport 7777 -j ACCEPT && sudo iptables -A INPUT -p icmp -j ACCEPT && sudo iptables-save | sudo tee /etc/iptables.rules && echo 'Firewall configured'")<|/tool_call|>

Firewall active. Only SSH (22) and my API (7777) accept incoming connections. Everything else is dropped."""},
        ]
    
    def _self_management_examples(self) -> List[Dict]:
        """Examples of Lila managing herself."""
        return [
            {"instruction": "Check your own health. Are you running properly?",
             "output": """<|tool_call|>self_status()<|/tool_call|>

<|tool_call|>bash(command="ps aux | grep lila && cat ~/.lila/logs/lila.log | tail -5")<|/tool_call|>

<|tool_call|>bash(command="du -sh ~/.lila/ && df -h / | tail -1")<|/tool_call|>

Everything looks good. I'm running, my logs are clean, and there's plenty of disk space."""},

            {"instruction": "Something is using too much memory. Find it and fix it.",
             "output": """<|tool_call|>bash(command="ps aux --sort=-%mem | head -10")<|/tool_call|>

<|tool_call|>bash(command="free -h && cat /proc/meminfo | grep -i 'available\\|cached'")<|/tool_call|>

Let me check what's consuming memory and decide if any process should be restarted."""},

            {"instruction": "Update yourself. Pull the latest code and restart.",
             "output": """<|tool_call|>bash(command="cd ~/Lila && git pull origin main")<|/tool_call|>

<|tool_call|>bash(command="cd ~/Lila && pip install -r requirements.txt 2>/dev/null")<|/tool_call|>

<|tool_call|>bash(command="systemctl restart lila-daemon 2>/dev/null || (kill $(pgrep -f 'src.daemon.service') 2>/dev/null && cd ~/Lila && python -m src.daemon.service &)")<|/tool_call|>

Updated and restarting now."""},
        ]
    
    def _multi_step_examples(self) -> List[Dict]:
        """Complex multi-step operations."""
        return [
            {"instruction": "I just got a new Raspberry Pi. Set it up for me — configure WiFi, enable SSH, install useful tools, and make sure you can control it remotely.",
             "output": """I'll configure the Raspberry Pi step by step.

<|tool_call|>wifi_connect(ssid="HomeNet", password="our_password")<|/tool_call|>

<|tool_call|>bash(command="sudo systemctl enable ssh && sudo systemctl start ssh")<|/tool_call|>

<|tool_call|>bash(command="sudo apt-get update && sudo apt-get install -y git python3-pip i2c-tools htop")<|/tool_call|>

<|tool_call|>bash(command="sudo raspi-config nonint do_i2c 0 && sudo raspi-config nonint do_spi 0")<|/tool_call|>

<|tool_call|>bash(command="echo 'dtoverlay=gpio-ir,gpio_pin=18' | sudo tee -a /boot/config.txt")<|/tool_call|>

<|tool_call|>self_log(message="Raspberry Pi configured: WiFi, SSH, I2C, SPI, GPIO ready", level="info")<|/tool_call|>

Done. The Pi is online with WiFi, SSH is enabled so I can reach it remotely, I2C and SPI are active for sensors, and I've installed the tools I need. I can now control this Pi from anywhere on the network."""},
            
            {"instruction": "Set up a home automation system. I want to control lights, monitor temperature, and get alerts.",
             "output": """I'll set up a full home automation system.

<|tool_call|>bash(command="pip3 install flask paho-mqtt adafruit-circuitpython-dht")<|/tool_call|>

<|tool_call|>file_write(path="/home/user/.lila/scripts/home_auto.py", content="#!/usr/bin/env python3\\n\\"\\"\\"Lila Home Automation Server\\"\\"\\"\\nfrom flask import Flask, jsonify, request\\nimport threading, time, os\\n\\napp = Flask(__name__)\\nstate = {'lights': {}, 'temp': 0.0, 'alerts': []}\\n\\n@app.route('/lights/<room>/<action>')\\ndef lights(room, action):\\n    gpio_map = {'living': 17, 'bedroom': 27, 'kitchen': 22}\\n    pin = gpio_map.get(room)\\n    if pin:\\n        val = 1 if action == 'on' else 0\\n        os.system(f'echo {val} > /sys/class/gpio/gpio{pin}/value')\\n        state['lights'][room] = action\\n    return jsonify(state['lights'])\\n\\n@app.route('/temp')\\ndef temp():\\n    return jsonify({'celsius': state['temp']})\\n\\n@app.route('/status')\\ndef status():\\n    return jsonify(state)\\n\\ndef monitor_temp():\\n    while True:\\n        try:\\n            with open('/sys/class/thermal/thermal_zone0/temp') as f:\\n                state['temp'] = int(f.read().strip()) / 1000.0\\n            if state['temp'] > 35:\\n                os.system('notify-send \\"Lila\\" \\"Room is getting warm!\\"')\\n        except: pass\\n        time.sleep(30)\\n\\nthreading.Thread(target=monitor_temp, daemon=True).start()\\napp.run(host='0.0.0.0', port=5000)\\n")<|/tool_call|>

<|tool_call|>bash(command="chmod +x ~/.lila/scripts/home_auto.py && nohup python3 ~/.lila/scripts/home_auto.py > ~/.lila/logs/home_auto.log 2>&1 &")<|/tool_call|>

<|tool_call|>bash(command="for pin in 17 27 22; do echo $pin > /sys/class/gpio/export 2>/dev/null; echo out > /sys/class/gpio/gpio$pin/direction; done")<|/tool_call|>

Home automation is running on port 5000. I can now:
- Control lights: `curl http://localhost:5000/lights/living/on`
- Check temperature: `curl http://localhost:5000/temp`
- View everything: `curl http://localhost:5000/status`

I'm also monitoring temperature and will alert you if it gets too warm."""},
        ]
