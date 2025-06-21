# ECG Sensor Control WebServer

A FastAPI-based webserver for controlling an ECG sensor connected via serial communication. The server provides a modern web interface with real-time command buttons and status monitoring.

## Features

- **Modern Web Interface**: Responsive design with beautiful UI for sensor control
- **Real-time Communication**: WebSocket support for live updates
- **Serial Communication**: Direct control of sensor via serial port
- **Command Logging**: Real-time log of all commands and responses
- **Status Monitoring**: Live system status including connection state
- **RESTful API**: Clean API endpoints for programmatic access

## Prerequisites

- Python 3.8 or higher
- Raspberry Pi (or any device with serial port)
- ECG sensor connected via serial communication
- Network access for web interface

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure serial port** (if needed):
   - Edit `main.py` and update the `SERIAL_PORT` variable
   - Default is `/dev/ttyUSB0` for Raspberry Pi
   - For Windows, use `COM1`, `COM2`, etc.
   - For macOS, use `/dev/tty.usbserial-*`

## Usage

### Starting the Server

```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Accessing the Web Interface

1. Open your web browser
2. Navigate to `http://your-raspberry-pi-ip:8000`
3. You'll see the control interface with three main buttons:
   - **Start**: Initiates sensor data collection
   - **Stop**: Stops sensor data collection
   - **Record**: Starts recording sensor data

### API Endpoints

#### Web Interface
- `GET /` - Main web interface

#### Commands
- `POST /command/{cmd}` - Send command to sensor
  - Valid commands: `start`, `stop`, `record`
  - Returns: JSON response with status and message

#### Status
- `GET /status` - Get system status
  - Returns: Serial connection status, port info, active connections

#### WebSocket
- `WS /ws` - WebSocket endpoint for real-time updates
  - Receives: Command responses and system updates
  - Sends: Echo of received messages (for testing)

## Configuration

### Serial Port Settings

Edit the following variables in `main.py`:

```python
SERIAL_PORT = "/dev/ttyUSB0"  # Change to your serial port
BAUD_RATE = 9600              # Change to match your sensor's baud rate
```

### Common Serial Ports

- **Raspberry Pi**: `/dev/ttyUSB0`, `/dev/ttyACM0`
- **Windows**: `COM1`, `COM2`, `COM3`, etc.
- **macOS**: `/dev/tty.usbserial-*`, `/dev/tty.usbmodem*`

## Troubleshooting

### Serial Connection Issues

1. **Check port permissions** (Linux/Raspberry Pi):
   ```bash
   sudo usermod -a -G dialout $USER
   # Then logout and login again
   ```

2. **List available ports**:
   ```bash
   # Linux/macOS
   ls /dev/tty*
   
   # Windows (PowerShell)
   [System.IO.Ports.SerialPort]::getportnames()
   ```

3. **Test serial connection**:
   ```bash
   # Linux/macOS
   screen /dev/ttyUSB0 9600
   
   # Windows
   # Use PuTTY or similar terminal program
   ```

### Web Interface Issues

1. **Check if server is running**:
   ```bash
   curl http://localhost:8000/status
   ```

2. **Check firewall settings**:
   - Ensure port 8000 is open
   - For Raspberry Pi: `sudo ufw allow 8000`

3. **Check network connectivity**:
   ```bash
   # From another device
   ping your-raspberry-pi-ip
   ```

## Development

### Project Structure

```
ecgweb/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── templates/
│   └── index.html      # Web interface template
├── static/             # Static files (CSS, JS, images)
└── README.md           # This file
```

### Adding New Commands

1. Add the command to the validation list in `main.py`:
   ```python
   if cmd not in ["start", "stop", "record", "your_new_command"]:
   ```

2. Update the web interface in `templates/index.html`:
   ```html
   <button class="control-btn your-command" onclick="sendCommand('your_new_command')">
       Your Command
   </button>
   ```

### Extending the Interface

The web interface is built with vanilla HTML, CSS, and JavaScript. You can easily:
- Add new control buttons
- Modify the styling
- Add real-time data visualization
- Implement additional status monitoring

## Security Considerations

- The server runs on all interfaces (`0.0.0.0`) by default
- Consider adding authentication for production use
- Use HTTPS in production environments
- Restrict access to trusted networks

## License

This project is open source. Feel free to modify and distribute as needed.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your serial port configuration
3. Check the command log in the web interface for error messages
4. Ensure your sensor is properly connected and powered 