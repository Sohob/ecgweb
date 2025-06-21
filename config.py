"""
Configuration file for ECG Sensor Control WebServer
Modify these settings according to your setup
"""

# Server Configuration
SERVER_HOST = "0.0.0.0"  # Bind to all interfaces
SERVER_PORT = 8000        # Port number

# Serial Communication Settings
SERIAL_PORT = "/dev/ttyUSB0"  # Serial port for sensor connection
BAUD_RATE = 115200              # Baud rate for serial communication
SERIAL_TIMEOUT = 1            # Timeout in seconds

# Sensor Communication Protocol
SERIAL_COMMAND_LENGTH = 5     # All commands are 5 bytes
DONTCARE_BYTE = 0x00          # Don't care byte value

# Command Definitions (5-byte format)
CMD_STOP_DAQ = 0x20           # Stop Data Acquisition
CMD_SET_SAMPLING_FREQ = 0x30  # Set Sampling Frequency
CMD_SET_RECORDING_TIME = 0x40 # Set Recording Time
CMD_START_DAQ = 0x50          # Start Data Acquisition

# Command Parameters
DEFAULT_SAMPLING_FREQ = 1000  # Default 1 kHz if not set
CONTINUOUS_RECORDING = 0      # 0 = continuous recording
FREE_RUNNING = 0x00           # Free running mode (not external trigger)
EXTERNAL_TRIGGER = 0xAA       # Wait for external trigger

# Sensor Configuration
DEFAULT_FREQUENCY = 100       # Default sampling frequency in Hz
DEFAULT_REFRESH_RATE = 10     # Default refresh rate in Hz
MIN_FREQUENCY = 1             # Minimum allowed frequency
MAX_FREQUENCY = 1000          # Maximum allowed frequency
MIN_REFRESH_RATE = 1          # Minimum allowed refresh rate
MAX_REFRESH_RATE = 60         # Maximum allowed refresh rate

# WebSocket Settings
WEBSOCKET_RECONNECT_ATTEMPTS = 5  # Maximum reconnection attempts
WEBSOCKET_RECONNECT_DELAY = 2     # Delay between reconnection attempts (seconds)

# Logging Settings
LOG_MAX_ENTRIES = 50         # Maximum number of log entries to keep in web interface
LOG_UPDATE_INTERVAL = 5      # Status update interval (seconds)

# Valid Commands
VALID_COMMANDS = ["start", "stop", "record"]

# Development Settings
DEBUG_MODE = False           # Enable debug mode
AUTO_RELOAD = False          # Enable auto-reload for development

# Platform-specific settings
import platform
if platform.system() == "Windows":
    # Windows serial port examples
    SERIAL_PORT = "COM6"  # Change to your COM port
elif platform.system() == "Darwin":
    # macOS serial port examples
    SERIAL_PORT = "/dev/tty.usbserial-0001"  # Change to your port
else:
    # Linux/Raspberry Pi (default)
    SERIAL_PORT = "/dev/ttyUSB0"  # Change to your port 