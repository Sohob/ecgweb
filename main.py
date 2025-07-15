from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Depends, status, Form
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import serial
import json
import asyncio
import time
import numpy as np
import struct
from typing import List, Dict, Any, Optional
import os
from config import *
from scipy import signal
from collections import deque
import math
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from auth_config import *
import io

app = FastAPI(title="ECG Sensor Control Server", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security
security = HTTPBearer()

# Session storage (in-memory for simplicity)
active_sessions = {}

def verify_password(plain_password, hashed_password):
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """Hash a password"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    """Verify a JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return username
    except JWTError:
        return None

def get_current_user(request: Request):
    """Get current authenticated user from session"""
    session_token = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_token:
        return None
    
    username = verify_token(session_token)
    if not username or username not in active_sessions:
        return None
    
    return username

def require_auth(request: Request):
    """Dependency to require authentication"""
    user = get_current_user(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return user

# Serial connection (will be initialized when needed)
serial_connection = None
serial_connected = False

# Current settings and state
current_frequency = DEFAULT_FREQUENCY
current_refresh_rate = DEFAULT_REFRESH_RATE
is_running = False
current_sampling_freq = DEFAULT_SAMPLING_FREQ
current_recording_time = CONTINUOUS_RECORDING

# Recording state and data
recording_mode = None  # 'interval' or 'manual'
is_recording = False
recording_start_time = None
recording_duration = None  # For interval recording
recording_scheduled_start = None  # Scheduled start time for interval recording
recording_task = None  # Background task for interval recording
recorded_data = None  # Will be initialized after lead definitions

# Playback state
is_playback = False
playback_position = 0
playback_speed = 1.0

# Data reading settings
SAMPLE_SIZE = 27  # 27 bytes per sample from sensor
serial_data_buffer = bytearray()
last_sample_time = 0
samples_received = 0

# ECG Data Processing Constants (placeholders - adjust based on your hardware)
class ECGConstants:
    REFERENCE_VOLTAGE_MV = 2400.0  # Reference voltage in mV
    LSB_STEPS = 16777216-1
    OFFSET_LSB_STEPS = LSB_STEPS / 4  # Offset in LSB steps
    ADC_RESOLUTION = 24  # ADC resolution in bits
    MAX_ADC_VALUE = (2 ** 24) - 1  # Maximum ADC value

# ECG Lead Names
ECG_LEADS = ['Lead1', 'Lead2', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
DERIVED_LEADS = ['Lead3', 'aVL', 'aVR', 'aVF']
ALL_LEADS = ECG_LEADS + DERIVED_LEADS  # All 12 channels

# Initialize recording data structure
recorded_data = {
    'timestamps': [],
    'measured_leads': {lead: [] for lead in ECG_LEADS},
    'derived_leads': {lead: [] for lead in DERIVED_LEADS},
    'filtered_leads': {lead: [] for lead in ALL_LEADS},
    'metadata': {
        'start_time': None,
        'end_time': None,
        'duration': None,
        'sampling_frequency': None,
        'mode': None
    }
}

# IIR Filter Configuration
class IIRFilterConfig:
    HPF_CUTOFF = 0.5   # High-pass cutoff frequency in Hz
    LPF_CUTOFF = 150.0 # Low-pass cutoff frequency in Hz
    FILTER_ORDER = 4   # Butterworth filter order (2 or 4 recommended)

# Online IIR Filter Implementation
class OnlineIIRFilter:
    """Online IIR bandpass filter for real-time processing (Butterworth)"""
    def __init__(self, b, a):
        self.b = b
        self.a = a
        self.zi = None  # Filter state
        self.reset()

    def filter(self, input_value: float) -> float:
        y, self.zi = signal.lfilter(self.b, self.a, [input_value], zi=self.zi)
        return y[0]

    def reset(self):
        # Initialize filter state for step response at zero
        self.zi = signal.lfilter_zi(self.b, self.a) * 0.0

# Initialize IIR filters for all channels
iir_filters = {}

def initialize_iir_filters(sampling_rate: float):
    """
    Initialize IIR bandpass filters for all ECG channels.
    This function dynamically recalculates filter coefficients and state
    whenever the sampling rate changes.
    """
    global iir_filters
    nyquist = 0.5 * sampling_rate
    low = IIRFilterConfig.HPF_CUTOFF / nyquist
    high = IIRFilterConfig.LPF_CUTOFF / nyquist
    if high >= 1.0:
        print(f"Warning: LPF cutoff {IIRFilterConfig.LPF_CUTOFF} Hz is too high for sampling rate {sampling_rate} Hz")
        high = 0.99
    if low <= 0.0:
        print(f"Warning: HPF cutoff {IIRFilterConfig.HPF_CUTOFF} Hz is too low for sampling rate {sampling_rate} Hz")
        low = 0.001
    b, a = signal.butter(IIRFilterConfig.FILTER_ORDER, [low, high], btype='band')
    print(f"IIR Filter Design:")
    print(f"  Sampling Rate: {sampling_rate} Hz")
    print(f"  Bandpass: {IIRFilterConfig.HPF_CUTOFF}–{IIRFilterConfig.LPF_CUTOFF} Hz")
    print(f"  Normalized: {low:.4f}–{high:.4f}")
    print(f"  Order: {IIRFilterConfig.FILTER_ORDER}")
    iir_filters.clear()
    for lead_name in ALL_LEADS:
        iir_filters[lead_name] = OnlineIIRFilter(b, a)
    print(f"Initialized IIR filters for {len(ALL_LEADS)} channels: {ALL_LEADS}")

def reset_iir_filters():
    """Reset all IIR filter states."""
    global iir_filters
    for filter_obj in iir_filters.values():
        filter_obj.reset()
    print("All IIR filters reset")

# Plotting data storage - now includes filtered data
plot_data = {
    'timestamps': [],
    'measured_leads': {lead: [] for lead in ECG_LEADS},
    'derived_leads': {lead: [] for lead in DERIVED_LEADS},
    'filtered_leads': {lead: [] for lead in ALL_LEADS},  # Filtered data for all leads
    'raw_samples': [],  # Store raw sample data for debugging
    'heart_rate': 0     # Store calculated heart rate (BPM)
}
max_data_points = 1000  # Maximum number of points to keep in memory

# Pydantic model for configuration
class SensorConfig(BaseModel):
    frequency: int
    refresh_rate: int

# Pydantic models for recording
class RecordingConfig(BaseModel):
    mode: str  # 'interval' or 'manual'
    duration: Optional[int] = None  # Duration in seconds for interval recording
    start_time: Optional[str] = None  # Start time for interval recording (ISO format)
    
    class Config:
        # Allow extra fields to be ignored
        extra = "ignore"
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.mode not in ['interval', 'manual']:
            raise ValueError("Mode must be either 'interval' or 'manual'")

class ExportRequest(BaseModel):
    format: str  # 'csv' or 'bdf'
    filename: Optional[str] = None
    bdf_metadata: Optional[dict] = None
    
    class Config:
        # Allow extra fields to be ignored
        extra = "ignore"
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.format not in ['csv', 'bdf']:
            raise ValueError("Format must be either 'csv' or 'bdf'")

# WebSocket connections for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove dead connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

def init_serial():
    """Initialize serial connection to the sensor"""
    global serial_connection, serial_connected
    try:
        serial_connection = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
        serial_connected = True
        print(f"Serial connection established on {SERIAL_PORT}")
        return True
    except Exception as e:
        serial_connected = False
        print(f"Failed to connect to serial port: {e}")
        return False

def close_serial():
    """Close serial connection"""
    global serial_connection, serial_connected
    if serial_connection and serial_connection.is_open:
        serial_connection.close()
        serial_connected = False
        print("Serial connection closed")

def format_5byte_command(command_byte, data_bytes):
    """Format a 5-byte command with proper byte structure"""
    if len(data_bytes) != 4:
        raise ValueError("Data must be exactly 4 bytes")
    
    command = bytearray([command_byte] + data_bytes)
    return command

def int_to_4bytes(value):
    """Convert integer to 4 bytes (MSB first)"""
    return list(struct.pack('>I', value))  # Big-endian, unsigned int

def send_serial_command(command_byte, data_value=0):
    """Send a 5-byte command to the sensor"""
    global serial_connection, serial_connected
    
    if not serial_connected or not serial_connection or not serial_connection.is_open:
        return {"status": "error", "message": "Serial connection not available"}
    
    try:
        # Convert data value to 4 bytes (MSB first)
        data_bytes = int_to_4bytes(data_value)
        
        # Format the 5-byte command
        command = format_5byte_command(command_byte, data_bytes)
        
        # Send the command
        serial_connection.write(command)
        serial_connection.flush()  # Ensure data is sent immediately
        
        # Log the command for debugging
        command_hex = ' '.join([f'0x{b:02x}' for b in command])
        print(f"Sent command: {command_hex}")
        
        return {"status": "success", "message": f"Command 0x{command_byte:02x} sent successfully"}
        
    except Exception as e:
        return {"status": "error", "message": f"Failed to send command: {e}"}

def send_sensor_command(command_type, frequency=None, refresh_rate=None):
    """Send appropriate sensor command based on command type"""
    global current_sampling_freq, current_recording_time
    
    if command_type == "start":
        # Set sampling frequency first
        if frequency:
            current_sampling_freq = frequency
            freq_result = send_serial_command(CMD_SET_SAMPLING_FREQ, frequency)
            if freq_result["status"] != "success":
                return freq_result
        
        # Set recording time (continuous)
        current_recording_time = CONTINUOUS_RECORDING
        time_result = send_serial_command(CMD_SET_RECORDING_TIME, CONTINUOUS_RECORDING)
        if time_result["status"] != "success":
            return time_result
        
        # Start DAQ in free running mode
        start_result = send_serial_command(CMD_START_DAQ, FREE_RUNNING)
        return start_result
        
    elif command_type == "stop":
        # Stop DAQ
        return send_serial_command(CMD_STOP_DAQ)
        
    elif command_type == "record":
        # For recording, we could set a specific recording time
        # For now, just start DAQ (continuous recording)
        return send_serial_command(CMD_START_DAQ, FREE_RUNNING)
    
    else:
        return {"status": "error", "message": "Unknown command type"}

def get_voltage_from_raw_data(value: int, gain: int = 1) -> float:
    """Convert raw ADC value to voltage (equivalent to C# GetVoltageFromRawData)"""
    lsb = (2 * ECGConstants.REFERENCE_VOLTAGE_MV) / ECGConstants.MAX_ADC_VALUE
    
    # Apply XOR with 0x800000 and subtract 1 (equivalent to C# logic)
    val = (value ^ 0x800000) - 1
    retval = (val * lsb - ECGConstants.REFERENCE_VOLTAGE_MV) / gain
    
    return retval

def get_lsb_from_raw_data(value: int) -> int:
    """Get LSB from raw data (equivalent to C# GetLSBFromRawData)"""
    return ((value ^ 0x800000) - 1) - ECGConstants.OFFSET_LSB_STEPS

def get_lsb_from_voltage(voltage: float, gain: int = 1) -> int:
    """Get LSB from voltage (equivalent to C# GetLSBFromVoltage)"""
    lsb = (2 * ECGConstants.REFERENCE_VOLTAGE_MV) / ECGConstants.MAX_ADC_VALUE
    
    retval = int((voltage * gain + ECGConstants.REFERENCE_VOLTAGE_MV) / lsb)
    retval = retval - (-ECGConstants.OFFSET_LSB_STEPS)
    return retval

def extract_3byte_value(data: bytes, start_index: int) -> int:
    """Extract 3-byte value with LSB at the end"""
    if start_index + 2 >= len(data):
        raise ValueError(f"Not enough data to extract 3-byte value at index {start_index}")
    
    # Extract 3 bytes and convert to 24-bit value (LSB at end)
    byte1, byte2, byte3 = data[start_index], data[start_index + 1], data[start_index + 2]
    
    # Combine bytes: MSB first, LSB last
    value = (0x00 << 24) | (byte1 << 16) | (byte2 << 8) | byte3
    
    
    return value

def calculate_derived_leads(lead1_voltage: float, lead2_voltage: float) -> Dict[str, float]:
    """Calculate derived ECG leads from Lead I and Lead II"""
    # LEAD III = LEAD II - LEAD I
    lead3 = lead2_voltage - lead1_voltage
    
    # aVL = 1/2(LEAD I - LEAD III)
    avl = 0.5 * (lead1_voltage - lead3)
    
    # -aVR = 1/2(LEAD I + LEAD II) (note: this is -aVR, so we'll store it as negative)
    avr = -0.5 * (lead1_voltage + lead2_voltage)
    
    # aVF = 1/2(LEAD II + LEAD III)
    avf = 0.5 * (lead2_voltage + lead3)
    
    return {
        'Lead3': lead3,
        'aVL': avl,
        'aVR': avr,
        'aVF': avf
    }

def apply_iir_filters(lead_data: Dict[str, float], derived_leads: Dict[str, float]) -> Dict[str, float]:
    """Apply IIR filters to all leads and return filtered values"""
    global iir_filters
    filtered_data = {}
    for lead_name in ECG_LEADS:
        if lead_name in iir_filters:
            filtered_data[lead_name] = iir_filters[lead_name].filter(lead_data[lead_name])
        else:
            filtered_data[lead_name] = lead_data[lead_name]
    for lead_name in DERIVED_LEADS:
        if lead_name in iir_filters:
            filtered_data[lead_name] = iir_filters[lead_name].filter(derived_leads[lead_name])
        else:
            filtered_data[lead_name] = derived_leads[lead_name]
    return filtered_data

# --- Heart rate calculation state ---
_hr_window_seconds = 60
_hr_last_calc_time = None
_hr_peak_times = []
_hr_prev_value = 0
_hr_prev_sign = 0
_hr_min_peak_height = 0.9  # mV, adjust as needed for your signal

def process_sensor_sample(sample_data: bytes, timestamp: float):
    """Process a complete 27-byte sample from the sensor"""
    global samples_received, plot_data
    global _hr_last_calc_time, _hr_peak_times, _hr_prev_value, _hr_prev_sign
    
    if len(sample_data) != SAMPLE_SIZE:
        print(f"Warning: Received incomplete sample of {len(sample_data)} bytes")
        return
    
    samples_received += 1
    
    try:
        # Extract status bytes (first 3 bytes - ignored for now)
        status_bytes = sample_data[0:3]
        
        # Extract 8 channels of 3-byte data each (starting from byte 3)
        lead_data = {}
        for i, lead_name in enumerate(ECG_LEADS):
            start_index = 3 + (i * 3)  # Start after status bytes
            raw_value = extract_3byte_value(sample_data, start_index)
            
            # Convert to voltage (assuming gain = 1 for now)
            voltage = get_voltage_from_raw_data(raw_value, gain=6)
            lead_data[lead_name] = voltage
        
        # Calculate derived leads
        derived_leads = calculate_derived_leads(lead_data['Lead1'], lead_data['Lead2'])
        
        # Apply IIR filters to all leads
        filtered_leads = apply_iir_filters(lead_data, derived_leads)
        
        # Store data for plotting
        plot_data['timestamps'].append(timestamp)
        
        # Store measured leads (unfiltered)
        for lead_name in ECG_LEADS:
            plot_data['measured_leads'][lead_name].append(lead_data[lead_name])
        
        # Store derived leads (unfiltered)
        for lead_name in DERIVED_LEADS:
            plot_data['derived_leads'][lead_name].append(derived_leads[lead_name])
        
        # Store filtered leads (for plotting)
        for lead_name in ALL_LEADS:
            plot_data['filtered_leads'][lead_name].append(filtered_leads[lead_name])
        
        # Store raw sample for debugging
        sample_hex = ' '.join([f'0x{b:02x}' for b in sample_data])
        plot_data['raw_samples'].append({
            'sample_number': samples_received,
            'timestamp': timestamp,
            'raw_hex': sample_hex,
            'lead_data': lead_data,
            'derived_leads': derived_leads,
            'filtered_leads': filtered_leads
        })
        
        # Add data to recording if active
        add_to_recording(timestamp, lead_data, derived_leads, filtered_leads)
        
        # Keep only last max_data_points
        if len(plot_data['timestamps']) > max_data_points:
            plot_data['timestamps'] = plot_data['timestamps'][-max_data_points:]
            for lead_name in ECG_LEADS:
                plot_data['measured_leads'][lead_name] = plot_data['measured_leads'][lead_name][-max_data_points:]
            for lead_name in DERIVED_LEADS:
                plot_data['derived_leads'][lead_name] = plot_data['derived_leads'][lead_name][-max_data_points:]
            for lead_name in ALL_LEADS:
                plot_data['filtered_leads'][lead_name] = plot_data['filtered_leads'][lead_name][-max_data_points:]
            plot_data['raw_samples'] = plot_data['raw_samples'][-max_data_points:]
        
        # --- Heart rate calculation (simple threshold peak detection on Lead1) ---
        lead_name = 'Lead1'
        value = filtered_leads[lead_name]
        v = value if isinstance(value, float) else float(value)
        # Detect upward zero-crossing (simple peak detection)
        sign = 1 if v > _hr_min_peak_height else 0
        if _hr_prev_sign == 0 and sign == 1:
            _hr_peak_times.append(timestamp)
        _hr_prev_sign = sign
        # Remove peaks outside the window
        _hr_peak_times = [t for t in _hr_peak_times if t >= timestamp - _hr_window_seconds]
        # Calculate BPM every second
        if _hr_last_calc_time is None or timestamp - _hr_last_calc_time > 1.0:
            beats = len(_hr_peak_times)
            plot_data['heart_rate'] = int(beats * (60 / _hr_window_seconds))
            _hr_last_calc_time = timestamp
        
    except Exception as e:
        print(f"Error processing sample #{samples_received}: {e}")
        # Print raw data for debugging
        sample_hex = ' '.join([f'0x{b:02x}' for b in sample_data])
        print(f"Raw sample data: {sample_hex}")

def read_serial_data():
    """Read data from serial port and process complete samples"""
    global serial_connection, serial_connected, serial_data_buffer, last_sample_time
    
    if not serial_connected or not serial_connection or not serial_connection.is_open:
        return
    
    try:
        # Read available data
        if serial_connection.in_waiting > 0:
            data = serial_connection.read(serial_connection.in_waiting)
            serial_data_buffer.extend(data)
            
            # Process complete samples
            while len(serial_data_buffer) >= SAMPLE_SIZE:
                # Extract one complete sample
                sample = serial_data_buffer[:SAMPLE_SIZE]
                serial_data_buffer = serial_data_buffer[SAMPLE_SIZE:]
                
                # Get current timestamp
                current_time = time.time()
                
                # Process the sample
                process_sensor_sample(sample, current_time)
                
                # Update timing
                last_sample_time = current_time
                
    except Exception as e:
        print(f"Error reading serial data: {e}")
        # Reset buffer on error
        serial_data_buffer.clear()

def generate_sinusoidal_data():
    """Generate sinusoidal wave data for testing"""
    current_time = time.time()
    
    # Generate three different sinusoidal waves
    # Signal 1: Main frequency wave
    freq1 = current_frequency
    signal1 = np.sin(2 * np.pi * freq1 * current_time) + 0.5 * np.sin(2 * np.pi * freq1 * 2 * current_time)
    
    # Signal 2: Higher frequency component
    freq2 = current_frequency * 1.5
    signal2 = 0.7 * np.sin(2 * np.pi * freq2 * current_time) + 0.3 * np.cos(2 * np.pi * freq2 * 0.5 * current_time)
    
    # Signal 3: Lower frequency component with noise
    freq3 = current_frequency * 0.3
    noise = 0.1 * np.random.normal(0, 1)
    signal3 = 0.8 * np.sin(2 * np.pi * freq3 * current_time) + noise
    
    return current_time, signal1, signal2, signal3

def update_plot_data(timestamp, signal1, signal2, signal3):
    """Update plot data and maintain maximum data points"""
    global plot_data
    
    plot_data['timestamps'].append(timestamp)
    plot_data['signal1'].append(float(signal1))
    plot_data['signal2'].append(float(signal2))
    plot_data['signal3'].append(float(signal3))
    
    # Keep only the last max_data_points
    if len(plot_data['timestamps']) > max_data_points:
        plot_data['timestamps'] = plot_data['timestamps'][-max_data_points:]
        plot_data['signal1'] = plot_data['signal1'][-max_data_points:]
        plot_data['signal2'] = plot_data['signal2'][-max_data_points:]
        plot_data['signal3'] = plot_data['signal3'][-max_data_points:]

def largest_triangle_three_buckets(data: list, threshold: int):
    """
    LTTB downsampling for a list of (x, y) tuples.
    Returns a list of (x, y) tuples downsampled to 'threshold' points.
    """
    data_length = len(data)
    if threshold >= data_length or threshold == 0:
        return data
    sampled = [data[0]]
    every = (data_length - 2) / (threshold - 2)
    a = 0
    for i in range(threshold - 2):
        avg_range_start = int(math.floor((i + 1) * every) + 1)
        avg_range_end = int(math.floor((i + 2) * every) + 1)
        avg_range_end = avg_range_end if avg_range_end < data_length else data_length
        avg_range_length = avg_range_end - avg_range_start
        avg_x = avg_y = 0.0
        if avg_range_length > 0:
            for idx in range(avg_range_start, avg_range_end):
                avg_x += data[idx][0]
                avg_y += data[idx][1]
            avg_x /= avg_range_length
            avg_y /= avg_range_length
        else:
            avg_x = data[a][0]
            avg_y = data[a][1]
        range_offs = int(math.floor((i + 0) * every) + 1)
        range_to = int(math.floor((i + 1) * every) + 1)
        range_to = range_to if range_to < data_length else data_length
        max_area = -1.0
        max_area_point = None
        next_a = None
        for idx in range(range_offs, range_to):
            area = abs((data[a][0] - avg_x) * (data[idx][1] - data[a][1]) -
                       (data[a][0] - data[idx][0]) * (avg_y - data[a][1])) * 0.5
            if area > max_area:
                max_area = area
                max_area_point = data[idx]
                next_a = idx
        if max_area_point is not None:
            sampled.append(max_area_point)
            a = next_a
    sampled.append(data[-1])
    return sampled

async def serial_reading_task():
    """Background task for reading serial data from sensor"""
    global is_running, current_sampling_freq
    
    while True:
        if is_running and serial_connected:
            # Read data from serial port
            read_serial_data()
            
            # Sleep based on sampling frequency to avoid overwhelming the system
            # We read as fast as possible but process at the sampling rate
            await asyncio.sleep(1.0 / (current_sampling_freq * 2))  # Read twice as fast as sampling
        else:
            # When not running, sleep longer to reduce CPU usage
            await asyncio.sleep(0.1)

async def data_generation_task():
    """Background task for generating and broadcasting plot data"""
    LTTB_TARGET_POINTS = 150  # Number of points to send to frontend
    while True:
        if is_running:
            # Check if we have real sensor data
            if len(plot_data['timestamps']) > 0 and len(plot_data['filtered_leads']['Lead1']) > 0:
                latest_index = -1
                # Downsample each signal for plotting
                timestamps = plot_data['timestamps']
                signals = {
                    'Lead1': plot_data['filtered_leads']['Lead1'],
                    'Lead2': plot_data['filtered_leads']['Lead2'],
                    'V1': plot_data['filtered_leads']['V1'],
                    'V2': plot_data['filtered_leads']['V2'],
                    'V3': plot_data['filtered_leads']['V3'],
                    'V4': plot_data['filtered_leads']['V4'],
                    'V5': plot_data['filtered_leads']['V5'],
                    'V6': plot_data['filtered_leads']['V6'],
                    'Lead3': plot_data['filtered_leads']['Lead3'],
                    'aVL': plot_data['filtered_leads']['aVL'],
                    'aVR': plot_data['filtered_leads']['aVR'],
                    'aVF': plot_data['filtered_leads']['aVF'],
                }
                downsampled = {}
                for key, ydata in signals.items():
                    if len(timestamps) == len(ydata) and len(timestamps) > 2:
                        xy = list(zip(timestamps, ydata))
                        lttb = largest_triangle_three_buckets(xy, LTTB_TARGET_POINTS)
                        # Unzip
                        _, y_down = zip(*lttb)
                        downsampled[key] = list(y_down)
                    else:
                        downsampled[key] = ydata[-LTTB_TARGET_POINTS:] if len(ydata) > LTTB_TARGET_POINTS else ydata
                # Downsample timestamps for x-axis
                if len(timestamps) > 2:
                    xy = list(zip(timestamps, signals['Lead1']))
                    lttb = largest_triangle_three_buckets(xy, LTTB_TARGET_POINTS)
                    x_down, _ = zip(*lttb)
                    downsampled_timestamps = list(x_down)
                else:
                    downsampled_timestamps = timestamps[-LTTB_TARGET_POINTS:] if len(timestamps) > LTTB_TARGET_POINTS else timestamps
                # Use last available values for meta info
                timestamp = downsampled_timestamps[-1] if downsampled_timestamps else 0
                plot_update = {
                    "type": "plot_data",
                    "timestamp": timestamp,
                    "frequency": current_frequency,
                    "refresh_rate": current_refresh_rate,
                    "samples_received": samples_received,
                    "buffer_size": len(serial_data_buffer),
                    "heart_rate": plot_data['heart_rate'],
                    "downsampled_timestamps": downsampled_timestamps,
                }
                # Add all 12 leads to the plot_update
                for key in signals.keys():
                    plot_update[f"downsampled_{key}"] = downsampled[key]
                    # For backward compatibility, keep the first 6 as signal1..signal6
                plot_update["signal1"] = downsampled['Lead1'][-1] if downsampled['Lead1'] else 0
                plot_update["signal2"] = downsampled['Lead2'][-1] if downsampled['Lead2'] else 0
                plot_update["signal3"] = downsampled['V1'][-1] if downsampled['V1'] else 0
                plot_update["lead3"] = downsampled['Lead3'][-1] if downsampled['Lead3'] else 0
                plot_update["avl"] = downsampled['aVL'][-1] if downsampled['aVL'] else 0
                plot_update["avf"] = downsampled['aVF'][-1] if downsampled['aVF'] else 0
            else:
                # Generate test data when no sensor data available
                timestamp, signal1, signal2, signal3 = generate_sinusoidal_data()
                update_plot_data(timestamp, signal1, signal2, signal3)
                plot_update = {
                    "type": "plot_data",
                    "timestamp": timestamp,
                    "signal1": float(signal1),
                    "signal2": float(signal2),
                    "signal3": float(signal3),
                    "frequency": current_frequency,
                    "refresh_rate": current_refresh_rate,
                    "samples_received": samples_received,
                    "buffer_size": len(serial_data_buffer),
                    "heart_rate": plot_data['heart_rate']
                }
            # Broadcast plot data to all connected clients
            await manager.broadcast(json.dumps(plot_update))
            # Sleep based on refresh rate
            await asyncio.sleep(1.0 / current_refresh_rate)
        else:
            # When not running, sleep longer to reduce CPU usage
            await asyncio.sleep(1.0)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main web interface"""
    # Check if user is authenticated
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Serve the login page"""
    # If already authenticated, redirect to main page
    user = get_current_user(request)
    if user:
        return RedirectResponse(url="/", status_code=302)
    
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Handle login authentication"""
    # Check credentials
    if username == DEFAULT_USERNAME and verify_password(password, get_password_hash(DEFAULT_PASSWORD)):
        # Create session token
        access_token = create_access_token(data={"sub": username})
        
        # Store session
        active_sessions[username] = {
            "token": access_token,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow()
        }
        
        # Create response with session cookie
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=access_token,
            max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            httponly=SESSION_COOKIE_HTTPONLY,
            secure=SESSION_COOKIE_SECURE,
            samesite=SESSION_COOKIE_SAMESITE
        )
        return response
    else:
        # Invalid credentials
        return RedirectResponse(url="/login?error=1", status_code=302)

@app.get("/logout")
async def logout(request: Request):
    """Handle logout"""
    user = get_current_user(request)
    if user and user in active_sessions:
        del active_sessions[user]
    
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie(SESSION_COOKIE_NAME)
    return response

@app.post("/command/{cmd}")
async def send_sensor_command_endpoint(cmd: str, user: str = Depends(require_auth)):
    """Handle sensor commands (start, stop, record)"""
    global is_running, samples_received
    
    if cmd not in VALID_COMMANDS:
        return {"status": "error", "message": "Invalid command"}
    
    if not serial_connected:
        return {"status": "error", "message": "Serial connection not available. Cannot send commands."}
    
    if cmd == "start":
        if is_running:
            return {"status": "error", "message": "Sensor is already running"}
        is_running = True
        samples_received = 0  # Reset sample counter
        # Initialize IIR filters with current sampling frequency
        initialize_iir_filters(current_sampling_freq)
        reset_iir_filters()
        result = send_sensor_command(cmd, current_frequency, current_refresh_rate)
    elif cmd == "stop":
        is_running = False
        result = send_sensor_command(cmd)
    else:
        result = send_sensor_command(cmd)
    
    # Broadcast command status to all connected clients
    await manager.broadcast(json.dumps({
        "type": "command_response",
        "command": cmd,
        "frequency": current_frequency if cmd == "start" else None,
        "refresh_rate": current_refresh_rate if cmd == "start" else None,
        "is_running": is_running,
        "result": result
    }))
    
    return result

@app.post("/configure")
async def configure_sensor(config: SensorConfig, user: str = Depends(require_auth)):
    """Configure the sensor settings (only when not running)"""
    global current_frequency, current_refresh_rate
    
    if is_running:
        raise HTTPException(
            status_code=400, 
            detail="Cannot configure sensor while it is running. Stop the sensor first."
        )
    
    if config.frequency < MIN_FREQUENCY or config.frequency > MAX_FREQUENCY:
        raise HTTPException(
            status_code=400, 
            detail=f"Frequency must be between {MIN_FREQUENCY} and {MAX_FREQUENCY} Hz"
        )
    
    if config.refresh_rate < MIN_REFRESH_RATE or config.refresh_rate > MAX_REFRESH_RATE:
        raise HTTPException(
            status_code=400, 
            detail=f"Refresh rate must be between {MIN_REFRESH_RATE} and {MAX_REFRESH_RATE} Hz"
        )
    
    current_frequency = config.frequency
    current_refresh_rate = config.refresh_rate
    
    # Broadcast configuration update to all connected clients
    await manager.broadcast(json.dumps({
        "type": "config_update",
        "frequency": current_frequency,
        "refresh_rate": current_refresh_rate
    }))
    
    return {
        "status": "success", 
        "message": f"Configuration updated: {current_frequency} Hz, {current_refresh_rate} Hz refresh",
        "frequency": current_frequency,
        "refresh_rate": current_refresh_rate
    }

@app.get("/configure")
async def get_configuration(user: str = Depends(require_auth)):
    """Get current sensor configuration"""
    return {
        "frequency": current_frequency,
        "refresh_rate": current_refresh_rate,
        "is_running": is_running,
        "min_frequency": MIN_FREQUENCY,
        "max_frequency": MAX_FREQUENCY,
        "min_refresh_rate": MIN_REFRESH_RATE,
        "max_refresh_rate": MAX_REFRESH_RATE,
        "default_frequency": DEFAULT_FREQUENCY,
        "default_refresh_rate": DEFAULT_REFRESH_RATE,
        "sampling_frequency_options": SAMPLING_FREQUENCY_OPTIONS,
        "default_sampling_frequency": DEFAULT_SAMPLING_FREQ
    }

@app.get("/plot-data")
async def get_plot_data(user: str = Depends(require_auth)):
    """Get current plot data for initial load"""
    return {
        "timestamps": plot_data['timestamps'],
        "signal1": plot_data['signal1'] if 'signal1' in plot_data else [],
        "signal2": plot_data['signal2'] if 'signal2' in plot_data else [],
        "signal3": plot_data['signal3'] if 'signal3' in plot_data else [],
        "measured_leads": plot_data['measured_leads'],
        "derived_leads": plot_data['derived_leads'],
        "filtered_leads": plot_data['filtered_leads'],
        "frequency": current_frequency,
        "refresh_rate": current_refresh_rate,
        "heart_rate": plot_data['heart_rate']
    }

@app.get("/status")
async def get_status(user: str = Depends(require_auth)):
    """Get current sensor and connection status"""
    recording_status = get_recording_status()
    
    return {
        "serial_status": "connected" if serial_connected else "disconnected",
        "serial_port": SERIAL_PORT,
        "baud_rate": BAUD_RATE,
        "active_connections": len(manager.active_connections),
        "current_frequency": current_frequency,
        "current_refresh_rate": current_refresh_rate,
        "is_running": is_running,
        "sampling_frequency": current_sampling_freq,
        "recording_time": current_recording_time,
        "samples_received": samples_received,
        "buffer_size": len(serial_data_buffer),
        "iir_filters_initialized": len(iir_filters) > 0,
        "recording_status": recording_status
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Note: WebSocket authentication would require additional implementation
    # For now, we'll allow WebSocket connections without authentication
    # In a production environment, you might want to implement token-based auth for WebSockets
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Recording API endpoints
@app.post("/recording/start")
async def start_recording_endpoint(config: RecordingConfig, user: str = Depends(require_auth)):
    """Start recording with specified configuration"""
    try:
        print(f"Received recording config: mode={config.mode}, duration={config.duration}, start_time={config.start_time}")
        
        success, message = start_recording(
            mode=config.mode,
            duration=config.duration,
            start_time=config.start_time
        )
        
        if success:
            return {"status": "success", "message": message}
        else:
            return {"status": "error", "message": message}
    except Exception as e:
        print(f"Recording start error: {e}")
        return {"status": "error", "message": f"Failed to start recording: {str(e)}"}

@app.post("/recording/stop")
async def stop_recording_endpoint(user: str = Depends(require_auth)):
    """Stop current recording"""
    try:
        success, message = stop_recording()
        
        if success:
            return {"status": "success", "message": message}
        else:
            return {"status": "error", "message": message}
    except Exception as e:
        return {"status": "error", "message": f"Failed to stop recording: {str(e)}"}

@app.get("/recording/status")
async def get_recording_status_endpoint(user: str = Depends(require_auth)):
    """Get current recording status"""
    try:
        status = get_recording_status()
        return {
            "status": "success",
            "recording_status": status
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to get recording status: {str(e)}"}

@app.post("/recording/export")
async def export_recording_endpoint(request: ExportRequest, user: str = Depends(require_auth)):
    """Export recorded data to specified format"""
    try:
        success, message = export_recording(
            format_type=request.format,
            filename=request.filename,
            bdf_metadata=request.bdf_metadata
        )
        
        if success:
            return {"status": "success", "message": message}
        else:
            return {"status": "error", "message": message}
    except Exception as e:
        return {"status": "error", "message": f"Failed to export recording: {str(e)}"}

@app.get("/recording/data")
async def get_recording_data_endpoint(user: str = Depends(require_auth)):
    """Get recorded data for playback"""
    try:
        global recorded_data
        
        if not recorded_data['timestamps']:
            return {"status": "error", "message": "No recorded data available"}
        
        # Return a subset of data for efficient transmission
        # In a real application, you might want to implement pagination
        max_points = 1000  # Limit data points for transmission
        
        if len(recorded_data['timestamps']) > max_points:
            # Downsample data
            step = len(recorded_data['timestamps']) // max_points
            indices = list(range(0, len(recorded_data['timestamps']), step))
            
            playback_data = {
                'timestamps': [recorded_data['timestamps'][i] for i in indices],
                'filtered_leads': {}
            }
            
            for lead in ALL_LEADS:
                playback_data['filtered_leads'][lead] = [
                    recorded_data['filtered_leads'][lead][i] 
                    for i in indices 
                    if i < len(recorded_data['filtered_leads'][lead])
                ]
        else:
            playback_data = {
                'timestamps': recorded_data['timestamps'],
                'filtered_leads': recorded_data['filtered_leads']
            }
        
        return {
            "status": "success",
            "data": playback_data,
            "metadata": recorded_data['metadata']
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to get recording data: {str(e)}"}

@app.get("/recording/download-csv")
async def download_csv(user: str = Depends(require_auth)):
    """Download the latest recorded data as a CSV file with a unique name."""
    global recorded_data
    import csv
    import time
    if not recorded_data['timestamps']:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="No recorded data available")
    # Generate unique filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"ecg_recording_{timestamp}.csv"
    # Write CSV to in-memory buffer
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    header = ['Timestamp'] + ALL_LEADS
    writer.writerow(header)
    for i, ts in enumerate(recorded_data['timestamps']):
        row = [ts]
        for lead in ALL_LEADS:
            if i < len(recorded_data['filtered_leads'][lead]):
                row.append(recorded_data['filtered_leads'][lead][i])
            else:
                row.append('')
        writer.writerow(row)
    buffer.seek(0)
    # Return as downloadable file
    return StreamingResponse(buffer, media_type='text/csv', headers={
        'Content-Disposition': f'attachment; filename="{filename}"'
    })

@app.on_event("startup")
async def startup_event():
    """Initialize serial connection on startup and start data generation task"""
    print("Starting ECG Sensor Control Server...")
    print(f"Serial port: {SERIAL_PORT}")
    print(f"Baud rate: {BAUD_RATE}")
    print(f"Default frequency: {DEFAULT_FREQUENCY} Hz")
    print(f"Default refresh rate: {DEFAULT_REFRESH_RATE} Hz")
    print(f"Default sampling frequency: {DEFAULT_SAMPLING_FREQ} Hz")
    print(f"Available sampling frequencies: {SAMPLING_FREQUENCY_OPTIONS} Hz")
    print(f"Sample size: {SAMPLE_SIZE} bytes")
    print(f"ECG leads: {ECG_LEADS}")
    print(f"Derived leads: {DERIVED_LEADS}")
    print(f"All leads (12 channels): {ALL_LEADS}")
    print(f"IIR filter HPF cutoff: {IIRFilterConfig.HPF_CUTOFF} Hz")
    print(f"IIR filter LPF cutoff: {IIRFilterConfig.LPF_CUTOFF} Hz")
    print(f"IIR filter order: {IIRFilterConfig.FILTER_ORDER}")
    
    if init_serial():
        print(f"Serial connection established on {SERIAL_PORT}")
    else:
        print("Warning: Serial connection failed. Commands will not be sent to sensor.")
    
    # Start the background tasks
    asyncio.create_task(data_generation_task())
    asyncio.create_task(serial_reading_task())
    print("Background tasks started: data generation and serial reading")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    close_serial()

# Recording functions
def start_recording(mode: str, duration: int = None, start_time: str = None):
    """Start recording with specified mode and parameters"""
    global is_recording, recording_mode, recording_start_time, recording_duration, recorded_data, recording_scheduled_start, recording_task
    
    # Check if we have a scheduled recording that needs to be cancelled
    if recording_mode == 'interval' and recording_scheduled_start and not is_recording:
        # Cancel the scheduled recording
        recording_scheduled_start = None
        recording_mode = None
        recording_duration = None
        print("Previous scheduled recording cancelled")
    
    if is_recording:
        return False, "Recording already in progress"
    
    if not is_running:
        return False, "Cannot start recording: Data reading is not active. Please start the sensor first."
    
    if mode not in ['interval', 'manual']:
        return False, "Invalid recording mode"
    
    # Clear previous recording data
    recorded_data = {
        'timestamps': [],
        'measured_leads': {lead: [] for lead in ECG_LEADS},
        'derived_leads': {lead: [] for lead in DERIVED_LEADS},
        'filtered_leads': {lead: [] for lead in ALL_LEADS},
        'metadata': {
            'start_time': None,
            'end_time': None,
            'duration': None,
            'sampling_frequency': current_sampling_freq,
            'mode': mode
        }
    }
    
    recording_mode = mode
    recording_duration = duration
    
    if mode == 'manual':
        # Manual recording starts immediately
        is_recording = True
        recording_start_time = time.time()
        recording_scheduled_start = None
        
        # Set metadata
        recorded_data['metadata']['start_time'] = recording_start_time
        recorded_data['metadata']['sampling_frequency'] = current_sampling_freq
        recorded_data['metadata']['mode'] = mode
        
        print(f"Manual recording started immediately")
        return True, "Manual recording started immediately"
    
    elif mode == 'interval':
        # Interval recording - check if we need to schedule a start time
        if start_time:
            # Parse the start time
            try:
                scheduled_start = time.time()
                if start_time != 'now':
                    # Calculate delay from start_time (which is an ISO timestamp)
                    from datetime import datetime
                    start_datetime = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    scheduled_start = start_datetime.timestamp()
                
                current_time = time.time()
                delay_seconds = scheduled_start - current_time
                
                if delay_seconds > 0:
                    # Schedule recording to start later
                    recording_scheduled_start = scheduled_start
                    is_recording = False
                    recording_start_time = None
                    
                    print(f"Interval recording scheduled to start in {delay_seconds:.1f} seconds")
                    return True, f"Interval recording scheduled to start in {delay_seconds:.1f} seconds"
                else:
                    # Start immediately if scheduled time has passed
                    is_recording = True
                    recording_start_time = time.time()
                    recording_scheduled_start = None
                    
                    # Set metadata
                    recorded_data['metadata']['start_time'] = recording_start_time
                    recorded_data['metadata']['sampling_frequency'] = current_sampling_freq
                    recorded_data['metadata']['mode'] = mode
                    
                    print(f"Interval recording started immediately (scheduled time has passed)")
                    return True, "Interval recording started immediately"
                    
            except Exception as e:
                return False, f"Invalid start time format: {str(e)}"
        else:
            # No start time specified, start immediately
            is_recording = True
            recording_start_time = time.time()
            recording_scheduled_start = None
            
            # Set metadata
            recorded_data['metadata']['start_time'] = recording_start_time
            recorded_data['metadata']['sampling_frequency'] = current_sampling_freq
            recorded_data['metadata']['mode'] = mode
            
            print(f"Interval recording started immediately")
            return True, "Interval recording started immediately"

def stop_recording():
    """Stop current recording"""
    global is_recording, recording_start_time, recorded_data, recording_scheduled_start, recording_mode
    
    # Check if we have a scheduled recording that hasn't started yet
    if recording_mode == 'interval' and recording_scheduled_start and not is_recording:
        recording_scheduled_start = None
        recording_mode = None
        recording_duration = None
        print("Scheduled recording cancelled")
        return True, "Scheduled recording cancelled"
    
    if not is_recording:
        return False, "No recording in progress"
    
    recording_end_time = time.time()
    duration = recording_end_time - recording_start_time
    
    # Update metadata
    recorded_data['metadata']['end_time'] = recording_end_time
    recorded_data['metadata']['duration'] = duration
    
    is_recording = False
    recording_start_time = None
    recording_scheduled_start = None
    
    print(f"Recording stopped. Duration: {duration:.2f} seconds")
    return True, f"Recording stopped. Duration: {duration:.2f} seconds"

def add_to_recording(timestamp: float, lead_data: dict, derived_leads: dict, filtered_leads: dict):
    """Add data point to current recording"""
    global recorded_data, is_recording, recording_start_time, recording_scheduled_start
    
    # Check if we need to start recording (for scheduled interval recording)
    if recording_mode == 'interval' and recording_scheduled_start and not is_recording:
        current_time = time.time()
        if current_time >= recording_scheduled_start:
            # Start recording now
            is_recording = True
            recording_start_time = current_time
            recording_scheduled_start = None
            
            # Set metadata
            recorded_data['metadata']['start_time'] = recording_start_time
            recorded_data['metadata']['sampling_frequency'] = current_sampling_freq
            recorded_data['metadata']['mode'] = recording_mode
            
            print(f"Interval recording started after countdown")
    
    if not is_recording:
        return
    
    # Check if interval recording should stop based on duration
    if recording_mode == 'interval' and recording_duration and recording_start_time:
        elapsed = time.time() - recording_start_time
        if elapsed >= recording_duration:
            stop_recording()
            return
    
    # Add data to recording
    recorded_data['timestamps'].append(timestamp)
    
    for lead in ECG_LEADS:
        if lead in lead_data:
            recorded_data['measured_leads'][lead].append(lead_data[lead])
    
    for lead in DERIVED_LEADS:
        if lead in derived_leads:
            recorded_data['derived_leads'][lead].append(derived_leads[lead])
    
    for lead in ALL_LEADS:
        if lead in filtered_leads:
            recorded_data['filtered_leads'][lead].append(filtered_leads[lead])

def get_recording_status():
    """Get current recording status"""
    global is_recording, recording_mode, recording_start_time, recording_duration, recorded_data, recording_scheduled_start
    
    data_points = len(recorded_data['timestamps'])
    
    if not is_recording:
        # Check if we have a scheduled recording
        if recording_mode == 'interval' and recording_scheduled_start:
            current_time = time.time()
            countdown = recording_scheduled_start - current_time
            if countdown > 0:
                return {
                    'is_recording': False,
                    'mode': recording_mode,
                    'elapsed': 0,
                    'duration': recording_duration,
                    'data_points': data_points,
                    'scheduled_start': recording_scheduled_start,
                    'countdown': countdown
                }
        
        return {
            'is_recording': False,
            'mode': recorded_data['metadata']['mode'] if recorded_data['metadata']['mode'] else None,
            'elapsed': 0,
            'duration': recorded_data['metadata']['duration'],
            'data_points': data_points
        }
    
    elapsed = time.time() - recording_start_time
    
    return {
        'is_recording': True,
        'mode': recording_mode,
        'elapsed': elapsed,
        'duration': recording_duration,
        'data_points': data_points
    }

def export_recording(format_type: str, filename: str = None, bdf_metadata: dict = None):
    """Export recorded data to CSV or BDF format"""
    global recorded_data
    
    if not recorded_data['timestamps']:
        return False, "No recorded data to export"
    
    if format_type not in ['csv', 'bdf']:
        return False, "Unsupported export format"
    
    if not filename:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"ecg_recording_{timestamp}.{format_type}"
    
    try:
        if format_type == 'csv':
            return export_to_csv(filename)
        elif format_type == 'bdf':
            return export_to_bdf(filename, bdf_metadata)
    except Exception as e:
        return False, f"Export error: {str(e)}"

def export_to_csv(filename: str):
    """Export recorded data to CSV format"""
    import csv
    import os
    
    filepath = os.path.join('recordings', filename)
    os.makedirs('recordings', exist_ok=True)
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        header = ['Timestamp'] + ALL_LEADS
        writer.writerow(header)
        
        # Write data
        for i, timestamp in enumerate(recorded_data['timestamps']):
            row = [timestamp]
            for lead in ALL_LEADS:
                if i < len(recorded_data['filtered_leads'][lead]):
                    row.append(recorded_data['filtered_leads'][lead][i])
                else:
                    row.append('')
            writer.writerow(row)
    
    return True, f"Data exported to {filepath}"

def export_to_bdf(filename: str, bdf_metadata: dict = None):
    """Export recorded data to BDF format using pyedflib"""
    import os
    import pyedflib
    from datetime import datetime
    
    filepath = os.path.join('recordings', filename)
    os.makedirs('recordings', exist_ok=True)
    
    try:
        # Get recording metadata
        start_time = recorded_data['metadata']['start_time']
        duration = recorded_data['metadata']['duration']
        sampling_freq = recorded_data['metadata']['sampling_frequency']
        
        # Convert start time to datetime
        if start_time:
            start_datetime = datetime.fromtimestamp(start_time)
        else:
            start_datetime = datetime.now()
        
        # Calculate number of samples
        num_samples = len(recorded_data['timestamps'])
        if num_samples == 0:
            return False, "No data to export"
        
        # Create BDF file
        f = pyedflib.EdfWriter(filepath, len(ALL_LEADS))
        
        # Convert birthdate format if provided
        birthdate = ''
        if bdf_metadata and bdf_metadata.get('birthdate'):
            try:
                # Convert from ISO format (YYYY-MM-DD) to BDF format (DD MMM YYYY)
                date_obj = datetime.strptime(bdf_metadata['birthdate'], '%Y-%m-%d')
                birthdate = date_obj.strftime('%d %b %Y')
            except ValueError:
                # If conversion fails, use empty string
                birthdate = ''
        
        # Set header information with user-provided metadata or defaults
        header = {
            'technician': bdf_metadata.get('technician', 'ECG Sensor') if bdf_metadata else 'ECG Sensor',
            'recording_additional': bdf_metadata.get('recording_additional', 'ECG recording') if bdf_metadata else 'ECG recording',
            'patientname': bdf_metadata.get('patientname', 'Test Patient') if bdf_metadata else 'Test Patient',
            'patient_additional': bdf_metadata.get('patient_additional', '') if bdf_metadata else '',
            'patientcode': bdf_metadata.get('patientcode', '') if bdf_metadata else '',
            'birthdate': birthdate,
            'gender': bdf_metadata.get('gender', '') if bdf_metadata else '',
            'patient_weight': bdf_metadata.get('patient_weight', '') if bdf_metadata else '',
            'patient_height': bdf_metadata.get('patient_height', '') if bdf_metadata else '',
            'patient_comment': bdf_metadata.get('patient_comment', '') if bdf_metadata else '',
            'admincode': bdf_metadata.get('admincode', '') if bdf_metadata else '',
            'equipment': bdf_metadata.get('equipment', 'ECG Sensor') if bdf_metadata else 'ECG Sensor',
            'hospital_additional': bdf_metadata.get('hospital_additional', '') if bdf_metadata else '',
            'hospital': bdf_metadata.get('hospital', '') if bdf_metadata else '',
            'department_additional': bdf_metadata.get('department_additional', '') if bdf_metadata else '',
            'department': bdf_metadata.get('department', '') if bdf_metadata else '',
            'startdate': start_datetime,
        }
        
        f.setHeader(header)
        
        # Set minimal channel information for testing
        channel_info = []
        for lead in ALL_LEADS:
            info = {
                'label': lead,
                'dimension': 'mV',
                'sample_rate': sampling_freq,
                'physical_max': 1000.0,
                'physical_min': -1000.0,
                'digital_max': 32767,
                'digital_min': -32768,
                'prefilter': '',
                'transducer': ''
            }
            channel_info.append(info)
        
        f.setSignalHeaders(channel_info)
        
        # Prepare data for each channel
        import numpy as np
        
        # Write all channels at once for better performance
        all_data = []
        for lead in ALL_LEADS:
            lead_data = recorded_data['filtered_leads'][lead]
            
            # Ensure all channels have the same length
            if len(lead_data) < num_samples:
                # Pad with zeros if necessary
                lead_data = lead_data + [0.0] * (num_samples - len(lead_data))
            elif len(lead_data) > num_samples:
                # Truncate if necessary
                lead_data = lead_data[:num_samples]
            
            # Convert to numpy array
            data_array = np.array(lead_data, dtype=np.float64)
            all_data.append(data_array)
        
        # Write all samples at once
        f.writeSamples(all_data)
        
        # Close the file
        f.close()
        
        return True, f"Data exported to {filepath} (BDF format)"
        
    except ImportError:
        return False, "pyedflib library not available. Please install it with: pip install pyedflib"
    except Exception as e:
        return False, f"BDF export error: {str(e)}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT) 