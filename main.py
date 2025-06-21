from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import serial
import json
import asyncio
import time
import numpy as np
import struct
from typing import List, Dict, Any
import os
from config import *
from scipy import signal
from collections import deque

app = FastAPI(title="ECG Sensor Control Server", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Serial connection (will be initialized when needed)
serial_connection = None
serial_connected = False

# Current settings and state
current_frequency = DEFAULT_FREQUENCY
current_refresh_rate = DEFAULT_REFRESH_RATE
is_running = False
current_sampling_freq = DEFAULT_SAMPLING_FREQ
current_recording_time = CONTINUOUS_RECORDING

# Data reading settings
SAMPLE_SIZE = 27  # 27 bytes per sample from sensor
serial_data_buffer = bytearray()
last_sample_time = 0
samples_received = 0

# ECG Data Processing Constants (placeholders - adjust based on your hardware)
class ECGConstants:
    REFERENCE_VOLTAGE_MV = 2500.0  # Reference voltage in mV
    OFFSET_LSB_STEPS = 0  # Offset in LSB steps
    ADC_RESOLUTION = 24  # ADC resolution in bits
    MAX_ADC_VALUE = (2 ** 24) - 1  # Maximum ADC value

# ECG Lead Names
ECG_LEADS = ['Lead1', 'Lead2', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
DERIVED_LEADS = ['Lead3', 'aVL', 'aVR', 'aVF']
ALL_LEADS = ECG_LEADS + DERIVED_LEADS  # All 12 channels

# FIR Filter Configuration
class FIRFilterConfig:
    CUTOFF_FREQ = 150.0  # Cutoff frequency in Hz
    FILTER_ORDER = 41    # Filter order (number of coefficients)
    NYQUIST_FREQ = 500.0 # Nyquist frequency (should be > 2 * cutoff_freq)

# FIR Filter Implementation
class OnlineFIRFilter:
    """Online FIR filter implementation for real-time processing"""
    
    def __init__(self, coefficients: np.ndarray):
        self.coefficients = coefficients
        self.buffer = deque(maxlen=len(coefficients))
        # Initialize buffer with zeros
        for _ in range(len(coefficients)):
            self.buffer.append(0.0)
    
    def filter(self, input_value: float) -> float:
        """Apply FIR filter to a single input value"""
        # Add new input to buffer
        self.buffer.append(input_value)
        
        # Apply convolution
        result = 0.0
        for i, coeff in enumerate(self.coefficients):
            result += coeff * list(self.buffer)[-(i+1)]
        
        return result
    
    def reset(self):
        """Reset filter buffer"""
        self.buffer.clear()
        for _ in range(len(self.coefficients)):
            self.buffer.append(0.0)

# Initialize FIR filters for all channels
fir_filters = {}

def initialize_fir_filters(sampling_rate: float):
    """Initialize FIR low-pass filters for all ECG channels"""
    global fir_filters
    
    # Design FIR low-pass filter
    # Normalize cutoff frequency to Nyquist frequency
    nyquist_freq = sampling_rate / 2.0
    normalized_cutoff = FIRFilterConfig.CUTOFF_FREQ / nyquist_freq
    
    # Ensure cutoff frequency is valid
    if normalized_cutoff >= 1.0:
        print(f"Warning: Cutoff frequency {FIRFilterConfig.CUTOFF_FREQ} Hz is too high for sampling rate {sampling_rate} Hz")
        normalized_cutoff = 0.9  # Use 90% of Nyquist frequency as fallback
    
    # Design FIR filter using scipy
    coefficients = signal.firwin(
        numtaps=FIRFilterConfig.FILTER_ORDER,
        cutoff=normalized_cutoff,
        window='hamming',
        pass_zero='lowpass'
    )
    
    print(f"FIR Filter Design:")
    print(f"  Sampling Rate: {sampling_rate} Hz")
    print(f"  Cutoff Frequency: {FIRFilterConfig.CUTOFF_FREQ} Hz")
    print(f"  Normalized Cutoff: {normalized_cutoff:.3f}")
    print(f"  Filter Order: {FIRFilterConfig.FILTER_ORDER}")
    print(f"  Number of Coefficients: {len(coefficients)}")
    
    # Create filters for all channels
    fir_filters.clear()
    for lead_name in ALL_LEADS:
        fir_filters[lead_name] = OnlineFIRFilter(coefficients)
    
    print(f"Initialized FIR filters for {len(ALL_LEADS)} channels: {ALL_LEADS}")

def reset_fir_filters():
    """Reset all FIR filters"""
    global fir_filters
    for filter_obj in fir_filters.values():
        filter_obj.reset()
    print("All FIR filters reset")

# Plotting data storage - now includes filtered data
plot_data = {
    'timestamps': [],
    'measured_leads': {lead: [] for lead in ECG_LEADS},
    'derived_leads': {lead: [] for lead in DERIVED_LEADS},
    'filtered_leads': {lead: [] for lead in ALL_LEADS},  # Filtered data for all leads
    'raw_samples': []  # Store raw sample data for debugging
}
max_data_points = 1000  # Maximum number of points to keep in memory

# Pydantic model for configuration
class SensorConfig(BaseModel):
    frequency: int
    refresh_rate: int

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
    value = (byte1 << 16) | (byte2 << 8) | byte3
    
    # Handle 24-bit signed value (sign bit is the 23rd bit)
    if value & 0x800000:  # If sign bit is set
        value = value - 0x1000000  # Convert to negative value
    
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

def apply_fir_filters(lead_data: Dict[str, float], derived_leads: Dict[str, float]) -> Dict[str, float]:
    """Apply FIR filters to all leads and return filtered values"""
    global fir_filters
    
    filtered_data = {}
    
    # Apply filters to measured leads
    for lead_name in ECG_LEADS:
        if lead_name in fir_filters:
            filtered_data[lead_name] = fir_filters[lead_name].filter(lead_data[lead_name])
        else:
            filtered_data[lead_name] = lead_data[lead_name]  # No filter available
    
    # Apply filters to derived leads
    for lead_name in DERIVED_LEADS:
        if lead_name in fir_filters:
            filtered_data[lead_name] = fir_filters[lead_name].filter(derived_leads[lead_name])
        else:
            filtered_data[lead_name] = derived_leads[lead_name]  # No filter available
    
    return filtered_data

def process_sensor_sample(sample_data: bytes, timestamp: float):
    """Process a complete 27-byte sample from the sensor"""
    global samples_received, plot_data
    
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
            voltage = get_voltage_from_raw_data(raw_value, gain=1)
            lead_data[lead_name] = voltage
        
        # Calculate derived leads
        derived_leads = calculate_derived_leads(lead_data['Lead1'], lead_data['Lead2'])
        
        # Apply FIR filters to all leads
        filtered_leads = apply_fir_filters(lead_data, derived_leads)
        
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
        
        # Print processed data for debugging (every 100th sample to avoid spam)
        if samples_received % 100 == 0:
            print(f"Sample #{samples_received} at {timestamp:.3f}s:")
            print(f"  Lead1: {lead_data['Lead1']:.3f} mV -> {filtered_leads['Lead1']:.3f} mV (filtered)")
            print(f"  Lead2: {lead_data['Lead2']:.3f} mV -> {filtered_leads['Lead2']:.3f} mV (filtered)")
            print(f"  V1: {lead_data['V1']:.3f} mV -> {filtered_leads['V1']:.3f} mV (filtered)")
            print(f"  Lead3: {derived_leads['Lead3']:.3f} mV -> {filtered_leads['Lead3']:.3f} mV (filtered)")
        
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
    while True:
        if is_running:
            # Check if we have real sensor data
            if len(plot_data['timestamps']) > 0 and len(plot_data['filtered_leads']['Lead1']) > 0:
                # Use real sensor data (filtered)
                latest_index = -1
                timestamp = plot_data['timestamps'][latest_index]
                
                # Get latest filtered leads for plotting
                lead1 = plot_data['filtered_leads']['Lead1'][latest_index]
                lead2 = plot_data['filtered_leads']['Lead2'][latest_index]
                v1 = plot_data['filtered_leads']['V1'][latest_index]
                lead3 = plot_data['filtered_leads']['Lead3'][latest_index]
                avl = plot_data['filtered_leads']['aVL'][latest_index]
                avf = plot_data['filtered_leads']['aVF'][latest_index]
                
                plot_update = {
                    "type": "plot_data",
                    "timestamp": timestamp,
                    "signal1": float(lead1),
                    "signal2": float(lead2),
                    "signal3": float(v1),
                    "lead3": float(lead3),
                    "avl": float(avl),
                    "avf": float(avf),
                    "frequency": current_frequency,
                    "refresh_rate": current_refresh_rate,
                    "samples_received": samples_received,
                    "buffer_size": len(serial_data_buffer)
                }
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
                    "buffer_size": len(serial_data_buffer)
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
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/command/{cmd}")
async def send_sensor_command_endpoint(cmd: str):
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
        
        # Initialize FIR filters with current sampling frequency
        initialize_fir_filters(current_sampling_freq)
        reset_fir_filters()
        
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
async def configure_sensor(config: SensorConfig):
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
async def get_configuration():
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
        "default_refresh_rate": DEFAULT_REFRESH_RATE
    }

@app.get("/plot-data")
async def get_plot_data():
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
        "refresh_rate": current_refresh_rate
    }

@app.get("/status")
async def get_status():
    """Get current sensor and connection status"""
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
        "fir_filters_initialized": len(fir_filters) > 0
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            # Echo back for now, can be extended for bidirectional communication
            await manager.send_personal_message(f"Message received: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.on_event("startup")
async def startup_event():
    """Initialize serial connection on startup and start data generation task"""
    print("Starting ECG Sensor Control Server...")
    print(f"Serial port: {SERIAL_PORT}")
    print(f"Baud rate: {BAUD_RATE}")
    print(f"Default frequency: {DEFAULT_FREQUENCY} Hz")
    print(f"Default refresh rate: {DEFAULT_REFRESH_RATE} Hz")
    print(f"Default sampling frequency: {DEFAULT_SAMPLING_FREQ} Hz")
    print(f"Sample size: {SAMPLE_SIZE} bytes")
    print(f"ECG leads: {ECG_LEADS}")
    print(f"Derived leads: {DERIVED_LEADS}")
    print(f"All leads (12 channels): {ALL_LEADS}")
    print(f"FIR filter cutoff: {FIRFilterConfig.CUTOFF_FREQ} Hz")
    print(f"FIR filter order: {FIRFilterConfig.FILTER_ORDER}")
    
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT) 