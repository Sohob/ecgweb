#!/usr/bin/env python3
"""
Test script for ECG Sensor Control WebServer
Tests the server functionality without requiring a physical sensor
"""

import requests
import json
import time
import sys
from urllib.parse import urljoin

def test_server(base_url="http://localhost:8000"):
    """Test the server endpoints"""
    
    print("🧪 Testing ECG Sensor Control WebServer...")
    print(f"📍 Server URL: {base_url}")
    print()
    
    # Test 1: Check if server is running
    print("1️⃣ Testing server availability...")
    try:
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print("✅ Server is running and responding")
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        print("💡 Make sure the server is running with: python main.py")
        return False
    
    # Test 2: Check status endpoint
    print("\n2️⃣ Testing status endpoint...")
    try:
        response = requests.get(urljoin(base_url, "/status"), timeout=5)
        if response.status_code == 200:
            status = response.json()
            print("✅ Status endpoint working")
            print(f"   Serial status: {status['serial_status']}")
            print(f"   Serial port: {status['serial_port']}")
            print(f"   Active connections: {status['active_connections']}")
        else:
            print(f"❌ Status endpoint returned: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing status endpoint: {e}")
        return False
    
    # Test 3: Test command endpoints
    print("\n3️⃣ Testing command endpoints...")
    commands = ["start", "stop", "record"]
    
    for cmd in commands:
        try:
            response = requests.post(urljoin(base_url, f"/command/{cmd}"), timeout=5)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Command '{cmd}': {result['status']} - {result['message']}")
            else:
                print(f"❌ Command '{cmd}' returned: {response.status_code}")
        except Exception as e:
            print(f"❌ Error testing command '{cmd}': {e}")
    
    # Test 4: Test invalid command
    print("\n4️⃣ Testing invalid command...")
    try:
        response = requests.post(urljoin(base_url, "/command/invalid"), timeout=5)
        if response.status_code == 200:
            result = response.json()
            if result['status'] == 'error':
                print("✅ Invalid command properly rejected")
            else:
                print("❌ Invalid command should have been rejected")
        else:
            print(f"❌ Invalid command test returned: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing invalid command: {e}")
    
    print("\n🎉 All tests completed!")
    print("\n📱 You can now open your browser and go to:")
    print(f"   {base_url}")
    print("\n🔧 To test with a real sensor:")
    print("   1. Connect your sensor to the serial port")
    print("   2. Update the SERIAL_PORT in config.py")
    print("   3. Restart the server")
    
    return True

def main():
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:8000"
    
    success = test_server(base_url)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 