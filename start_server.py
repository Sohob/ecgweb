#!/usr/bin/env python3
"""
Startup script for ECG Sensor Control WebServer
Provides better error handling and configuration options
"""

import sys
import os
import argparse
import uvicorn
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'pyserial',
        'websockets',
        'jinja2'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall them with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def check_files():
    """Check if all required files exist"""
    required_files = [
        'main.py',
        'templates/index.html',
        'requirements.txt'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ All required files are present")
    return True

def main():
    parser = argparse.ArgumentParser(description='Start ECG Sensor Control WebServer')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to (default: 8000)')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    parser.add_argument('--skip-checks', action='store_true', help='Skip dependency and file checks')
    
    args = parser.parse_args()
    
    print("🚀 Starting ECG Sensor Control WebServer...")
    print(f"📍 Host: {args.host}")
    print(f"🔌 Port: {args.port}")
    print(f"🔄 Auto-reload: {'Yes' if args.reload else 'No'}")
    print()
    
    if not args.skip_checks:
        print("🔍 Running pre-flight checks...")
        
        if not check_dependencies():
            sys.exit(1)
        
        if not check_files():
            sys.exit(1)
        
        print("✅ All checks passed!")
        print()
    
    try:
        print("🌐 Starting server...")
        print(f"📱 Web interface will be available at: http://{args.host}:{args.port}")
        print("⏹️  Press Ctrl+C to stop the server")
        print()
        
        uvicorn.run(
            "main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 