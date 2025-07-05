#!/usr/bin/env python3
"""
Simple script to change the default password in auth_config.py
Run this script to update the password for better security.
"""

import getpass
import re
from pathlib import Path

def is_strong_password(password):
    """Check if password meets security requirements"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    
    if not re.search(r"\d", password):
        return False, "Password must contain at least one number"
    
    return True, "Password is strong"

def change_password():
    """Interactive password change function"""
    print("🔐 ECG Sensor Control - Password Change")
    print("=" * 40)
    print()
    
    # Get new password
    while True:
        new_password = getpass.getpass("Enter new password: ")
        confirm_password = getpass.getpass("Confirm new password: ")
        
        if new_password != confirm_password:
            print("❌ Passwords do not match. Please try again.")
            continue
        
        # Check password strength
        is_strong, message = is_strong_password(new_password)
        if not is_strong:
            print(f"❌ {message}")
            continue
        
        break
    
    # Read current auth_config.py
    auth_config_path = Path("auth_config.py")
    if not auth_config_path.exists():
        print("❌ auth_config.py not found!")
        return
    
    try:
        with open(auth_config_path, 'r') as f:
            content = f.read()
        
        # Update the password line
        import re
        new_content = re.sub(
            r'DEFAULT_PASSWORD = ".*?"',
            f'DEFAULT_PASSWORD = "{new_password}"',
            content
        )
        
        # Write back to file
        with open(auth_config_path, 'w') as f:
            f.write(new_content)
        
        print("✅ Password updated successfully!")
        print("🔒 Please restart the server for changes to take effect.")
        
    except Exception as e:
        print(f"❌ Error updating password: {e}")

if __name__ == "__main__":
    change_password() 