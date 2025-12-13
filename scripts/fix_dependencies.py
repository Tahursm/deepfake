"""
Script to fix dependency conflicts, particularly protobuf issues.
"""

import subprocess
import sys


def run_command(cmd):
    """Run a shell command and return success status."""
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return False


def main():
    print("=" * 60)
    print("Fixing Dependency Conflicts")
    print("=" * 60)
    
    # Check if mysql-connector-python is installed
    print("\n1. Checking for mysql-connector-python...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", "mysql-connector-python"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("   Found mysql-connector-python (not needed for this project)")
        response = input("   Uninstall it? (y/n): ").strip().lower()
        if response == 'y':
            print("   Uninstalling mysql-connector-python...")
            run_command(f"{sys.executable} -m pip uninstall mysql-connector-python -y")
        else:
            print("   Keeping mysql-connector-python (may cause protobuf conflicts)")
    else:
        print("   mysql-connector-python not found (good)")
    
    # Fix protobuf version
    print("\n2. Fixing protobuf version...")
    print("   Installing compatible protobuf version...")
    run_command(f"{sys.executable} -m pip install 'protobuf>=4.21.0,<4.26.0'")
    
    # Reinstall requirements to ensure compatibility
    print("\n3. Reinstalling requirements with fixed versions...")
    run_command(f"{sys.executable} -m pip install -r requirements.txt --upgrade")
    
    print("\n" + "=" * 60)
    print("Dependency fix complete!")
    print("=" * 60)
    print("\nIf you still see warnings, they are usually non-critical.")
    print("The packages should work despite the warnings.")


if __name__ == '__main__':
    main()

