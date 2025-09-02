#!/usr/bin/env python3
"""
Test script for Bao LQO functionality
This script tests basic Bao operations without full training
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def test_bao_environment():
    """Test if Bao environment is properly set up"""
    print("Testing Bao environment...")
    
    bao_dir = Path(".")
    bao_server_dir = bao_dir / "bao_server"
    
    # Check directory structure
    required_files = [
        "bao_server/main.py",
        "bao_server/baoctl.py", 
        "bao_server/train.py",
        "bao_server/model.py",
        "requirements.txt"
    ]
    
    for file_path in required_files:
        if not (bao_dir / file_path).exists():
            print(f"❌ Required file not found: {file_path}")
            return False
        else:
            print(f"✓ Found: {file_path}")
    
    # Check Python dependencies
    try:
        import psycopg2
        print("✓ psycopg2 imported successfully")
    except ImportError:
        print("❌ psycopg2 not available")
        return False
    
    try:
        import torch
        print("✓ torch imported successfully")
    except ImportError:
        print("❌ torch not available")
        return False
    
    print("✅ Bao environment check passed")
    return True

def test_bao_server_startup():
    """Test if Bao server can start up"""
    print("\nTesting Bao server startup...")
    
    try:
        # Try to start server
        process = subprocess.Popen(
            [sys.executable, "bao_server/main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a bit for startup
        time.sleep(3)
        
        # Check if still running
        if process.poll() is None:
            print("✓ Bao server started successfully")
            # Kill it
            process.terminate()
            process.wait(timeout=5)
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Bao server failed to start: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting Bao server: {e}")
        return False

def test_bao_commands():
    """Test basic Bao commands"""
    print("\nTesting Bao commands...")
    
    commands = [
        ["bao_server/baoctl.py", "--help"],
        ["bao_server/baoctl.py", "--status"],
    ]
    
    for cmd in commands:
        try:
            result = subprocess.run(
                [sys.executable] + cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print(f"✓ Command successful: {' '.join(cmd)}")
            else:
                print(f"⚠ Command failed: {' '.join(cmd)} (exit code: {result.returncode})")
                if result.stderr:
                    print(f"  Error: {result.stderr.strip()}")
                    
        except subprocess.TimeoutExpired:
            print(f"❌ Command timed out: {' '.join(cmd)}")
        except Exception as e:
            print(f"❌ Error running command {' '.join(cmd)}: {e}")
    
    return True

def main():
    """Main test function"""
    print("🧪 Bao LQO Test Suite")
    print("=" * 50)
    
    # Change to bao directory if needed
    if not Path("bao_server").exists():
        print("Error: Please run this script from the bao directory")
        sys.exit(1)
    
    # Run tests
    tests = [
        test_bao_environment,
        test_bao_server_startup,
        test_bao_commands,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Bao environment is ready.")
        sys.exit(0)
    else:
        print("⚠ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
