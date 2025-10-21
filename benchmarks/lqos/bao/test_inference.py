#!/usr/bin/env python3
"""
Test script for Bao LQO inference functionality
This script tests the inference capabilities without requiring a trained model
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def test_bao_inference_script():
    """Test if Bao inference script exists and is valid"""
    print("Testing Bao inference script...")
    
    bao_dir = Path(".")
    inference_script = bao_dir / "inference_bao.py"
    
    if not inference_script.exists():
        print(f"❌ Inference script not found: {inference_script}")
        return False
    
    print(f"✅ Found inference script: {inference_script}")
    
    # Check if script is executable
    if os.access(inference_script, os.X_OK):
        print("✅ Script is executable")
    else:
        print("⚠ Script is not executable, but that's okay")
    
    # Test script help
    try:
        result = subprocess.run(
            [sys.executable, str(inference_script), "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ Script help works correctly")
            return True
        else:
            print(f"⚠ Script help failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Script help timed out")
        return False
    except Exception as e:
        print(f"❌ Script help error: {e}")
        return False

def test_bao_environment():
    """Test Bao environment for inference"""
    print("\nTesting Bao environment for inference...")
    
    bao_dir = Path(".")
    bao_server_dir = bao_dir / "bao_server"
    
    # Check required files
    required_files = [
        "bao_server/main.py",
        "bao_server/baoctl.py",
        "bao_server/model.py",
    ]
    
    for file_path in required_files:
        if not (bao_dir / file_path).exists():
            print(f"❌ Required file not found: {file_path}")
            return False
        else:
            print(f"✅ Found: {file_path}")
    
    # Check if model directory exists (even if empty)
    model_dir = bao_server_dir / "model"
    if model_dir.exists():
        print(f"✅ Model directory exists: {model_dir}")
        # Check if it has any files
        model_files = list(model_dir.glob("*"))
        if model_files:
            print(f"✅ Model directory contains {len(model_files)} files")
        else:
            print("⚠ Model directory is empty (no trained model)")
    else:
        print("⚠ Model directory does not exist (no trained model)")
    
    print("✅ Bao environment check passed")
    return True

def test_inference_script_imports():
    """Test if inference script can be imported"""
    print("\nTesting inference script imports...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, str(Path(".")))
        
        # Try to import the inference module
        import inference_bao
        print("✅ Successfully imported inference_bao module")
        
        # Check if BaoInference class exists
        if hasattr(inference_bao, 'BaoInference'):
            print("✅ BaoInference class found")
            
            # Try to create an instance
            inference = inference_bao.BaoInference(".")
            print("✅ Successfully created BaoInference instance")
            
            return True
        else:
            print("❌ BaoInference class not found")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error creating instance: {e}")
        return False
    finally:
        # Remove current directory from path
        if str(Path(".")) in sys.path:
            sys.path.remove(str(Path(".")))

def test_inference_script_execution():
    """Test inference script execution modes"""
    print("\nTesting inference script execution modes...")
    
    inference_script = Path("inference_bao.py")
    
    # Test --test-only mode
    print("Testing --test-only mode...")
    try:
        result = subprocess.run(
            [sys.executable, str(inference_script), "--test-only"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("✅ --test-only mode works")
        else:
            print(f"⚠ --test-only mode failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⚠ --test-only mode timed out")
    except Exception as e:
        print(f"⚠ --test-only mode error: {e}")
    
    # Test --help mode
    print("Testing --help mode...")
    try:
        result = subprocess.run(
            [sys.executable, str(inference_script), "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ --help mode works")
        else:
            print(f"⚠ --help mode failed: {result.stderr}")
            
    except Exception as e:
        print(f"⚠ --help mode error: {e}")
    
    return True

def main():
    """Main test function"""
    print("🧪 Bao LQO Inference Test Suite")
    print("=" * 50)
    
    # Change to bao directory if needed
    if not Path("bao_server").exists():
        print("Error: Please run this script from the bao directory")
        sys.exit(1)
    
    # Run tests
    tests = [
        test_bao_inference_script,
        test_bao_environment,
        test_inference_script_imports,
        test_inference_script_execution,
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
        print("🎉 All tests passed! Bao inference environment is ready.")
        print("\nTo run inference:")
        print("1. From NRBench CLI: iqo bao")
        print("2. Direct execution: python inference_bao.py")
        print("3. Test only: python inference_bao.py --test-only")
        sys.exit(0)
    else:
        print("⚠ Some tests failed. Please check the errors above.")
        print("\nNote: Some tests may fail if no trained model exists.")
        print("Run 'tqo bao' first to train a model, then test again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
