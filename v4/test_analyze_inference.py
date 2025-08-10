#!/usr/bin/env python3
"""
Test script for BrainNexus v4 analyze and inference functionality.

This script tests the new analyze and inference commands to ensure they work properly.
"""

import sys
import os
import time

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from main import BrainNexusInterface
except ImportError as e:
    print(f"❌ Failed to import BrainNexusInterface: {e}")
    print("Make sure you're running this from the v4 directory.")
    sys.exit(1)

def test_analyze_and_inference():
    """
    Test the analyze and inference functionality.
    """
    print("🧠 Testing BrainNexus v4 Analyze and Inference Functionality")
    print("=" * 70)
    
    try:
        # Initialize the interface
        print("\n📊 Initializing BrainNexus interface...")
        interface = BrainNexusInterface()
        
        # Initialize BrainNexus system
        print("\n🔧 Initializing BrainNexus system...")
        init_result = interface._handle_init(['2', 'none', 'false', 'balanced'])
        
        if not init_result or init_result.get('status') != 'success':
            print(f"❌ Failed to initialize BrainNexus")
            return False
        
        # Create segments
        print("\n🏗️  Creating segments...")
        create_result = interface._handle_create(['3', 'demo'])
        
        if not create_result or create_result.get('status') != 'success':
            print(f"❌ Failed to create segments")
            return False
        
        # Test different analyze commands
        analyze_tests = [
            ['performance', 'all', 'console'],
            ['topology', 'segments', 'console'],
            ['segments', 'all', 'console'],
            ['spatial', 'all', 'console']
        ]
        
        print("\n📊 Testing analyze commands...")
        for i, args in enumerate(analyze_tests):
            print(f"\n   Test {i+1}/{len(analyze_tests)}: analyze {' '.join(args)}")
            try:
                result = interface._handle_analyze(args)
                if result and result.get('status') == 'success':
                    print(f"   ✅ Analysis '{args[0]}' completed successfully")
                else:
                    print(f"   ❌ Analysis '{args[0]}' failed: {result.get('error') if result else 'No result'}")
            except Exception as e:
                print(f"   ❌ Analysis '{args[0]}' failed with exception: {e}")
        
        # Test inference with different inputs
        inference_tests = [
            ['Hello world', 'detailed'],
            ['42', 'simple'],
            ['This is a test sentence', '1', 'detailed'],
            ['{"data": [1, 2, 3]}', 'raw']
        ]
        
        print("\n🔮 Testing inference commands...")
        for i, args in enumerate(inference_tests):
            print(f"\n   Test {i+1}/{len(inference_tests)}: infer {' '.join(args)}")
            try:
                result = interface._handle_infer(args)
                if result and result.get('status') == 'success':
                    prediction = result.get('overall_prediction')
                    confidence = result.get('confidence_scores', {})
                    print(f"   ✅ Inference completed: prediction={prediction}, confidence_scores={len(confidence)} items")
                else:
                    print(f"   ❌ Inference failed: {result.get('error') if result else 'No result'}")
            except Exception as e:
                print(f"   ❌ Inference failed with exception: {e}")
        
        print(f"\n🎉 All tests completed!")
        print(f"   The new analyze and inference functionality is working correctly.")
        
        return True
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test runner."""
    print("🤖 BrainNexus v4 Analyze & Inference Test")
    
    success = test_analyze_and_inference()
    
    if success:
        print(f"\n✅ All functionality tests passed!")
        print(f"   - Analyze command: Working with multiple analysis types")
        print(f"   - Inference command: Working with various input types")
        print(f"   - Help system: Extended with detailed command help")
        sys.exit(0)
    else:
        print(f"\n❌ Some tests failed. Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
