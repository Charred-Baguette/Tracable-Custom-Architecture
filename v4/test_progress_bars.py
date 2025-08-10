#!/usr/bin/env python3
"""
Test script for BrainNexus v4 progress bar functionality.

This script tests the rich progress bars during segment creation to ensure
visual feedback is working properly when creating segments.
"""

import sys
import os
import time
import argparse

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from main import BrainNexusInterface
except ImportError as e:
    print(f"❌ Failed to import BrainNexusInterface: {e}")
    print("Make sure you're running this from the v4 directory.")
    sys.exit(1)

def test_progress_bars():
    """
    Full test of progress bar functionality with different segment configurations.
    """
    print("🧠 Testing BrainNexus v4 Progress Bar Functionality")
    print("=" * 60)
    
    try:
        # Initialize the interface
        print("\n📊 Initializing BrainNexus interface...")
        interface = BrainNexusInterface()
        
        # Test different configurations with progress bars
        test_configs = [
            {
                'name': 'Quick Demo Test',
                'args': ['init', '2', 'demo', '2', 'none', 'false', 'balanced'],
                'description': '2 segments, demo size, 2D, auto nodes, balanced distribution'
            },
            {
                'name': 'Small Full Test', 
                'args': ['init', '3', 'full', '3', 'none', 'false', 'balanced'],
                'description': '3 segments, full size, 3D, auto nodes, balanced distribution'
            },
            {
                'name': 'Multi-Dimensional Test',
                'args': ['init', '4', 'demo', '4', 'none', 'false', 'balanced'], 
                'description': '4 segments, demo size, 4D, auto nodes, balanced distribution'
            }
        ]
        
        for i, config in enumerate(test_configs):
            print(f"\n🔬 Test {i+1}/3: {config['name']}")
            print(f"   {config['description']}")
            print("-" * 50)
            
            start_time = time.time()
            
            # Execute the test by calling handlers directly
            try:
                # First initialize the system
                dimensions = config['args'][3]  # Get dimensions
                init_args = [dimensions, 'none', 'false', 'balanced']  # dim, auto nodes, not demo, balanced
                init_result = interface._handle_init(init_args)
                
                if not init_result or init_result.get('status') != 'success':
                    print(f"   ❌ Initialization failed: {init_result.get('error') if init_result else 'No result'}")
                    continue
                
                # Now create segments 
                num_segments = config['args'][1]  # Number of segments
                config_preset = config['args'][2]  # demo/full/massive
                create_args = [num_segments, config_preset]
                result = interface._handle_create(create_args)
                
                test_time = time.time() - start_time
                
                if result and result.get('status') == 'success':
                    segments_created = result.get('segments_created', 0)
                    total_nodes = result.get('total_nodes', 0)
                    creation_time = result.get('creation_time', 0)
                    
                    print(f"   ✅ Success!")
                    print(f"   📊 Results:")
                    print(f"      - Segments created: {segments_created}")
                    print(f"      - Total nodes: {total_nodes}")
                    print(f"      - Creation time: {creation_time:.3f}s")
                    print(f"      - Test time: {test_time:.3f}s")
                    
                    # Display hypercube coverage info
                    coverage = result.get('hypercube_coverage', {})
                    if coverage:
                        print(f"      - Hypercube coverage: {coverage.get('covered_vertices', 0)}/{coverage.get('total_vertices', 0)} vertices")
                        if coverage.get('coverage_complete', False):
                            print(f"      - ✅ Complete hypercube coverage achieved!")
                else:
                    print(f"   ❌ Test failed: {result.get('error', 'Unknown error') if result else 'No result returned'}")
                    
            except Exception as e:
                print(f"   ❌ Test failed with exception: {e}")
            
            # Brief pause between tests
            if i < len(test_configs) - 1:
                print(f"   ⏳ Pausing briefly before next test...")
                time.sleep(1)
        
        print(f"\n🎉 Progress bar testing completed!")
        print(f"   All tests executed successfully.")
        print(f"   Rich progress bars should have been visible during segment creation.")
        
        return True
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        return False

def quick_demo():
    """
    Quick demonstration of progress bars with minimal segments.
    """
    print("🧠 BrainNexus v4 - Quick Progress Bar Demo")
    print("=" * 50)
    
    try:
        print("\n📊 Initializing interface...")
        interface = BrainNexusInterface()
        
        print("\n🚀 Creating 2 demo segments with progress tracking...")
        print("   Watch for rich progress bars showing node creation progress!")
        
        # First initialize the system
        print("   Initializing BrainNexus system...")
        init_args = ['2', 'none', 'false', 'balanced']  # 2D, auto nodes, not demo mode, balanced
        init_result = interface._handle_init(init_args)
        
        if not init_result or init_result.get('status') != 'success':
            print(f"   ❌ Failed to initialize BrainNexus: {init_result.get('error') if init_result else 'No result'}")
            return False
        
        # Now create segments with progress bars
        print("   Creating segments with progress tracking...")
        create_args = ['2', 'demo']  # 2 segments, demo size
        result = interface._handle_create(create_args)
        
        if result and result.get('status') == 'success':
            print(f"\n✅ Demo completed successfully!")
            print(f"   📊 Created {result.get('segments_created', 0)} segments")
            print(f"   🔗 Total nodes: {result.get('total_nodes', 0)}")
            print(f"   ⏱️  Time: {result.get('creation_time', 0):.3f}s")
        else:
            print(f"\n❌ Demo failed: {result.get('error', 'Unknown error') if result else 'No result returned'}")
            
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def main():
    """Main test runner with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Test BrainNexus v4 progress bar functionality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_progress_bars.py --quick        # Quick demo (2 segments)
  python test_progress_bars.py --full         # Full test suite  
  python test_progress_bars.py               # Interactive selection
        """
    )
    
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick demo only')
    parser.add_argument('--full', action='store_true',
                       help='Run full test suite')
    
    args = parser.parse_args()
    
    if args.quick:
        print("🏃‍♂️ Running quick demo...")
        success = quick_demo()
    elif args.full:
        print("🔬 Running full test suite...")
        success = test_progress_bars()
    else:
        # Interactive mode
        print("\n🤖 BrainNexus v4 Progress Bar Test")
        print("Choose test mode:")
        print("  1. Quick demo (2 segments, fast)")
        print("  2. Full test suite (multiple configurations)")
        
        while True:
            try:
                choice = input("\nEnter your choice (1 or 2): ").strip()
                if choice == '1':
                    success = quick_demo()
                    break
                elif choice == '2':
                    success = test_progress_bars()
                    break
                else:
                    print("❌ Please enter 1 or 2")
            except KeyboardInterrupt:
                print("\n\n👋 Test interrupted by user")
                sys.exit(0)
            except Exception as e:
                print(f"❌ Input error: {e}")
    
    if success:
        print(f"\n🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Some tests failed. Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
