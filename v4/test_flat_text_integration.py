#!/usr/bin/env python3
"""
Test script for flat text training integration in BrainNexus v4
Tests the new flat text training methods for batch mine processing
"""

import sys
import os
import json
import tempfile
import traceback
from pathlib import Path

# Add the v4 directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from main import BrainNexusInterface
    print("✅ Successfully imported BrainNexusInterface")
except ImportError as e:
    print(f"❌ Failed to import BrainNexusInterface: {e}")
    sys.exit(1)

def create_test_batch_mine_file():
    """Create a test batch_mine.txt file for testing"""
    test_content = """This is a sample text for flat text training.
The BrainNexus system can learn from various types of input data.
Neural networks benefit from diverse training examples.
Machine learning requires careful preprocessing of text data.
Tokenization is an important step in natural language processing.
Context windows help models understand relationships between words.
Training data should be representative of the target domain.
Overfitting can be prevented through proper regularization techniques.
The transformer architecture has revolutionized language modeling.
Attention mechanisms allow models to focus on relevant information."""
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='_batch_mine.txt', delete=False)
    temp_file.write(test_content)
    temp_file.close()
    
    print(f"✅ Created test batch mine file: {temp_file.name}")
    return temp_file.name

def test_flat_text_training():
    """Test the flat text training integration"""
    print("\n🧪 Testing Flat Text Training Integration")
    print("=" * 50)
    
    # Create test file
    test_file = create_test_batch_mine_file()
    
    try:
        # Create interface instance
        interface = BrainNexusInterface()
        print("✅ Created BrainNexusInterface instance")
        
        # Test the _load_batch_mine_data method directly
        print("\n📥 Testing batch mine data loading...")
        
        # Test token prediction approach
        print("\n🔤 Testing token prediction approach...")
        token_data = interface._create_flat_text_training_data(
            open(test_file, 'r').read(),
            approach='token_prediction',
            context_window=5,
            sample_count=10
        )
        print(f"   Generated {len(token_data)} token prediction examples")
        if token_data:
            print(f"   Sample: {token_data[0]}")
        
        # Test sentence completion approach
        print("\n📝 Testing sentence completion approach...")
        sentence_data = interface._create_flat_text_training_data(
            open(test_file, 'r').read(),
            approach='sentence_completion',
            context_window=10,
            sample_count=10
        )
        print(f"   Generated {len(sentence_data)} sentence completion examples")
        if sentence_data:
            print(f"   Sample: {sentence_data[0]}")
        
        # Test paragraph continuation approach
        print("\n📄 Testing paragraph continuation approach...")
        paragraph_data = interface._create_flat_text_training_data(
            open(test_file, 'r').read(),
            approach='paragraph_continuation',
            context_window=15,
            sample_count=10
        )
        print(f"   Generated {len(paragraph_data)} paragraph continuation examples")
        if paragraph_data:
            print(f"   Sample: {paragraph_data[0]}")
        
        # Test full batch mine loading
        print("\n📦 Testing full batch mine data loading...")
        loaded_data = interface._load_batch_mine_data(
            test_file, 
            data_type='flat_text', 
            sample_count=20
        )
        print(f"   Loaded data type: {type(loaded_data)}")
        print(f"   Number of training examples: {len(loaded_data)}")
        
        if loaded_data:
            print("   Sample training examples:")
            for i, example in enumerate(loaded_data[:3]):
                print(f"     Example {i+1}: {example}")
        
        print("\n✅ All flat text training tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        traceback.print_exc()
    
    finally:
        # Clean up test file
        try:
            os.unlink(test_file)
            print(f"🧹 Cleaned up test file: {test_file}")
        except:
            pass

def test_batch_mine_training_command():
    """Test the full batch mine training command"""
    print("\n🚂 Testing Batch Mine Training Command")
    print("=" * 50)
    
    # Create test file
    test_file = create_test_batch_mine_file()
    
    try:
        # Test the command-line interface
        interface = BrainNexusInterface()
        
        # Simulate command line arguments for batch mine training
        test_args = [
            'train', 'test_segment', 
            '--batch-mine', test_file,
            '--approach', 'token_prediction',
            '--context-window', '5',
            '--max-examples', '50'
        ]
        
        print(f"🎯 Testing command: {' '.join(test_args)}")
        
        # Create parser directly (since interface doesn't expose _create_parser)
        import argparse
        parser = argparse.ArgumentParser(description="Test parser for batch mine training")
        subparsers = parser.add_subparsers(dest='command')
        
        # Add train subcommand
        train_parser = subparsers.add_parser('train')
        train_parser.add_argument('segment_name', help='Name of the segment')
        train_parser.add_argument('--batch-mine', help='Batch mine file path')
        train_parser.add_argument('--approach', default='token_prediction', help='Training approach')
        train_parser.add_argument('--context-window', type=int, default=5, help='Context window size')
        train_parser.add_argument('--max-examples', type=int, default=100, help='Maximum examples')
        
        # Parse arguments
        args = parser.parse_args(test_args)
        
        print(f"✅ Arguments parsed successfully:")
        print(f"   Command: {args.command}")
        print(f"   Segment name: {args.segment_name}")
        print(f"   Batch mine file: {args.batch_mine}")
        print(f"   Approach: {args.approach}")
        print(f"   Context window: {args.context_window}")
        print(f"   Max examples: {args.max_examples}")
        
        # Test that the interface can load the data
        print("\n📥 Testing data loading with parsed arguments...")
        loaded_data = interface._load_batch_mine_data(
            args.batch_mine,
            data_type='flat_text',
            sample_count=args.max_examples
        )
        print(f"   Successfully loaded {len(loaded_data)} examples")
        
    except Exception as e:
        print(f"❌ Command test failed: {e}")
        traceback.print_exc()
    
    finally:
        # Clean up
        try:
            os.unlink(test_file)
        except:
            pass

if __name__ == "__main__":
    print("🧪 BrainNexus v4 Flat Text Training Integration Test")
    print("=" * 60)
    
    test_flat_text_training()
    test_batch_mine_training_command()
    
    print("\n🎉 All tests completed!")
