#!/usr/bin/env python3
"""
BrainNexus Batch Mine Training Example

This script demonstrates how to train BrainNexus segments using batch_mine data files.
The batch_mine files contain large amounts of text data that can be used for training.
"""

import os
import sys
from main import BrainNexusInterface

def run_batch_mine_training_demo():
    """Demonstrate training segments with batch_mine data."""
    print("🚀 BRAINNEXUS BATCH MINE TRAINING DEMO")
    print("=" * 60)
    
    # Create interface
    interface = BrainNexusInterface()
    
    try:
        # Step 1: Initialize BrainNexus
        print("\n📋 Step 1: Initializing BrainNexus...")
        interface.run("init", ["64", "true"])
        
        # Step 2: Create segments
        print("\n🏗️ Step 2: Creating brain segments...")
        interface.run("create", ["3", "balanced", "demo"])
        
        # Step 3: Check if batch_mine files exist
        print("\n📁 Step 3: Checking for batch_mine files...")
        import glob
        batch_files = glob.glob("batch_mine_*.json")
        
        if batch_files:
            latest_batch = max(batch_files, key=os.path.getmtime)
            print(f"✅ Found batch_mine file: {os.path.basename(latest_batch)}")
            print(f"   Size: {os.path.getsize(latest_batch) / 1024 / 1024:.1f} MB")
            
            # Step 4: Train with different approaches
            print("\n🎓 Step 4: Training segments with batch_mine data...")
            
            # Text classification training
            print("\n--- Text Classification Training ---")
            interface.run("train", ["supervised", "classification", "file", "20", "32"])
            
            # Multi-modal training
            print("\n--- Multi-Modal Training ---")
            interface.run("train", ["multi_modal", "text", "file", "15", "16"])
            
            # Node-specific training
            print("\n--- Node-Specific Training ---")
            interface.run("train", ["node_specific", "text", "file", "10", "24"])
            
            # Step 5: Analyze training results
            print("\n📊 Step 5: Analyzing training results...")
            interface.run("analyze", ["training", "all", "console"])
            interface.run("analyze", ["performance", "all", "console"])
            
            # Step 6: Test inference on sample data
            print("\n🔮 Step 6: Testing inference...")
            interface.run("infer", ["cybersecurity threat detection", "detailed"])
            interface.run("infer", ["nuclear energy systems", "simple"])
            
        else:
            print("❌ No batch_mine files found!")
            print("   Please ensure you have batch_mine_*.json files in the current directory.")
            print("   Falling back to synthetic data training...")
            
            # Train with synthetic data as fallback
            interface.run("train", ["supervised", "classification", "synthetic", "10", "16"])
        
        # Step 7: Show system status
        print("\n💫 Step 7: Final system status...")
        interface.run("status", [])
        
        print("\n✅ Batch mine training demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_specific_training_examples():
    """Run specific training examples with detailed explanations."""
    print("\n🎯 SPECIFIC TRAINING EXAMPLES")
    print("-" * 50)
    
    interface = BrainNexusInterface()
    
    # Initialize system
    interface.run("init", ["32", "true"])
    interface.run("create", ["2", "mixed", "demo"])
    
    print("\nExample 1: Classification Training on Batch Mine Data")
    print("Command: train supervised classification file 30 64")
    print("- Uses supervised learning")
    print("- Focuses on classification tasks") 
    print("- Loads data from batch_mine files")
    print("- Trains for 30 epochs with batch size 64")
    
    interface.run("train", ["supervised", "classification", "file", "30", "64"])
    
    print("\nExample 2: Evolutionary Training for Complex Patterns")
    print("Command: train evolutionary challenging file 25 32")
    print("- Uses evolutionary algorithms")
    print("- Optimized for challenging pattern recognition")
    print("- Adapts neural topology during training")
    
    interface.run("train", ["evolutionary", "challenging", "file", "25", "32"])
    
    print("\nExample 3: Reinforcement Learning with Batch Data")
    print("Command: train reinforcement text file")
    print("- Uses reward-based learning")
    print("- Optimizes decision-making policies")
    print("- Learns from interaction feedback")
    
    interface.run("train", ["reinforcement", "text", "file"])

def show_training_help():
    """Show detailed training help."""
    interface = BrainNexusInterface()
    interface.run("help", ["train"])

if __name__ == "__main__":
    print("🧠 BrainNexus Batch Mine Training Examples")
    print("Choose an option:")
    print("1. Run full batch mine training demo")
    print("2. Run specific training examples")
    print("3. Show training help")
    print("4. Exit")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            run_batch_mine_training_demo()
        elif choice == "2":
            run_specific_training_examples()
        elif choice == "3":
            show_training_help()
        elif choice == "4":
            print("👋 Goodbye!")
        else:
            print("❌ Invalid choice. Running default demo...")
            run_batch_mine_training_demo()
            
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
