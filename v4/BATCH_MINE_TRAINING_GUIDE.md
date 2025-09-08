# BrainNexus Batch Mine Training Guide

## Overview

The BrainNexus v4 architecture supports training segments using large batch mining data files. This guide explains how to use the `batch_mine_*.json` files to train your neural segments with real-world data.

## Quick Start

### 1. Prerequisites

Make sure you have:
- A `batch_mine_*.json` file in your working directory
- BrainNexus v4 system initialized
- At least one brain segment created

### 2. Basic Training Commands

```bash
# Initialize the system
python main.py
> init 64 true

# Create segments  
> create 3 balanced demo

# Train with batch mine data
> train supervised classification file 50 32
```

## Training Types

### Supervised Training
Best for classification and regression tasks:
```bash
> train supervised classification file 50 32
> train supervised text file 100 64
```

### Multi-Modal Training
For complex data processing:
```bash
> train multi_modal text file 30 16
> train multi_modal vision file 25 32
```

### Reinforcement Learning
For adaptive behavior learning:
```bash
> train reinforcement demo file 40 24
> train reinforcement text file 60 48
```

### Evolutionary Training
For topology optimization:
```bash
> train evolutionary challenging file 35 28
```

### Node-Specific Training
For targeted node type training:
```bash
> train node_specific text file 20 16
```

## Data Processing

### Automatic Batch Mine Detection
The system automatically:
1. Scans for `batch_mine_*.json` files in the current directory
2. Uses the most recent file based on modification time
3. Processes the JSON content into training samples

### Content Classification
The system extracts features and creates labels based on content analysis:

- **Category 0**: Cybersecurity (cyber, security, firewall, encryption)
- **Category 1**: Security Threats (attack, threat, vulnerability, exploit)
- **Category 2**: Nuclear Technology (nuclear, reactor, uranium, plutonium)
- **Category 3**: Energy Systems (energy, power, electricity, renewable)
- **Category 4**: Computer Systems (computer, software, system, network)
- **Category 5**: Data Processing (data, information, database, mining)
- **Category 6**: Government/Policy (government, policy, regulation, law)
- **Category 7**: Research/Analysis (research, study, analysis, investigation)
- **Category 8**: International Affairs (international, global, country, nation)
- **Category 9**: General/Other (everything else)

### Feature Extraction
For each text sample, the system extracts:
- Text length and word count
- Character-based statistics (uppercase ratio, digit ratio, punctuation ratio)
- Keyword presence indicators
- Statistical measures (average word length, max word length)
- Hash-based numerical features

## Complete Training Workflow

### Step 1: System Setup
```bash
# Start the interface
python main.py

# Initialize BrainNexus
> init 64 true

# Create segments
> create 5 balanced demo
```

### Step 2: Training
```bash
# Primary training with batch mine data
> train supervised classification file 75 64

# Additional specialized training
> train multi_modal text file 50 32
> train node_specific text file 25 16
```

### Step 3: Analysis
```bash
# Analyze training results
> analyze training all console
> analyze performance all console
> analyze topology segments console
```

### Step 4: Testing
```bash
# Test inference
> infer "cybersecurity threat analysis" detailed
> infer "nuclear power systems" simple
> infer "data mining techniques" raw
```

## Advanced Configuration

### Custom Training Parameters

You can specify additional parameters:
```bash
# Format: train <type> <task> <source> <epochs> <batch_size>
> train supervised classification file 100 128  # 100 epochs, batch size 128
> train evolutionary challenging file 80 64     # 80 epochs, batch size 64
> train reinforcement text file 60 32           # 60 episodes, batch size 32
```

### Task Configuration Options

- **classification**: Multi-class classification tasks
- **text**: Text processing and NLP tasks  
- **vision**: Computer vision tasks (converts text to image-like features)
- **demo**: Quick demonstration tasks
- **challenging**: Complex patterns to promote neural evolution

### Data Source Options

- **file**: Load from batch_mine JSON files (recommended)
- **synthetic**: Generate synthetic training data
- **random**: Random data for testing

## Monitoring and Debugging

### View Training Progress
```bash
> analyze training all console
```

### Check System Status
```bash
> status
```

### Review Performance Metrics
```bash
> analyze performance all console
```

### Detailed Help
```bash
> help train
> help analyze
> help infer
```

## Common Issues and Solutions

### Issue: No batch_mine files found
**Solution**: Ensure `batch_mine_*.json` files are in the working directory

### Issue: Out of memory during training
**Solution**: Reduce batch size or use fewer epochs
```bash
> train supervised classification file 25 16  # Smaller batch size
```

### Issue: Poor training performance
**Solutions**:
1. Increase epochs: `> train supervised classification file 100 32`
2. Try evolutionary training: `> train evolutionary challenging file 50 32`
3. Use node-specific training: `> train node_specific text file 30 24`

### Issue: Training takes too long
**Solutions**:
1. Reduce epochs: `> train supervised classification file 20 32`
2. Use demo configuration: `> create 2 balanced demo`
3. Increase batch size: `> train supervised classification file 50 64`

## Example Scripts

### Complete Training Example
```python
# Run the batch mine training example
python batch_mine_training_example.py
```

### Manual Interface Usage
```python
from main import BrainNexusInterface

interface = BrainNexusInterface()
interface.run("init", ["64", "true"])
interface.run("create", ["3", "balanced", "demo"])
interface.run("train", ["supervised", "classification", "file", "50", "32"])
interface.run("analyze", ["performance", "all", "console"])
```

## Best Practices

1. **Start Small**: Begin with demo configuration and fewer epochs
2. **Monitor Progress**: Use analysis commands to track training effectiveness
3. **Experiment**: Try different training types to find what works best
4. **Save Results**: Use `> save` command to preserve trained models
5. **Use File Data**: Always prefer `file` data source over synthetic when batch_mine data is available

## File Structure

```
your_project/
├── batch_mine_20250809_135204.json    # Your batch mine data
├── main.py                            # Main interface
├── batch_mine_training_example.py     # Training examples
├── BrainNexus.py                      # Core architecture
├── BrainNexusLearning.py             # Learning algorithms
└── brainnexus_workspace/             # Generated files
    ├── segments/                     # Saved segments
    ├── models/                       # Trained models
    ├── results/                      # Training results
    └── logs/                         # Log files
```

## Next Steps

1. Run the example script: `python batch_mine_training_example.py`
2. Try different training configurations
3. Analyze results and optimize parameters
4. Scale up to larger configurations as needed
5. Implement custom training pipelines for specific use cases

For more information, use the built-in help system:
```bash
python main.py
> help train
> help analyze
> help infer
```
