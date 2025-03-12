# Advanced AI-Enhanced Optical Tweezers Analysis System

A comprehensive Python system for simulating and analyzing optical tweezers experiments using artificial intelligence for enhanced particle detection and trajectory analysis.

## Overview

This project provides a complete framework for:
- Simulating optical tweezers with accurate Brownian dynamics
- Detecting and tracking particles using deep learning
- Analyzing particle trajectories with advanced statistical methods
- Visualizing and storing experimental results

## Features

- **Physics-Based Particle Simulation**
  - Realistic Brownian motion modeling
  - Configurable trap stiffness and positions
  - Temperature and viscosity controls
  - Einstein-Stokes diffusion implementation

- **AI-Powered Particle Detection**
  - Convolutional neural network with attention mechanism
  - Encoder-decoder architecture for precise localization
  - Batch normalization for training stability

- **Comprehensive Analysis Tools**
  - Mean square displacement (MSD) calculation
  - Diffusion coefficient estimation
  - Anomalous exponent analysis
  - Velocity autocorrelation
  - Statistical distributions and ensemble properties

- **Automated Visualization**
  - Position distribution plots
  - Velocity distribution histograms
  - Diffusion coefficient analysis
  - Anomalous exponent statistics

## Requirements

- numpy
- torch
- scipy
- sklearn
- matplotlib
- pandas
- seaborn

## Usage

### Basic Usage

```python
from optical_tweezers import ParticleSimulator, ParticleDetectorCNN, AdvancedAnalyzer

# Initialize components
simulator = ParticleSimulator(n_particles=10)
detector = ParticleDetectorCNN()
analyzer = AdvancedAnalyzer()

# Generate simulation data
n_frames = 1000
all_positions = []
for i in range(n_frames):
    frame = simulator.generate_frame()
    all_positions.append(simulator.positions.copy())

# Analyze trajectories
results = analyzer.analyze_ensemble(all_positions)

# Save results
analyzer.save_results()
```

### Configuration Options

The `ParticleSimulator` can be configured with various parameters:

```python
simulator = ParticleSimulator(
    n_particles=10,              # Number of particles to simulate
    frame_size=(512, 512),       # Size of output image
    dt=0.01,                     # Time step for simulation
    temperature=300,             # Temperature in Kelvin
    viscosity=1e-3,              # Fluid viscosity (water = 1e-3 Pa·s)
    particle_radius=1e-6         # Particle radius in meters
)
```

## Code Structure

- `ParticleSimulator`: Generates realistic optical tweezers simulations
- `ParticleDetectorCNN`: Neural network for detecting particles in images
- `AdvancedAnalyzer`: Statistical tools for analyzing particle behaviors

## Results

The analysis generates several outputs:
- JSON files with numerical results
- Position distribution plots
- Velocity distribution histograms
- Diffusion coefficient statistics
- Anomalous exponent analysis

All results are saved in a timestamped directory under `results/`.

## Applications

This system can be used for:
- Teaching optical tweezers concepts
- Testing analysis algorithms
- Developing new particle tracking methods
- Validating experimental setups
- Exploring theoretical models of Brownian motion

## Future Work

- Dynamic trap manipulation
- Multi-trap interaction modeling
- Integration with experimental data
- Real-time analysis capabilities
- GPU acceleration for simulation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
