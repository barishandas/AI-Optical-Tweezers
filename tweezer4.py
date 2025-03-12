import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import norm, gamma, kde
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import json

class ParticleSimulator:
    """Advanced particle simulator with Brownian dynamics"""
    def __init__(self, 
                 n_particles: int = 10,
                 frame_size: Tuple[int, int] = (512, 512),
                 dt: float = 0.01,
                 temperature: float = 300,  # Kelvin
                 viscosity: float = 1e-3,   # Pa·s (water)
                 particle_radius: float = 1e-6):  # meters
        
        self.n_particles = n_particles
        self.frame_size = frame_size
        self.dt = dt
        self.kB = 1.380649e-23  # Boltzmann constant
        self.T = temperature
        self.eta = viscosity
        self.radius = particle_radius
        
        # Calculate diffusion coefficient (Einstein-Stokes)
        self.D = self.kB * self.T / (6 * np.pi * self.eta * self.radius)
        
        # Initialize trap parameters
        self.trap_stiffness = 1e-6  # N/m
        self.trap_positions = None
        self.initialize_traps()
        
        # Initialize particle positions
        self.positions = None
        self.velocities = None
        self.initialize_particles()
    
    def initialize_traps(self):
        """Initialize optical trap positions"""
        # Create a grid of traps
        grid_size = int(np.ceil(np.sqrt(self.n_particles)))
        x = np.linspace(100, self.frame_size[0]-100, grid_size)
        y = np.linspace(100, self.frame_size[1]-100, grid_size)
        XX, YY = np.meshgrid(x, y)
        
        self.trap_positions = np.column_stack((XX.ravel(), YY.ravel()))[:self.n_particles]
    
    def initialize_particles(self):
        """Initialize particle positions and velocities"""
        # Start particles at trap positions with small random offsets
        self.positions = self.trap_positions + np.random.normal(0, 5, (self.n_particles, 2))
        self.velocities = np.zeros((self.n_particles, 2))
    
    def update(self) -> np.ndarray:
        """Update particle positions using Brownian dynamics"""
        # Calculate forces from traps (Harmonic potential)
        displacement = self.positions - self.trap_positions
        trap_force = -self.trap_stiffness * displacement
        
        # Brownian force
        brownian_force = np.random.normal(0, np.sqrt(2 * self.D / self.dt), 
                                        (self.n_particles, 2))
        
        # Update velocities and positions (overdamped regime)
        self.velocities = (trap_force + brownian_force) / (6 * np.pi * self.eta * self.radius)
        self.positions += self.velocities * self.dt
        
        # Enforce boundaries
        self.positions = np.clip(self.positions, 0, 
                               [self.frame_size[0]-1, self.frame_size[1]-1])
        
        return self.positions
    
    def generate_frame(self) -> np.ndarray:
        """Generate an image frame with particles"""
        frame = np.zeros(self.frame_size, dtype=np.float32)
        
        # Update particle positions
        positions = self.update()
        
        # Draw particles as Gaussian spots
        for pos in positions:
            y, x = np.ogrid[-10:11, -10:11]
            spot = np.exp(-(x*x + y*y) / 4)
            
            # Get bounds for the spot
            y_min = max(0, int(pos[1]) - 10)
            y_max = min(self.frame_size[0], int(pos[1]) + 11)
            x_min = max(0, int(pos[0]) - 10)
            x_max = min(self.frame_size[1], int(pos[0]) + 11)
            
            # Add spot to frame
            frame[y_min:y_max, x_min:x_max] += spot[:y_max-y_min, :x_max-x_min]
        
        # Add noise
        frame += np.random.normal(0, 0.1, self.frame_size)
        frame = np.clip(frame, 0, 1)
        
        return frame


class ParticleDetectorCNN(nn.Module):
    """Enhanced CNN for particle detection"""
    def __init__(self):
        super(ParticleDetectorCNN, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Encode
        features = self.encoder(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Decode
        output = self.decoder(attended_features)
        
        return output, attention_weights


class AdvancedAnalyzer:
    """Advanced analysis of particle dynamics"""
    def __init__(self):
        self.results = {}
        
    def analyze_trajectory(self, positions: np.ndarray) -> Dict:
        """Analyze a single particle trajectory"""
        # Calculate basic statistics
        mean_pos = np.mean(positions, axis=0)
        std_pos = np.std(positions, axis=0)
        
        # Calculate velocities and accelerations
        velocities = np.diff(positions, axis=0)
        accelerations = np.diff(velocities, axis=0)
        
        # Calculate MSD
        tau = np.arange(1, len(positions)//2)
        msd = np.zeros_like(tau, dtype=float)
        
        for t in tau:
            diffs = positions[t:] - positions[:-t]
            msd[t-1] = np.mean(np.sum(diffs**2, axis=1))
        
        # Fit MSD to power law
        log_tau = np.log(tau)
        log_msd = np.log(msd)
        slope, intercept = np.polyfit(log_tau, log_msd, 1)
        
        # Calculate velocity autocorrelation
        vel_autocorr = np.correlate(velocities[:, 0], velocities[:, 0], mode='full')
        vel_autocorr = vel_autocorr[len(vel_autocorr)//2:]
        
        return {
            'mean_position': mean_pos,
            'std_position': std_pos,
            'mean_velocity': np.mean(velocities, axis=0),
            'std_velocity': np.std(velocities, axis=0),
            'mean_acceleration': np.mean(accelerations, axis=0),
            'diffusion_coefficient': np.exp(intercept)/4,
            'anomalous_exponent': slope,
            'velocity_autocorr': vel_autocorr
        }
    
    def analyze_ensemble(self, all_positions: List[np.ndarray]) -> Dict:
        """Analyze ensemble of particle trajectories"""
        # Individual trajectory analysis
        trajectory_results = [self.analyze_trajectory(pos) for pos in all_positions]
        
        # Combine all positions and velocities
        all_pos = np.concatenate(all_positions)
        all_vel = np.concatenate([np.diff(pos, axis=0) for pos in all_positions])
        
        # Fit Gaussian mixture model to positions
        gmm = GaussianMixture(n_components=3, random_state=42)
        gmm.fit(all_pos)
        
        # Perform PCA on trajectories
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(all_pos)
        
        # Calculate ensemble statistics
        ensemble_results = {
            'n_particles': len(all_positions),
            'total_positions': len(all_pos),
            'position_gmm_means': gmm.means_.tolist(),
            'position_gmm_covars': gmm.covariances_.tolist(),
            'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
            'mean_diffusion_coef': np.mean([r['diffusion_coefficient'] 
                                          for r in trajectory_results]),
            'mean_anomalous_exp': np.mean([r['anomalous_exponent'] 
                                         for r in trajectory_results])
        }
        
        self.results = {
            'trajectory_results': trajectory_results,
            'ensemble_results': ensemble_results
        }
        
        return self.results
    
    def save_results(self, output_dir: str = 'results'):
        """Save analysis results and generate plots"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(output_dir, f'analysis_{timestamp}')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save numerical results
        with open(os.path.join(save_dir, 'analysis_results.json'), 'w') as f:
            json.dump(self.results['ensemble_results'], f, indent=4)
        
        # Create plots
        self._plot_distributions(save_dir)
        self._plot_trajectory_statistics(save_dir)
        
        print(f"Results saved to {save_dir}")
    
    def _plot_distributions(self, save_dir: str):
        """Generate distribution plots"""
        # Position distribution
        plt.figure(figsize=(10, 8))
        for i, result in enumerate(self.results['trajectory_results']):
            plt.scatter(result['mean_position'][0], result['mean_position'][1], 
                       alpha=0.5, label=f'Particle {i+1}')
        plt.title('Particle Position Distribution')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'position_distribution.png'))
        plt.close()
        
        # Velocity distribution
        plt.figure(figsize=(10, 8))
        all_velocities = [r['mean_velocity'] for r in self.results['trajectory_results']]
        plt.hist(all_velocities, bins=30, density=True, alpha=0.7)
        plt.title('Velocity Distribution')
        plt.xlabel('Velocity')
        plt.ylabel('Density')
        plt.savefig(os.path.join(save_dir, 'velocity_distribution.png'))
        plt.close()
    
    def _plot_trajectory_statistics(self, save_dir: str):
        """Generate statistical plots"""
        # Diffusion coefficient distribution
        plt.figure(figsize=(10, 8))
        diff_coeffs = [r['diffusion_coefficient'] for r in self.results['trajectory_results']]
        sns.histplot(diff_coeffs, kde=True)
        plt.title('Diffusion Coefficient Distribution')
        plt.xlabel('Diffusion Coefficient')
        plt.ylabel('Count')
        plt.savefig(os.path.join(save_dir, 'diffusion_distribution.png'))
        plt.close()
        
        # Anomalous exponent distribution
        plt.figure(figsize=(10, 8))
        anomalous_exps = [r['anomalous_exponent'] for r in self.results['trajectory_results']]
        sns.histplot(anomalous_exps, kde=True)
        plt.title('Anomalous Exponent Distribution')
        plt.xlabel('Anomalous Exponent')
        plt.ylabel('Count')
        plt.savefig(os.path.join(save_dir, 'anomalous_exponent_distribution.png'))
        plt.close()


def main():
    """Main function to run the analysis"""
    print("Starting Advanced AI-Enhanced Optical Tweezers Analysis...")
    
    # Initialize components
    simulator = ParticleSimulator(n_particles=10)
    detector = ParticleDetectorCNN()
    analyzer = AdvancedAnalyzer()
    
    # Generate and analyze data
    n_frames = 1000
    all_positions = []
    
    print(f"Generating {n_frames} frames of particle trajectories...")
    for i in range(n_frames):
        if i % 100 == 0:
            print(f"Processing frame {i}/{n_frames}")
        
        # Generate frame and get true positions
        frame = simulator.generate_frame()
        all_positions.append(simulator.positions.copy())
    
    print("\nAnalyzing particle trajectories...")
    results = analyzer.analyze_ensemble(all_positions)
    
    print("\nSaving results and generating plots...")
    analyzer.save_results()
    
    print("\nAnalysis Summary:")
    print(f"Number of particles analyzed: {results['ensemble_results']['n_particles']}")
    print(f"Mean diffusion coefficient: {results['ensemble_results']['mean_diffusion_coef']:.2e}")
    print(f"Mean anomalous exponent: {results['ensemble_results']['mean_anomalous_exp']:.2f}")


if __name__ == "__main__":
    main()
