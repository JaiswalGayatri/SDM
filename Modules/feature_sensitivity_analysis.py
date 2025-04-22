import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import seaborn as sns
import os

class FeatureSensitivityAnalyzer:
    def __init__(self, model, feature_names: List[str], feature_ranges: Dict[str, Tuple[float, float]] = None):
        """
        Initialize the analyzer with a trained model and feature names.
        
        Args:
            model: Trained model (RandomForest or LogisticRegression)
            feature_names: List of feature names in the same order as model features
            feature_ranges: Dictionary mapping feature names to (min, max) tuples
        """
        self.model = model
        self.feature_names = feature_names
        self.feature_ranges = feature_ranges or {}
        
    def normalize_value(self, feature_name: str, value: float) -> float:
        """Convert actual value to normalized form using min-max scaling."""
        if feature_name in self.feature_ranges:
            min_val, max_val = self.feature_ranges[feature_name]
            return (value - min_val) / (max_val - min_val)
        return value
        
    def denormalize_value(self, feature_name: str, value: float) -> float:
        """Convert normalized value back to actual value."""
        if feature_name in self.feature_ranges:
            min_val, max_val = self.feature_ranges[feature_name]
            return value * (max_val - min_val) + min_val
        return value
    
    def find_high_probability_point(self, X: np.ndarray, threshold: float = 0.9) -> Tuple[np.ndarray, float]:
        """
        Find a point with high probability of presence.
        
        Args:
            X: Feature matrix
            threshold: Minimum probability threshold
            
        Returns:
            Tuple of (point features, probability)
        """
        probs = self.model.predict_proba(X)[:, 1]
        high_prob_indices = np.where(probs >= threshold)[0]
        
        if len(high_prob_indices) == 0:
            raise ValueError(f"No points found with probability >= {threshold}")
            
        # Select the point with highest probability
        best_idx = high_prob_indices[np.argmax(probs[high_prob_indices])]
        return X[best_idx], probs[best_idx]
    
    def vary_feature(self, base_point: np.ndarray, feature_idx: int, X: np.ndarray,
                    steps: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vary a single feature while keeping others constant.
        
        Args:
            base_point: Original point features
            feature_idx: Index of feature to vary
            X: Full feature matrix (used to determine feature range)
            steps: Number of steps in variation
            
        Returns:
            Tuple of (actual values, normalized values, probabilities)
        """
        feature_name = self.feature_names[feature_idx]
        
        # Get feature range
        if feature_name in self.feature_ranges:
            min_val, max_val = self.feature_ranges[feature_name]
        else:
            min_val = np.min(X[:, feature_idx])
            max_val = np.max(X[:, feature_idx])
        
        # Generate actual values
        actual_values = np.linspace(min_val, max_val, steps)
        
        # Convert to normalized form
        normalized_values = np.array([self.normalize_value(feature_name, v) for v in actual_values])
        
        # Create test points
        test_points = np.tile(base_point, (steps, 1))
        test_points[:, feature_idx] = normalized_values
        
        # Get probabilities
        probabilities = self.model.predict_proba(test_points)[:, 1]
        
        return actual_values, normalized_values, probabilities
    
    def analyze_all_features(self, base_point: np.ndarray, X: np.ndarray,
                           steps: int = 500) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Analyze sensitivity of all features.
        
        Args:
            base_point: Original point features
            X: Full feature matrix
            steps: Number of steps in variation (increased to 500 for smoother plots)
            
        Returns:
            Dictionary mapping feature names to (actual values, normalized values, probabilities)
        """
        results = {}
        print("\nAnalyzing feature sensitivity...")
        for i, feature_name in enumerate(self.feature_names):
            print(f"Processing {feature_name}...")
            actual_values, normalized_values, probabilities = self.vary_feature(
                base_point, i, X, steps
            )
            results[feature_name] = (actual_values, normalized_values, probabilities)
        return results
    
    def plot_feature_sensitivity(self, results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                               save_path: str = None):
        """
        Plot sensitivity analysis results.
        
        Args:
            results: Dictionary from analyze_all_features
            save_path: Path to save the plots (optional)
        """
        # Create directory for individual plots if save_path is provided
        if save_path:
            plot_dir = os.path.dirname(save_path)
            individual_plot_dir = os.path.join(plot_dir, 'individual_feature_plots')
            os.makedirs(individual_plot_dir, exist_ok=True)
        
        # Plot all features in a grid
        n_features = len(results)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        plt.figure(figsize=(20, 5*n_rows))
        
        for i, (feature_name, (actual_values, _, probs)) in enumerate(results.items()):
            # Main grid plot
            plt.subplot(n_rows, n_cols, i+1)
            plt.plot(actual_values, probs, 'b-', linewidth=2)
            plt.title(feature_name, fontsize=12)
            plt.xlabel('Actual Feature Value', fontsize=10)
            plt.ylabel('Probability', fontsize=10)
            plt.grid(True)
            
            # Add more x-axis ticks
            x_min, x_max = np.min(actual_values), np.max(actual_values)
            x_ticks = np.linspace(x_min, x_max, 10)
            plt.xticks(x_ticks, rotation=45)
            
            # Individual plot
            plt.figure(figsize=(12, 8))
            plt.plot(actual_values, probs, 'b-', linewidth=2)
            plt.title(f'Feature Sensitivity: {feature_name}', fontsize=14)
            plt.xlabel('Actual Feature Value', fontsize=12)
            plt.ylabel('Probability', fontsize=12)
            plt.grid(True)
            plt.xticks(x_ticks, rotation=45)
            
            if save_path:
                individual_plot_path = os.path.join(individual_plot_dir, f'{feature_name}_sensitivity.png')
                plt.savefig(individual_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
        
        # Save the grid plot
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nSensitivity plots saved to: {save_path}")
            print(f"Individual feature plots saved to: {individual_plot_dir}")
        plt.show()
    
    def get_feature_importance(self, results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Dict[str, float]:
        """
        Calculate feature importance based on sensitivity analysis.
        
        Args:
            results: Dictionary from analyze_all_features
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        importance_scores = {}
        print("\nCalculating feature importance scores...")
        
        for feature_name, (actual_values, _, probs) in results.items():
            # Calculate the range of probabilities
            prob_range = np.max(probs) - np.min(probs)
            
            # Calculate the average absolute gradient using actual values
            gradients = np.abs(np.diff(probs) / np.diff(actual_values))
            avg_gradient = np.mean(gradients[np.isfinite(gradients)])
            
            # Combine metrics for importance score
            importance_score = prob_range * avg_gradient
            importance_scores[feature_name] = importance_score
            
            print(f"\n{feature_name}:")
            print(f"  Probability range: {prob_range:.4f}")
            print(f"  Average gradient: {avg_gradient:.4f}")
            print(f"  Raw importance score: {importance_score:.4f}")
        
        # Normalize scores to sum to 1
        total_score = sum(importance_scores.values())
        if total_score > 0:
            importance_scores = {k: v/total_score for k, v in importance_scores.items()}
            print("\nNormalized importance scores:")
            for feature, score in sorted(importance_scores.items(), key=lambda x: x[1], reverse=True):
                print(f"{feature}: {score:.4f}")
        
        return importance_scores 