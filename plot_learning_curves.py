"""
Script to plot learning curves from training history.
Can load history from a JSON file or use example data for demonstration.
"""

import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path

def load_history(history_path='history.json'):
    """
    Load training history from a JSON file.
    
    Args:
        history_path: Path to the JSON file containing history data.
                     Expected format: {"train_loss": [...], "val_loss": [...],
                                     "train_acc": [...], "val_acc": [...]}
    
    Returns:
        Dictionary with training history or None if file not found.
    """
    history_file = Path(history_path)
    if history_file.exists():
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
            print(f"✓ Loaded history from {history_path}")
            return history
        except Exception as e:
            print(f"Error loading {history_path}: {e}")
            return None
    return None

def create_example_history(num_epochs=20):
    """
    Create example training history for demonstration purposes.
    
    Args:
        num_epochs: Number of epochs to generate data for.
    
    Returns:
        Dictionary with example training history.
    """
    print("⚠ No history file found. Using example data for demonstration.")
    
    # Generate realistic-looking training curves
    epochs = np.arange(1, num_epochs + 1)
    
    # Train loss: decreasing with some noise
    train_loss = 2.0 * np.exp(-epochs/8) + 0.1 + np.random.normal(0, 0.05, num_epochs)
    train_loss = np.maximum(train_loss, 0.1)  # Ensure non-negative
    
    # Val loss: similar but slightly higher with more noise
    val_loss = 2.2 * np.exp(-epochs/8) + 0.15 + np.random.normal(0, 0.08, num_epochs)
    val_loss = np.maximum(val_loss, 0.1)
    
    # Train accuracy: increasing
    train_acc = 0.3 + 0.6 * (1 - np.exp(-epochs/6)) + np.random.normal(0, 0.02, num_epochs)
    train_acc = np.clip(train_acc, 0, 1)
    
    # Val accuracy: slightly lower than train
    val_acc = 0.25 + 0.55 * (1 - np.exp(-epochs/6)) + np.random.normal(0, 0.03, num_epochs)
    val_acc = np.clip(val_acc, 0, 1)
    
    history = {
        'train_loss': train_loss.tolist(),
        'val_loss': val_loss.tolist(),
        'train_acc': train_acc.tolist(),
        'val_acc': val_acc.tolist()
    }
    
    return history

def plot_learning_curves(history, save_path='learning_curves.png'):
    """
    Plot training and validation loss and accuracy curves.
    
    Args:
        history: Dictionary with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save the figure
    """
    # Learning curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['val_acc'], label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Learning curves saved to {save_path}")

def main():
    """Main function to load history and plot learning curves."""
    # Try to load history from file, otherwise use example data
    history = load_history('history.json')
    
    if history is None:
        # Also try alternative paths
        alternative_paths = ['results/history.json', 'checkpoints/history.json']
        for path in alternative_paths:
            history = load_history(path)
            if history is not None:
                break
        
        # If still None, use example data
        if history is None:
            history = create_example_history(num_epochs=20)
    
    # Verify required keys are present
    required_keys = ['train_loss', 'val_loss', 'train_acc', 'val_acc']
    missing_keys = [key for key in required_keys if key not in history]
    
    if missing_keys:
        print(f"Error: Missing required keys in history: {missing_keys}")
        print("Expected keys: train_loss, val_loss, train_acc, val_acc")
        return
    
    # Plot the learning curves
    plot_learning_curves(history)

if __name__ == "__main__":
    main()

