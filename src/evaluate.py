"""
Comprehensive evaluation script for wafer defect detection model.
Implements F1-score, precision, recall, and confusion matrix analysis.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
from pathlib import Path
import json
from tqdm import tqdm

import data_loader
import model


def load_trained_model(checkpoint_path, num_classes=9, device='cuda'):
    """
    Load trained ViT model from checkpoint.
    
    Args:
        checkpoint_path: Path to saved model checkpoint
        num_classes: Number of classes (9 for wafer defects)
        device: Device to run on ('cuda' or 'cpu')
    
    Returns:
        Loaded model ready for inference
    """
    # Load model architecture
    vit_model, processor = model.model_master(num_classes=num_classes)
    
    # Load trained weights
    vit_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    vit_model.to(device)
    vit_model.eval()  # Set to evaluation mode
    
    print(f"✓ Model loaded from {checkpoint_path}")
    return vit_model


def evaluate_on_test_set(model, test_loader, device='cuda'):
    """
    Evaluate model on test set and return predictions.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test set
        device: Device to run inference on
        
    Returns:
        all_preds: All predictions
        all_labels: All true labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Running inference on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions
            preds = outputs.logits.argmax(dim=1)
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)


def calculate_metrics(y_true, y_pred, class_names):
    """
    Calculate comprehensive metrics for model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        metrics: Dictionary of metrics
    """
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Macro and weighted averages
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Create detailed report
    report = classification_report(y_true, y_pred, 
                                   target_names=class_names,
                                   output_dict=True)
    
    metrics = {
        'overall_accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'per_class_metrics': {
            name: {
                'precision': float(p),
                'recall': float(r),
                'f1': float(f)
            }
            for name, p, r, f in zip(class_names, precision, recall, f1)
        },
        'classification_report': report
    }
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save figure
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - Wafer Defect Detection', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to {save_path}")
    
    plt.close()


def print_metrics_table(metrics, class_names):
    """
    Print formatted metrics table to console.
    
    Args:
        metrics: Dictionary of metrics
        class_names: List of class names
    """
    print("\n" + "="*80)
    print("EVALUATION METRICS")
    print("="*80)
    
    print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")
    
    print("\nPer-Class Performance:")
    print("-"*80)
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*80)
    
    for class_name in class_names:
        class_metrics = metrics['per_class_metrics'][class_name]
        print(f"{class_name:<20} "
              f"{class_metrics['precision']:<12.4f} "
              f"{class_metrics['recall']:<12.4f} "
              f"{class_metrics['f1']:<12.4f}")
    
    print("="*80)


def save_results(metrics, cm, save_dir='results'):
    """
    Save evaluation results to JSON and figures.
    
    Args:
        metrics: Dictionary of metrics
        cm: Confusion matrix
        save_dir: Directory to save results
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Save metrics to JSON
    json_path = save_dir / 'metrics.json'
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics to {json_path}")
    
    # Save confusion matrix
    cm_path = save_dir / 'confusion_matrix.npy'
    np.save(cm_path, cm)
    print(f"✓ Saved confusion matrix to {cm_path}")


def main():
    """Main evaluation function."""
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "checkpoints/best_vit_model.pth"
    num_classes = 9
    
    print("="*60)
    print("Wafer Defect Detection - Model Evaluation")
    print("="*60)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    
    # Load data
    print("\nLoading test data...")
    _, _, test_loader, class_names = data_loader.get_data_loaders(
        data_dir='data/wm811k', batch_size=16, img_size=224
    )
    
    # Load trained model
    print("\nLoading trained model...")
    model_eval = load_trained_model(checkpoint_path, num_classes=num_classes, device=device)
    
    # Evaluate
    y_pred, y_true = evaluate_on_test_set(model_eval, test_loader, device=device)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(y_true, y_pred, class_names)
    
    # Print results
    print_metrics_table(metrics, class_names)
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, class_names, 
                         save_path='results/confusion_matrix.png')
    
    # Save results
    print("\nSaving results...")
    save_results(metrics, cm, save_dir='results')
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
