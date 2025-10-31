"""
Simple inference script to test the trained wafer defect detection model.
"""
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path
import model
import data_loader


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


def preprocess_image(image_path, img_size=224):
    """
    Preprocess image for ViT model.
    
    Args:
        image_path: Path to image file
        img_size: Target image size
    
    Returns:
        Preprocessed tensor
    """
    # Define the same transforms used in training
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor, image


def predict(model, image_tensor, class_names, device='cuda'):
    """
    Make prediction on preprocessed image.
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor
        class_names: List of class names
        device: Device to run inference on
    
    Returns:
        Predicted class name and confidence score
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs.logits, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted_idx.item()]
    confidence_score = confidence.item()
    
    # Get top 3 predictions
    top3_probs, top3_indices = torch.topk(probabilities, 3)
    
    predictions = []
    for i in range(3):
        predictions.append({
            'class': class_names[top3_indices[0][i].item()],
            'confidence': top3_probs[0][i].item()
        })
    
    return predicted_class, confidence_score, predictions


def test_on_image(model, image_path, class_names, device='cuda'):
    """
    Test model on a single image.
    
    Args:
        model: Trained model
        image_path: Path to test image
        class_names: List of class names
        device: Device to run on
    """
    print(f"\nTesting on: {image_path}")
    
    # Preprocess image
    image_tensor, original_image = preprocess_image(image_path)
    
    # Make prediction
    predicted_class, confidence, top3 = predict(model, image_tensor, class_names, device)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print(f"{'='*60}")
    
    print(f"\nTop 3 Predictions:")
    for i, pred in enumerate(top3, 1):
        print(f"  {i}. {pred['class']}: {pred['confidence']:.4f} ({pred['confidence']*100:.2f}%)")
    
    return predicted_class, confidence


def test_on_test_set(model, test_loader, class_names, device='cuda', num_samples=10):
    """
    Test model on test set and show sample predictions.
    
    Args:
        model: Trained model
        test_loader: Test DataLoader
        class_names: List of class names
        device: Device to run on
        num_samples: Number of samples to test
    """
    print(f"\nTesting model on {num_samples} random test samples...")
    print("="*60)
    
    correct = 0
    total = 0
    model.eval()
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if i >= num_samples // images.size(0) + 1:
                break
            
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.logits.argmax(dim=1)
            
            for j in range(len(images)):
                if total >= num_samples:
                    break
                
                true_label = class_names[labels[j].item()]
                pred_label = class_names[preds[j].item()]
                correct += 1 if labels[j] == preds[j] else 0
                
                # Get confidence
                probs = torch.softmax(outputs.logits[j], dim=0)
                confidence = probs[preds[j]].item()
                
                status = "✓" if labels[j] == preds[j] else "✗"
                print(f"{status} True: {true_label:15s} | Pred: {pred_label:15s} | Conf: {confidence:.3f}")
                
                total += 1
    
    accuracy = correct / total * 100
    print("="*60)
    print(f"Test Accuracy on {total} samples: {accuracy:.2f}%")
    
    return accuracy


def main():
    """Main function to run inference."""
    # Configuration
    checkpoint_path = "checkpoints/best_vit_model.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Class names
    class_names = [
        'Center', 'Donut', 'Edge Local', 'Edge Ring', 'Local', 
        'near full', 'none', 'random', 'Scratch'
    ]
    
    print("="*60)
    print("Wafer Defect Detection - Inference")
    print("="*60)
    print(f"Device: {device}")
    
    # Load trained model
    model_loaded = load_trained_model(checkpoint_path, num_classes=9, device=device)
    
    # Option 1: Test on a specific image
    print("\n" + "="*60)
    print("Option 1: Test on specific images")
    print("="*60)
    
    # Test on a few sample images from different classes
    test_images = [
        "data/wm811k/Center/641447.jpg",
        "data/wm811k/Donut/679360.jpg",
        "data/wm811k/Scratch/641447.jpg" if Path("data/wm811k/Scratch").exists() else None,
    ]
    
    for img_path in test_images:
        if img_path and Path(img_path).exists():
            try:
                test_on_image(model_loaded, img_path, class_names, device)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Option 2: Test on test set
    print("\n" + "="*60)
    print("Option 2: Test on test set")
    print("="*60)
    
    try:
        _, _, test_loader, _ = data_loader.get_data_loaders(
            data_dir='data/wm811k', batch_size=16, img_size=224
        )
        test_on_test_set(model_loaded, test_loader, class_names, device, num_samples=20)
    except Exception as e:
        print(f"Could not test on test set: {e}")
    
    print("\n" + "="*60)
    print("Inference complete!")
    print("="*60)


if __name__ == "__main__":
    main()

