"""
Quick script to test model on a single image.
Usage: python test_single_image.py path/to/image.jpg
"""
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import sys

# Import your model module
sys.path.append('src')
import model

def test_image(image_path, checkpoint_path="checkpoints/best_vit_model.pth"):
    """Test a single image."""
    
    # Class names
    class_names = ['Center', 'Donut', 'Edge Local', 'Edge Ring', 'Local', 
                   'near full', 'none', 'random', 'Scratch']
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    
    vit_model, _ = model.model_master(num_classes=9)
    vit_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    vit_model.to(device)
    vit_model.eval()
    
    # Load and preprocess image
    print(f"\nLoading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    print("\nRunning inference...")
    with torch.no_grad():
        outputs = vit_model(image_tensor)
        probabilities = torch.softmax(outputs.logits, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted_idx.item()]
    confidence_score = confidence.item()
    
    # Print result
    print("\n" + "="*50)
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence_score:.4f} ({confidence_score*100:.2f}%)")
    print("="*50)
    
    # Show top 3 predictions
    top3_probs, top3_indices = torch.topk(probabilities, 3)
    print("\nTop 3 Predictions:")
    for i in range(3):
        idx = top3_indices[0][i].item()
        prob = top3_probs[0][i].item()
        print(f"  {i+1}. {class_names[idx]}: {prob:.4f} ({prob*100:.2f}%)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_single_image.py <path_to_image>")
        print("\nExample:")
        print("  python test_single_image.py data/wm811k/Center/641447.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    
    test_image(image_path)

