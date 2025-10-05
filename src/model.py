#  Load pre-trained ViT from Hugging Face
#  Adapt it for 9 classes
#  Create function that returns the model

# Load model directly

from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch.nn as nn

def model_loader():
    
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model_load = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

    return processor, model_load

def model_adapt(model_load, num_classes=9):
    
    # adapt model for 9 classes
    model_load.classifier = nn.Linear(model_load.config.hidden_size, num_classes)
    return model_load

def model_master(num_classes = 9):

    processor, model_load = model_loader()
    model = model_adapt(model_load, num_classes)
    
    return model, processor

if __name__ == "__main__":
    model, processor = model_master(num_classes=9)
    print("Model loaded successfully!")
    print(f"Number of classes: {model.config.num_labels}")  # Access through model.config
    print(f"Classifier output size: {model.classifier.out_features}")  # Alternative way