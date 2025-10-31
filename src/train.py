import torch
import torch.nn as nn
from torch.optim import AdamW
from pathlib import Path
import json
import data_loader
import model
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Train model for one epoch"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs.logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item() * images.size(0)
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs.logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(num_epochs=20, batch_size=16, learning_rate=1e-4):
    """Main training function"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, test_loader, class_names = data_loader.get_data_loaders(
        data_dir='data/wm811k', batch_size=batch_size, img_size=224
    )

    # Load model and move to device
    vit_model, processor = model.model_master(num_classes=9)
    vit_model = vit_model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(vit_model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.3)

    # Directory to save checkpoints
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Initialize history dictionary to track training metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")

        train_loss, train_acc = train_one_epoch(vit_model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(vit_model, val_loader, criterion, device)

        # Store metrics in history
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = ckpt_dir / "best_vit_model.pth"
            torch.save(vit_model.state_dict(), save_path)
            print(f"Saved new best model with val_acc={best_val_acc:.4f}")

    print("\nTraining complete")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

    # Save training history to JSON file
    history_path = Path("history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Saved training history to {history_path}")

    # Optional: Evaluate final model on test set
    test_loss, test_acc = validate(vit_model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    train_model(num_epochs=10, batch_size=16, learning_rate=1e-4)
