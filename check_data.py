from pathlib import Path
from collections import Counter

# Path to your data
data_dir = Path('data')

# Find all image files
image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
all_images = []

for ext in image_extensions:
    all_images.extend(data_dir.rglob(f'*{ext}'))

print(f"Found {len(all_images)} total images")

# Count images per class (assuming folder structure)
if all_images:
    # Get parent folder names (defect classes)
    classes = [img.parent.name for img in all_images]
    class_counts = Counter(classes)
    
    print("\n📊 Images per class:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count} images")
else:
    print("❌ No images found! Check your data structure.")
    print(f"\nLooking in: {data_dir.absolute()}")
    print("\nWhat's actually in data/wm811k/?")
    for item in data_dir.iterdir():
        print(f"  - {item.name}")