# Project Journal 5 - Simple Plan

## Current Status ✅
- Model is already trained and saved in `checkpoints/best_vit_model.pth`
- You can test the model with: `python test_single_image.py data/wm811k/Center/641447.jpg`

## Goal for Journal 5
Show that your wafer defect detection model works by testing it and reporting the results.

---

## Step 1: Test Your Model (30 minutes) 🧪

**What to do:**
Run the inference script to test your model:

```bash
python src/inference.py
```

**What you'll see:**
- Predictions on sample images
- Accuracy on test images (like "87% correct")

**Why this matters:**
This proves your model learned to detect defects from the training data.

---

## Step 2: Calculate Basic Metrics (30 minutes) 📊

**What to do:**
Create a simple script to calculate:
- **Overall accuracy**: How many images classified correctly
- **Per-class accuracy**: Which defects are easiest/hardest to detect

**What you'll need:**
- Load the trained model
- Run it on the test set
- Count correct/incorrect predictions
- Calculate: accuracy = correct / total × 100

**Example code structure:**
```python
# Load model
model = load_trained_model()
# Test on test images
correct = 0
for image, true_label in test_images:
    prediction = model.predict(image)
    if prediction == true_label:
        correct += 1
accuracy = correct / total_images
print(f"Accuracy: {accuracy*100}%")
```

---

## Step 3: Document Your Results (1 hour) 📝

**What to write in your journal:**
1. **What you did**: Tested the trained model
2. **Results**: 
   - Overall accuracy: X%
   - Best class: (which defect was detected best)
   - Worst class: (which defect was hardest)
3. **What it means**: Does this accuracy meet your project goals?
4. **Challenges**: What was difficult?
5. **Next steps**: What would you improve?

**Simple format:**
```
Results:
- Overall Accuracy: 85%
- Center defects: Detected 95% correctly
- Scratch defects: Detected 75% correctly

Interpretation:
The model works well for most defects but struggles with similar-looking ones.
```

---

## What You'll Submit

1. **Screenshots** showing:
   - Model running predictions
   - Accuracy results
   
2. **Your journal entry** with:
   - Description of testing
   - Results table/chart
   - Brief analysis

3. **Code snippet** showing how you calculated accuracy

---

## Tools You Already Have ✅

- **`test_single_image.py`**: Test one image at a time
- **`src/inference.py`**: Test multiple images and get accuracy
- Trained model: `checkpoints/best_vit_model.pth`

---

## Success Criteria

Your journal is complete when you can answer:
- ✅ What is my model's accuracy?
- ✅ Does it work better on some defects than others?
- ✅ Is this good enough for real use?

---

## Help: If You Get Stuck

### Problem: "Model accuracy is low"
**Solution**: This is normal! You can mention in your journal that the model needs more training or data.

### Problem: "I don't understand the results"
**Solution**: Just report the numbers. 70%+ accuracy is reasonable for a new model.

### Problem: "Code doesn't run"
**Solution**: Use the simple `test_single_image.py` script - it's already working.

---

## Time Estimate

- **Step 1 (Test)**: 30 minutes
- **Step 2 (Metrics)**: 30 minutes  
- **Step 3 (Document)**: 1 hour
- **Total**: ~2 hours

---

## Key Point for Journal

Focus on **demonstrating that your model works**, not on having perfect results. Even if accuracy is 70%, that shows:
- ✅ The training process worked
- ✅ The model learned something
- ✅ The system can detect defects

You can discuss improvements for future work.

