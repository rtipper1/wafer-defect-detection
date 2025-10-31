# Project Journal 5 - Complete Package

## What's Included

### Evaluation Script
**File:** `src/evaluate.py`

Comprehensive evaluation script that:
- Calculates F1-score, precision, recall for all classes
- Generates confusion matrix visualization
- Saves results to JSON format
- Provides detailed performance analysis

## How to Run

### Step 1: Run Evaluation
```bash
python src/evaluate.py
```

**Expected Runtime:** 1-2 minutes  
**Outputs:**
- Console: Detailed metrics table
- `results/metrics.json`: Numerical metrics
- `results/confusion_matrix.png`: Visualization
- `results/confusion_matrix.npy`: Raw data

### Step 2: View Results
```bash
# View the confusion matrix
open results/confusion_matrix.png

# View JSON metrics
cat results/metrics.json
```
