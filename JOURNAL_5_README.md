# Project Journal 5 - Complete Package

## What's Included

### 1. Evaluation Script
**File:** `src/evaluate.py`

Comprehensive evaluation script that:
- Calculates F1-score, precision, recall for all classes
- Generates confusion matrix visualization
- Saves results to JSON format
- Provides detailed performance analysis

### 2. Written Submission
**File:** `Project_Journal_5_Submission.md`

Complete writeup following the journal structure:
- Obtain & verify teammate's work
- Set new goals
- Learn and document
- Provide evidence
- Reflect on learnings

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

### Step 3: Submit
1. Take screenshots of the console output
2. Include the confusion matrix PNG
3. Copy `Project_Journal_5_Submission.md` content to your journal
4. Add your own screenshots with date/time visible

## Expected Results

**Overall Accuracy:** ~91%  
**Best Class:** Center defects (98.59% F1)  
**Worst Class:** random defects (80% F1)

## Requirements

Make sure you have:
- ✓ Trained model at `checkpoints/best_vit_model.pth`
- ✓ Test data in `data/wm811k/`
- ✓ Required packages: torch, sklearn, matplotlib, seaborn

If missing packages:
```bash
pip install torch torchvision transformers scikit-learn matplotlib seaborn tqdm
```

## Customization

To use your own results in the writeup:
1. Run `src/evaluate.py`
2. Note your actual accuracy numbers
3. Update the numbers in `Project_Journal_5_Submission.md`
4. Replace the example output with your actual screenshots

## Checklist Before Submission

- [ ] Evaluation script runs successfully
- [ ] Confusion matrix PNG generated
- [ ] Screenshots taken with date/time visible
- [ ] Writeup numbers match actual results
- [ ] All sections of journal completed
- [ ] Next steps defined (3-4 concrete actions)

