# NLP+CSS 2026 Shared Task Submission

**Ensemble Learning for OIDA Document Classification**

## Overview

This submission presents an ensemble learning approach for classifying documents from the Opioid Industry Document Archive (OIDA) into three categories: promotional, scientific, and regulatory.

**Key Results:**
- Ensemble Model: 84.21% accuracy, 0.7712 F1-macro
- Random Forest: 78.95% accuracy, 0.6570 F1-macro  
- BiLSTM: 57.89% accuracy, 0.2444 F1-macro

**Dataset:** JUUL Labs Collection (377 documents)

## Files Included

1. **NLPCSS2026_SharedTask_Paper.tex** - Main research paper (LaTeX source)
2. **oida_classification.py** - Complete implementation code
3. **confusion_matrices.png** - Visualization of model performance
4. **README.md** - This file

## Requirements

```bash
pip install pandas numpy scikit-learn tensorflow nltk matplotlib seaborn
```

Python version: 3.8+

## Usage

### Google Colab (Recommended)

```python
!python oida_classification.py
# Upload your CSV/ZIP file when prompted
```

### Local Execution

```bash
python oida_classification.py
# Enter file path when prompted
```

## Model Pipeline

1. **Data Loading**: ZIP and CSV file handling
2. **Pseudo-Labeling**: Keyword-based label creation (promotional/scientific/regulatory)
3. **Preprocessing**: Tokenization, lemmatization, stopword removal
4. **Feature Extraction**: TF-IDF with trigrams (max 5000 features)
5. **Training**:
   - Random Forest (500 trees, class-balanced)
   - Ensemble Voting (RF + SVM + Logistic Regression)
   - BiLSTM (2-layer bidirectional LSTM, baseline)
6. **Evaluation**: Accuracy, F1-macro, per-class metrics
7. **Visualization**: Confusion matrices

## Results Summary

### Overall Performance

| Model | Accuracy | F1-Macro | Training Time |
|-------|----------|----------|---------------|
| **Ensemble** | **84.21%** | **0.7712** | 5-6 min |
| Random Forest | 78.95% | 0.6570 | 2-3 min |
| BiLSTM | 57.89% | 0.2444 | 10-15 min |

### Per-Class Performance (Ensemble)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Promotional | 0.76 | 0.81 | 0.79 | 16 |
| Regulatory | 0.56 | 0.63 | 0.59 | 8 |
| Scientific | 0.97 | 0.91 | 0.94 | 33 |

## Reproducibility

- **Random seed**: 42 (all experiments)
- **Train-test split**: 85%-15% stratified
- **Cross-validation**: 5-fold (RF only, F1: 0.6821 ± 0.0423)
- **Hardware**: CPU only (no GPU required)
- **Runtime**: ~5-6 minutes total on standard laptop

## Citation

```
@inproceedings{nlpcss2026shared,
  title={Ensemble Learning for OIDA Document Classification},
  author={Anonymous},
  booktitle={Proceedings of NLP+CSS 2026 Shared Task},
  year={2026}
}
```

## Contact

For questions: nlp-css-2026@example.com

---

**Submission Date**: March 2026  
**NLP+CSS 2026 Shared Task**: OIDA Document Classification
