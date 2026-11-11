# 🧬 DNA Sequence Classification & Genome Pattern Predictor

A Streamlit web application for the Computational Biology assignment:  
**"Predict the Genome Pattern: AI/ML-based classification of DNA sequences."**

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Features

| Section | Description |
|---------|-------------|
| **Dataset Management** | View, add, modify, or reset labelled DNA sequences |
| **Feature Extraction** | Automatic computation of 6 features (A/T/G/C counts, length, GC%) |
| **ML Training** | Decision Tree or Random Forest with accuracy & feature importance |
| **Prediction** | Classify unknown sequences with confidence scores |
| **Discussion** | Expandable sections covering feature interpretation, real-world applications, limitations, and OOD analysis |

## Requirements

- Python 3.8+
- streamlit, pandas, numpy, scikit-learn, matplotlib
