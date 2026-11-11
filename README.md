# 🧬 DNA Sequence Classification & Genome Pattern Predictor

A Streamlit web application for computational biology that classifies DNA sequences into groups using machine learning.

## Assignment
**Predict the Genome Pattern:** AI/ML-based classification of DNA sequences  
**Course:** Introduction to Computational Biology  
**SDG Alignment:** SDG 3 (Good Health & Well-Being) · SDG 9 (Industry, Innovation & Infrastructure)

## Features

- **Dataset Management** — View, add, and modify labelled DNA sequences
- **Automated Feature Extraction** — Computes 6 features: A/T/G/C counts, sequence length, GC content (%)
- **ML Model Training** — Decision Tree or Random Forest classifier with accuracy metrics & feature importance
- **Unknown Sequence Prediction** — Classify any DNA sequence with confidence scores
- **Critical Discussion** — Expandable sections covering real-world applications, limitations, and edge cases

## Run Locally

```bash
pip install -r requirements.txt
streamlit run dna_sequence_classifier.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub account and select this repository.
4. Set **Main file path** to `dna_sequence_classifier.py`.
5. Click **Deploy**.

## Tech Stack

- Python 3.9+
- Streamlit
- pandas & NumPy
- scikit-learn
- Plotly
