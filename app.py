"""
DNA Sequence Classification & Genome Pattern Predictor
=======================================================
A Streamlit web application for the Computational Biology assignment:
"Predict the Genome Pattern: AI/ML-based classification of DNA sequences."

This app covers:
  1. Dataset management (view, add, modify DNA sequences)
  2. Automated feature extraction (A, T, G, C counts, length, GC%)
  3. ML model training (Decision Tree / Random Forest) with accuracy & feature importance
  4. Prediction of unknown sequences with confidence scores
  5. Critical discussion sections aligned with SDG 3 & 9

Requirements: streamlit, pandas, numpy, scikit-learn, matplotlib
"""

# ──────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# ──────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="DNA Sequence Classifier",
    page_icon="🧬",
    layout="wide",
)

# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────

def extract_features(sequence: str) -> dict:
    """
    Given a DNA sequence string, compute the six required features:
      - Count of A, T, G, C
      - Sequence length
      - GC content (%)
    Returns a dictionary of feature values.
    """
    seq = sequence.upper().strip()
    length = len(seq)
    count_a = seq.count("A")
    count_t = seq.count("T")
    count_g = seq.count("G")
    count_c = seq.count("C")
    gc_content = ((count_g + count_c) / length * 100) if length > 0 else 0.0
    return {
        "Count_A": count_a,
        "Count_T": count_t,
        "Count_G": count_g,
        "Count_C": count_c,
        "Length": length,
        "GC_Content_%": round(gc_content, 2),
    }


def validate_sequence(seq: str) -> bool:
    """Return True if the sequence contains only valid DNA bases (A, T, G, C)."""
    return all(ch in "ATGC" for ch in seq.upper().strip()) and len(seq.strip()) > 0


def get_default_dataset() -> pd.DataFrame:
    """Return the default labelled DNA dataset."""
    data = {
        "Sequence_ID": ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"],
        "Sequence": [
            "ATGCGATC",   # Group A – moderate GC
            "ATGCGTAC",   # Group A – moderate GC
            "AATTGATC",   # Group A – lower GC
            "ATGATACG",   # Group A – lower GC
            "GCGCGTAT",   # Group B – high GC
            "GCGCGCGC",   # Group B – very high GC
            "GCGCTATA",   # Group B – high GC
            "GCGATAGC",   # Group B – high GC
        ],
        "Group": [
            "Group A", "Group A", "Group A", "Group A",
            "Group B", "Group B", "Group B", "Group B",
        ],
    }
    return pd.DataFrame(data)


# ──────────────────────────────────────────────
# Initialise session state
# ──────────────────────────────────────────────
if "dataset" not in st.session_state:
    st.session_state.dataset = get_default_dataset()

# ──────────────────────────────────────────────
# 1. APP TITLE & OVERVIEW
# ──────────────────────────────────────────────
st.title("🧬 DNA Sequence Classification & Genome Pattern Predictor")

st.markdown(
    """
    Welcome to the **Computational Biology** interactive assignment tool.  
    This application demonstrates **AI/ML-based classification of DNA sequences** — aligning with:

    | UN SDG | Relevance |
    |--------|-----------|
    | **SDG 3 – Good Health & Well-being** | Genomic pattern recognition supports disease diagnostics, pharmacogenomics, and personalised medicine. |
    | **SDG 9 – Industry, Innovation & Infrastructure** | Machine-learning pipelines accelerate genomic research and build scalable bioinformatics infrastructure. |

    **Course context:** *Introduction to Computational Biology* — bridging biology with data science
    to understand genome structure through feature engineering and classification.
    """
)

st.divider()

# ──────────────────────────────────────────────
# 2. DATASET & INPUT MANAGEMENT
# ──────────────────────────────────────────────
st.header("📂 Dataset & Input Management")

tab_view, tab_add, tab_modify = st.tabs(["View Dataset", "Add Sequence", "Modify / Reset"])

with tab_view:
    st.subheader("Raw DNA Dataset")
    st.dataframe(st.session_state.dataset, use_container_width=True, hide_index=True)
    st.caption(f"Total sequences: **{len(st.session_state.dataset)}**")

with tab_add:
    st.subheader("Add a New Labelled Sequence")
    col1, col2, col3 = st.columns([2, 3, 2])
    with col1:
        new_id = st.text_input("Sequence ID", value=f"S{len(st.session_state.dataset) + 1}")
    with col2:
        new_seq = st.text_input("DNA Sequence (A/T/G/C only)", value="")
    with col3:
        new_group = st.selectbox("Group Label", ["Group A", "Group B"])

    if st.button("➕ Add Sequence", use_container_width=True):
        if not new_seq:
            st.warning("Please enter a DNA sequence.")
        elif not validate_sequence(new_seq):
            st.error("Invalid sequence! Only characters A, T, G, C are allowed.")
        elif new_id in st.session_state.dataset["Sequence_ID"].values:
            st.error(f"Sequence ID '{new_id}' already exists. Choose a different ID.")
        else:
            new_row = pd.DataFrame(
                [{"Sequence_ID": new_id, "Sequence": new_seq.upper().strip(), "Group": new_group}]
            )
            st.session_state.dataset = pd.concat(
                [st.session_state.dataset, new_row], ignore_index=True
            )
            st.success(f"✅ Sequence **{new_id}** added to **{new_group}**!")
            st.rerun()

with tab_modify:
    st.subheader("Modify or Reset Dataset")
    st.info("Use the editor below to modify sequences directly, then click **Apply Changes**.")

    edited_df = st.data_editor(
        st.session_state.dataset,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        key="data_editor",
    )
    col_apply, col_reset = st.columns(2)
    with col_apply:
        if st.button("✅ Apply Changes", use_container_width=True):
            # Validate all sequences
            invalid = edited_df[
                ~edited_df["Sequence"].apply(lambda s: validate_sequence(str(s)))
            ]
            if not invalid.empty:
                st.error(
                    f"Invalid sequences found at rows: "
                    f"{invalid.index.tolist()}. Only A/T/G/C characters allowed."
                )
            else:
                st.session_state.dataset = edited_df.copy()
                st.success("Dataset updated successfully!")
                st.rerun()
    with col_reset:
        if st.button("🔄 Reset to Default", use_container_width=True):
            st.session_state.dataset = get_default_dataset()
            st.success("Dataset reset to defaults.")
            st.rerun()

st.divider()

# ──────────────────────────────────────────────
# 3. AUTOMATED FEATURE EXTRACTION
# ──────────────────────────────────────────────
st.header("🔬 Automated Feature Extraction")

# Build the feature table from the current dataset
feature_rows = []
for _, row in st.session_state.dataset.iterrows():
    features = extract_features(row["Sequence"])
    features["Sequence_ID"] = row["Sequence_ID"]
    features["Sequence"] = row["Sequence"]
    features["Group"] = row["Group"]
    feature_rows.append(features)

feature_df = pd.DataFrame(feature_rows)[
    ["Sequence_ID", "Sequence", "Count_A", "Count_T", "Count_G", "Count_C",
     "Length", "GC_Content_%", "Group"]
]

st.markdown(
    "The table below shows the **six computed features** for every sequence in the dataset. "
    "These features form the input to the ML classifier."
)
st.dataframe(
    feature_df.style.format({"GC_Content_%": "{:.2f}"}),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ──────────────────────────────────────────────
# 4. MACHINE LEARNING MODEL TRAINING
# ──────────────────────────────────────────────
st.header("🤖 Machine Learning Model Training")

# --- Classifier selection ---
col_model, col_depth, col_est = st.columns([2, 1, 1])
with col_model:
    classifier_name = st.selectbox(
        "Select Classifier",
        ["Decision Tree", "Random Forest"],
    )
with col_depth:
    max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=3)
with col_est:
    if classifier_name == "Random Forest":
        n_estimators = st.slider("Number of Trees", min_value=10, max_value=200, value=100, step=10)
    else:
        n_estimators = None

# --- Prepare training data ---
FEATURE_COLS = ["Count_A", "Count_T", "Count_G", "Count_C", "Length", "GC_Content_%"]
X = feature_df[FEATURE_COLS].values
y = feature_df["Group"].values

# Map labels for sklearn
label_map = {"Group A": 0, "Group B": 1}
inv_label_map = {v: k for k, v in label_map.items()}
y_encoded = np.array([label_map[lbl] for lbl in y])

# --- Train ---
if st.button("🚀 Train Model", use_container_width=True):
    if len(np.unique(y_encoded)) < 2:
        st.error("Need at least one sequence in each group to train a classifier.")
    else:
        # Instantiate the chosen model
        if classifier_name == "Decision Tree":
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        else:
            model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42
            )

        model.fit(X, y_encoded)

        # Store trained model in session state
        st.session_state.trained_model = model
        st.session_state.model_name = classifier_name

        # --- Training accuracy (on training set) ---
        y_pred = model.predict(X)
        train_acc = accuracy_score(y_encoded, y_pred)

        # --- Cross-validation (leave-one-out style for small datasets) ---
        cv_folds = min(5, len(X))
        if cv_folds >= 2:
            cv_scores = cross_val_score(model, X, y_encoded, cv=cv_folds)
        else:
            cv_scores = np.array([train_acc])

        st.success(f"✅ **{classifier_name}** trained successfully!")

        # Results columns
        res1, res2 = st.columns(2)

        with res1:
            st.metric("Training Accuracy", f"{train_acc * 100:.1f}%")
            st.metric(
                f"{cv_folds}-Fold CV Accuracy",
                f"{cv_scores.mean() * 100:.1f}% ± {cv_scores.std() * 100:.1f}%",
            )

        with res2:
            st.markdown("**Classification Report (Training Set)**")
            report = classification_report(
                y_encoded, y_pred,
                target_names=["Group A", "Group B"],
                output_dict=True,
            )
            report_df = pd.DataFrame(report).T
            st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)

        # --- Feature Importance ---
        st.subheader("📊 Feature Importance")
        importances = model.feature_importances_
        imp_df = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Importance": importances
        }).sort_values("Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(7, 3.5))
        bars = ax.barh(imp_df["Feature"], imp_df["Importance"], color="#4C9BE8")
        ax.set_xlabel("Importance Score")
        ax.set_title(f"Feature Importance — {classifier_name}")
        for bar, val in zip(bars, imp_df["Importance"]):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

        # Store importances for the discussion section
        st.session_state.feature_importances = imp_df

st.divider()

# ──────────────────────────────────────────────
# 5. UNKNOWN SEQUENCE PREDICTION
# ──────────────────────────────────────────────
st.header("🔮 Unknown Sequence Prediction")

unknown_seq = st.text_input(
    "Enter an unknown DNA sequence",
    value="ATGCGCGATC",
    help="Type any DNA sequence (A, T, G, C) to classify it.",
)

if st.button("🧪 Predict Group", use_container_width=True):
    if "trained_model" not in st.session_state:
        st.warning("⚠️ Please train the model first (Section 4 above).")
    elif not unknown_seq.strip():
        st.warning("Please enter a DNA sequence.")
    elif not validate_sequence(unknown_seq):
        st.error("Invalid sequence! Only characters A, T, G, C are allowed.")
    else:
        model = st.session_state.trained_model
        features = extract_features(unknown_seq)
        X_new = np.array([[features[f] for f in FEATURE_COLS]])

        prediction = model.predict(X_new)[0]
        probabilities = model.predict_proba(X_new)[0]

        predicted_label = inv_label_map[prediction]
        confidence = probabilities[prediction] * 100

        # Display results prominently
        st.markdown("---")
        r1, r2, r3 = st.columns([2, 1, 1])
        with r1:
            st.subheader(f"Predicted: **{predicted_label}**")
            if predicted_label == "Group A":
                st.success(f"🅰️ The sequence is classified as **{predicted_label}** "
                           f"with **{confidence:.1f}%** confidence.")
            else:
                st.info(f"🅱️ The sequence is classified as **{predicted_label}** "
                        f"with **{confidence:.1f}%** confidence.")
        with r2:
            st.metric("Group A Probability", f"{probabilities[0] * 100:.1f}%")
        with r3:
            st.metric("Group B Probability", f"{probabilities[1] * 100:.1f}%")

        # Show extracted features for the unknown sequence
        st.markdown("**Extracted Features for the Input Sequence:**")
        unk_feat_df = pd.DataFrame([features])
        unk_feat_df.insert(0, "Sequence", [unknown_seq.upper().strip()])
        st.dataframe(unk_feat_df, use_container_width=True, hide_index=True)

st.divider()

# ──────────────────────────────────────────────
# 6. CRITICAL DISCUSSION & REPORT ELEMENTS
# ──────────────────────────────────────────────
st.header("📝 Critical Discussion & Report Elements")

with st.expander("🔍 Which features were most useful and why?", expanded=False):
    if "feature_importances" in st.session_state:
        imp = st.session_state.feature_importances.sort_values("Importance", ascending=False)
        top_feature = imp.iloc[0]
        st.markdown(
            f"""
            Based on the trained **{st.session_state.model_name}**, the most important feature
            is **{top_feature['Feature']}** (importance = {top_feature['Importance']:.4f}).

            **Interpretation:**
            - **GC Content (%)** typically captures the overall nucleotide composition bias and
              is often the strongest discriminator between groups because it normalises raw
              counts by sequence length.
            - **Individual base counts** (Count_G, Count_C) contribute when groups differ in
              specific nucleotide frequencies.
            - **Sequence Length** is useful only when the groups have systematically different
              lengths; for equal-length sequences it carries no discriminative power.

            The feature importance scores below rank all six features:
            """
        )
        st.dataframe(imp.reset_index(drop=True), use_container_width=True, hide_index=True)
    else:
        st.info("Train the model first to see feature importance analysis.")

with st.expander("🌍 Real-World Genomics Application", expanded=False):
    st.markdown(
        """
        ### Pathogen Identification via Genomic Signatures

        One compelling real-world application is **metagenomic pathogen classification**.
        In clinical metagenomics, sequencing reads from a patient sample contain DNA from
        many organisms. ML classifiers trained on nucleotide composition features (k-mer
        frequencies, GC content, codon usage) can rapidly assign reads to taxonomic groups
        — distinguishing, for example, bacterial from viral sequences — enabling faster
        diagnosis of infectious diseases than traditional culture-based methods.

        **Example:** During the COVID-19 pandemic, ML-based genomic classifiers were used
        to identify SARS-CoV-2 variants from sequencing data, supporting real-time
        surveillance and public-health decisions (aligned with **SDG 3**).
        """
    )

with st.expander("⚠️ Major Limitation of AI/ML in Genomic Analysis", expanded=False):
    st.markdown(
        """
        ### Overfitting and Lack of Biological Context

        A major limitation is **overfitting to superficial statistical patterns** without
        capturing true biological meaning:

        1. **Small, biased training sets** — Genomic datasets are often small or
           taxonomically biased, causing models to memorise noise rather than
           generalisable patterns.
        2. **Feature simplicity** — Simple features like base counts and GC content
           ignore positional information, motifs, secondary structure, and epigenetic
           modifications that are biologically critical.
        3. **Distribution shift** — A model trained on one organism's genome may fail
           completely on another due to codon usage bias, horizontal gene transfer, or
           different evolutionary pressures.
        4. **Interpretability** — Complex models (deep neural networks) often act as
           "black boxes," making it difficult for biologists to trust or validate
           predictions without wet-lab confirmation.

        These limitations underscore the need for **domain-aware feature engineering**
        and **rigorous cross-validation** in computational biology.
        """
    )

with st.expander("🤔 What if a sequence is drastically different from training data?", expanded=False):
    st.markdown(
        """
        ### Out-of-Distribution (OOD) Sequences

        If a query sequence is **drastically different** from any sequence in the
        training set, several problems arise:

        - **Forced classification:** The model will still assign it to Group A or B,
          even if it truly belongs to neither. Standard classifiers have no built-in
          "I don't know" option.
        - **Misleading confidence:** Probability scores may appear confident (e.g., 95%)
          even for OOD inputs, because probabilities reflect the model's learned decision
          boundary — not true biological likelihood.
        - **Extrapolation failure:** Decision Trees and Random Forests partition feature
          space into discrete regions. A sequence with, say, 90% GC content (far beyond
          training range) will simply fall into the nearest partition, giving an unreliable
          prediction.

        **Mitigation strategies:**
        1. Flag sequences whose features fall outside the training data's range.
        2. Use anomaly/novelty detection before classification.
        3. Report prediction intervals or ensemble disagreement scores.
        4. Continuously update the training set with newly characterised sequences.

        > **Discussion prompt:** Try entering a sequence like `AAAAAAAAAA` (100% A) or
        > `GCGCGCGCGCGCGCGCGCGC` (100% GC) in the prediction box above and observe how
        > the model behaves. Is the confidence score trustworthy? Why or why not?
        """
    )

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.divider()
st.caption(
    "Built for the *Introduction to Computational Biology* assignment — "
    "\"Predict the Genome Pattern: AI/ML-based Classification of DNA Sequences.\" | "
    "Aligned with UN SDG 3 (Good Health) & SDG 9 (Innovation & Infrastructure)."
)
