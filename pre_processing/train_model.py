# train_model.py

import os
import pandas as pd
import nltk
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- NLTK SETUP ----------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# ---------------- PATHS ----------------
DATA_CSV = r"D:\narrative nexus\cleaned_data.csv"
MODEL_PATH = r"D:\narrative nexus\models\text_classifier.pkl"
CONF_MATRIX_PATH = r"D:\narrative nexus\models\confusion_matrix.png"

# ---------------- LOAD DATA ----------------
print(f"📂 Loading dataset: {DATA_CSV}")
df = pd.read_csv(DATA_CSV)

# Ensure columns exist
text_col = "text"
target_col = "category"

if text_col not in df.columns or target_col not in df.columns:
    raise ValueError(f"Dataset must contain '{text_col}' and '{target_col}' columns.")

# Drop missing/empty rows
df = df.dropna(subset=[text_col, target_col])
df = df[df[text_col].str.strip() != ""]

# Drop categories with <2 samples
df = df.groupby(target_col).filter(lambda x: len(x) > 1)

if df.empty:
    raise ValueError("No data left after cleaning! Check your dataset.")

X = df[text_col]
y = df[target_col]

print(f"📊 Final dataset: {len(df)} rows, {y.nunique()} categories")
print(f"📈 Category distribution (top 10):\n{y.value_counts().head(10)}")

# ---------------- TRAIN-TEST SPLIT ----------------
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")
except ValueError:
    # If stratified split fails, fall back to random
    print("⚠️ Stratified split failed. Using random split instead.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )

# ---------------- PIPELINE ----------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, stop_words='english')),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

print("🚀 Training pipeline...")
pipeline.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = pipeline.predict(X_test)
accuracy = pipeline.score(X_test, y_test)

print(f"\n🎯 Accuracy: {accuracy:.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# ---------------- CONFUSION MATRIX ----------------
try:
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
    n_classes = len(pipeline.classes_)
    fig_size = min(max(n_classes * 0.8, 8), 20)
    plt.figure(figsize=(fig_size, fig_size))
    
    show_annot = n_classes <= 15  # Annotate if classes <= 15
    
    sns.heatmap(cm, annot=show_annot, cmap="Blues", fmt='d',
                xticklabels=pipeline.classes_, yticklabels=pipeline.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(CONF_MATRIX_PATH), exist_ok=True)
    plt.savefig(CONF_MATRIX_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved to {CONF_MATRIX_PATH}")
except Exception as e:
    print(f"⚠️ Could not create confusion matrix: {e}")

# ---------------- SAVE MODEL ----------------
try:
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")
    
    # Save model info
    model_info_path = os.path.join(os.path.dirname(MODEL_PATH), "model_info.txt")
    with open(model_info_path, 'w') as f:
        f.write(f"text_column: {text_col}\n")
        f.write(f"target_column: {target_col}\n")
        f.write(f"classes: {list(pipeline.classes_)}\n")
        f.write(f"accuracy: {accuracy:.4f}\n")
    
    print(f"📋 Model info saved to {model_info_path}")
except Exception as e:
    print(f"❌ Error saving model: {e}")

print(f"\n🎉 Training completed successfully!")
print(f"📈 Final accuracy: {accuracy:.4f}")
print(f"🏷️ Number of classes: {len(pipeline.classes_)}")
