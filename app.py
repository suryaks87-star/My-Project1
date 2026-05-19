





























i created one svm model can you check and give the same code for sentiment analysis?

Sure — please upload your SVM model code (or paste it here), and I’ll:

check whether it’s correct,

fix any errors or improvements,

and give you the equivalent version for sentiment analysis using SVM.

If you already have:

dataset format,

preprocessing steps,

Streamlit UI,

or trained model code,

include those too so I can adapt it properly.



sentiment_model2.pkl
File

Stopped thinking
I checked your uploaded model file.
It contains an SVM sentiment model using:

scikit-learn LinearSVC

probably saved as a dictionary like:

{
    "model": model,
    ...
}
But the file cannot be fully loaded here because it was trained using a different NumPy/scikit-learn version (numpy._core module mismatch).

Here is a clean and correct SVM Sentiment Analysis code similar to your model:

# =========================
# SENTIMENT ANALYSIS USING SVM
# =========================

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# =========================
# LOAD DATASET
# =========================

# CSV should contain:
# review -> text
# sentiment -> positive/negative

df = pd.read_csv("sentiment_dataset.csv")

print(df.head())

# =========================
# INPUT AND OUTPUT
# =========================

X = df["review"]
y = df["sentiment"]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# TF-IDF VECTORIZER
# =========================

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================
# SVM MODEL
# =========================

model = LinearSVC()

model.fit(X_train_vec, y_train)

# =========================
# PREDICTION
# =========================

y_pred = model.predict(X_test_vec)

# =========================
# EVALUATION
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================

with open("sentiment_model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "vectorizer": vectorizer
    }, f)

print("Model Saved Successfully")

# =========================
# TEST CUSTOM INPUT
# =========================

sample = ["This movie was amazing and very emotional"]

sample_vec = vectorizer.transform(sample)

prediction = model.predict(sample_vec)

print("Prediction:", prediction[0])
Example dataset format:

review	sentiment
Movie was excellent	positive
Waste of time	negative
If you want, I can also give:

Streamlit UI version

NLP preprocessing version

confusion matrix + heatmap

accuracy graph

emoji sentiment detector

multilingual sentiment analysis

advanced version with stemming + lemmatization


