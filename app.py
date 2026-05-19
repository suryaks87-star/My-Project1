# =========================================
# SENTIMENT ANALYSIS USING SVM (EXCEL FILE)
# =========================================

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# =========================================
# LOAD EXCEL DATASET
# =========================================

# Replace with your Excel filename
df = pd.read_excel("dataset.xlsx")

print(df.head())

# =========================================
# CHECK COLUMN NAMES
# =========================================

print(df.columns)

# Example:
# review -> text column
# sentiment -> output column

X = df["review"]
y = df["sentiment"]

# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================
# TF-IDF VECTORIZATION
# =========================================

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================================
# SVM MODEL
# =========================================

model = LinearSVC()

model.fit(X_train_vec, y_train)

# =========================================
# PREDICTIONS
# =========================================

y_pred = model.predict(X_test_vec)

# =========================================
# ACCURACY
# =========================================

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy :", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================================
# SAVE MODEL
# =========================================

with open("sentiment_model2.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "vectorizer": vectorizer
    }, f)

print("Model saved successfully")

# =========================================
# TEST CUSTOM INPUT
# =========================================

sample = ["This product is really good"]

sample_vec = vectorizer.transform(sample)

prediction = model.predict(sample_vec)

print("Prediction :", prediction[0])
