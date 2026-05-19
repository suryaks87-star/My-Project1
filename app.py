# =========================================
# SENTIMENT ANALYSIS USING SVM
# =========================================

import streamlit as st
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# =========================================
# TITLE
# =========================================

st.title("Sentiment Analysis using SVM")

# =========================================
# FILE UPLOAD
# =========================================

uploaded_file = st.file_uploader(
    "Upload Excel Dataset",
    type=["xlsx"]
)

# =========================================
# IF FILE EXISTS
# =========================================
if uploaded_file is not None:
    try:

        # Read Excel
        df = pd.read_excel(uploaded_file)

        st.subheader("Dataset Preview")
        st.write(df.head())

        st.subheader("Column Names")
        st.write(df.columns)

        # Change these according to your dataset
        X = df["review"]
        y = df["sentiment"]

        # =========================================
        # SPLIT DATA
        # =========================================

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # =========================================
        # TF-IDF
        # =========================================

        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000
        )

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # =========================================
        # MODEL
        # =========================================

        model = LinearSVC()

        model.fit(X_train_vec, y_train)

        # =========================================
        # PREDICTION
        # =========================================

        y_pred = model.predict(X_test_vec)

        # =========================================
        # ACCURACY
        # =========================================

        accuracy = accuracy_score(y_test, y_pred)

        st.subheader("Accuracy")
        st.write(accuracy)

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        # =========================================
        # SAVE MODEL
        # =========================================

        with open("sentiment_model2.pkl", "wb") as f:
            pickle.dump({
                "model": model,
                "vectorizer": vectorizer
            }, f)

        st.success("Model saved as sentiment_model2.pkl")

        # =========================================
        # CUSTOM INPUT
        # =========================================

        st.subheader("Custom Sentiment Prediction")

        user_input = st.text_area("Enter Text")

        if st.button("Predict"):

            if user_input.strip() != "":

                user_vec = vectorizer.transform([user_input])

                prediction = model.predict(user_vec)

                st.success(f"Prediction: {prediction[0]}")

            else:
                st.warning("Please enter text")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Please upload an Excel dataset")
