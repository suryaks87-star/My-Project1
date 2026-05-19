# =========================================
# SENTIMENT PREDICTION
# =========================================

import streamlit as st
import pickle

# =========================================
# LOAD MODEL
# =========================================

with open("sentiment_model2.pkl", "rb") as f:
    saved_data = pickle.load(f)

model = saved_data["model"]
vectorizer = saved_data["vectorizer"]

# =========================================
# TITLE
# =========================================

st.title("Sentiment Analysis")

# =========================================
# USER INPUT
# =========================================

user_input = st.text_area("Enter Text")

# =========================================
# PREDICTION
# =========================================

if st.button("Predict"):

    if user_input.strip() != "":

        # Convert text
        text_vector = vectorizer.transform([user_input])

        # Predict sentiment
        prediction = model.predict(text_vector)[0]

        # Show result
        st.subheader("Prediction")
        st.success(prediction)

        # Confidence Score
        confidence = model.decision_function(text_vector)

        st.subheader("Confidence Score")
        st.write(confidence)

    else:
        st.warning("Please enter text")
