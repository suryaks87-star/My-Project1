# =========================================
# SENTIMENT PREDICTION ONLY
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

st.title("Sentiment Analysis Predictor")

# =========================================
# USER INPUT
# =========================================

user_input = st.text_area("Enter your text")

# =========================================
# PREDICT BUTTON
# =========================================

if st.button("Predict Sentiment"):

    if user_input.strip() != "":

        # Convert text
        text_vector = vectorizer.transform([user_input])

        # Prediction
        prediction = model.predict(text_vector)[0]

        st.subheader("Prediction")
        st.success(prediction)

        # =========================================
        # CONFIDENCE SCORE
        # =========================================

        confidence = abs(model.decision_function(text_vector)[0])

        st.subheader("Confidence Score")
        st.write(round(confidence, 2))

    else:
        st.warning("Please enter text")
