# =========================================
# SENTIMENT PREDICTION ONLY
# =========================================

import streamlit as st
import pickle
import numpy as np

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

        # Transform text
        text_vector = vectorizer.transform([user_input])

        # Prediction
        prediction = model.predict(text_vector)[0]

        st.subheader("Prediction")
        st.success(prediction)

        # =========================================
        # PROBABILITY SCORE
        # =========================================

        try:

            # For models supporting probability
            probabilities = model.predict_proba(text_vector)[0]

            classes = model.classes_

            st.subheader("Probability Scores")

            for cls, prob in zip(classes, probabilities):
                st.write(f"{cls}: {prob:.2%}")

        except:

            # LinearSVC does not support predict_proba
            decision = model.decision_function(text_vector)[0]

            confidence = abs(decision)

            st.subheader("Confidence Score")

            st.write(f"Confidence: {confidence:.2f}")

    else:
        st.warning("Please enter some text")
