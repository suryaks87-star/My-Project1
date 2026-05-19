# =========================================
# SENTIMENT ANALYSIS PREDICTION APP
# =========================================

import streamlit as st
import pickle
import numpy as np

# =========================================
# LOAD TRAINED MODEL
# =========================================

with open("sentiment_model2.pkl", "rb") as f:
    saved_data = pickle.load(f)

model = saved_data["model"]
vectorizer = saved_data["vectorizer"]

# =========================================
# PAGE TITLE
# =========================================

st.title("Sentiment Analysis using SVM")

st.write("Predict whether text is Positive, Negative, or Neutral")

# =========================================
# USER INPUT
# =========================================

user_input = st.text_area("Enter Your Text")

# =========================================
# LABEL MAPPING
# =========================================

label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# =========================================
# PREDICTION BUTTON
# =========================================

if st.button("Predict Sentiment"):

    if user_input.strip() != "":

        try:

            # Convert text to vector
            text_vector = vectorizer.transform([user_input])

            # Predict
            prediction = model.predict(text_vector)[0]

            # =========================================
            # HANDLE LABELS
            # =========================================

            if prediction in label_map:
                final_prediction = label_map[prediction]
            else:
                final_prediction = str(prediction)

            # =========================================
            # DISPLAY PREDICTION
            # =========================================

            st.subheader("Prediction Result")

            if final_prediction.lower() == "positive":
                st.success(f"Positive 😊")

            elif final_prediction.lower() == "negative":
                st.error(f"Negative 😔")

            elif final_prediction.lower() == "neutral":
                st.warning(f"Neutral 😐")

            else:
                st.info(final_prediction)

            # =========================================
            # CONFIDENCE SCORE WITH LABEL
            # =========================================

            st.subheader("Confidence Scores")

            scores = model.decision_function(text_vector)

            # Multiclass
            if len(scores.shape) > 1:

                scores = scores[0]

                for i, score in enumerate(scores):

                    if i in label_map:
                        label = label_map[i]
                    else:
                        label = str(i)

                    st.write(f"{label} : {round(score, 2)}")

            # Binary classification
            else:

                score = scores[0]

                if score > 0:
                    st.write(f"Positive : {round(score, 2)}")
                else:
                    st.write(f"Negative : {round(abs(score), 2)}")

        except Exception as e:
            st.error(f"Error: {e}")

    else:
        st.warning("Please enter some text")
