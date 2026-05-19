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
    # -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload your Excel file", type=["xlsx"])

if uploaded_file is None:
    st.info("👆 Upload dataset to see analysis")
    st.stop()

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_excel(uploaded_file)

st.subheader("📄 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# -----------------------------
# Check Columns
# -----------------------------
if 'body' not in df.columns or 'rating' not in df.columns:
    st.error("Dataset must contain 'body' and 'rating'")
    st.stop()

# -----------------------------
# Create Sentiment
# -----------------------------
if 'sentiment' not in df.columns:
    def get_sentiment(r):
        if r <= 2:
            return "Negative"
        elif r == 3:
            return "Neutral"
        else:
            return "Positive"
    
    df['sentiment'] = df['rating'].apply(get_sentiment)

# -----------------------------
# EDA Charts
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("⭐ Ratings")
    fig1, ax1 = plt.subplots()
    df['rating'].value_counts().sort_index().plot(kind='bar', ax=ax1)
    st.pyplot(fig1)

with col2:
    st.subheader("😊 Sentiment")
    fig2, ax2 = plt.subplots()
    df['sentiment'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax2)
    ax2.set_ylabel("")
    st.pyplot(fig2)

# -----------------------------
# ABSA
# -----------------------------
st.subheader("🔥 Aspect-Based Analysis")

aspects = ['camera', 'battery', 'performance', 'display', 'wifi']

def aspect_sentiment(text, aspect):
    sentences = str(text).lower().split('.')
    
    for sentence in sentences:
        if aspect in sentence:
            polarity = TextBlob(sentence).sentiment.polarity
            
            if polarity > 0:
                return "Positive"
            elif polarity < 0:
                return "Negative"
            else:
                return "Neutral"
    return None

for aspect in aspects:
    df[aspect + '_sentiment'] = df['body'].apply(lambda x: aspect_sentiment(x, aspect))

summary_data = []

for aspect in aspects:
    counts = df[aspect + '_sentiment'].value_counts()
    
    summary_data.append({
        'Aspect': aspect.capitalize(),
        'Positive': counts.get('Positive', 0),
        'Negative': counts.get('Negative', 0),
        'Neutral': counts.get('Neutral', 0)
    })

aspect_df = pd.DataFrame(summary_data).set_index('Aspect')

fig3, ax3 = plt.subplots(figsize=(8,5))
sns.heatmap(aspect_df, annot=True, fmt='d', cmap='YlGnBu', ax=ax3)

st.pyplot(fig3)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("<center>🚀 NLP Project Dashboard</center>", unsafe_allow_html=True)
