import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# ------------------------------
# Load Data Function
# ------------------------------
@st.cache_data
def load_data():
    # Read CSVs safely with UTF-8 encoding
    fake = pd.read_csv("Fake.csv", encoding="utf-8", engine="python", on_bad_lines="skip", errors="replace")
    true = pd.read_csv("True.csv", encoding="utf-8", engine="python", on_bad_lines="skip", errors="replace")


    # Ensure the column name is 'text'
    text_col_fake = "text"
    text_col_true = "text"

    fake[text_col_fake] = fake[text_col_fake].astype(str)
    true[text_col_true] = true[text_col_true].astype(str)

    fake["label"] = 0  # 0 = Fake
    true["label"] = 1  # 1 = Real

    # Combine datasets
    data = pd.concat([fake, true], axis=0)
    data = data[[text_col_fake, "label"]]
    data = data.rename(columns={text_col_fake: "text"})
    data = data.dropna()
    return data

# ------------------------------
# Train Model Function
# ------------------------------
@st.cache_data
def train_model(data):
    X = data["text"]
    y = data["label"]

    vectorizer = CountVectorizer(stop_words="english")
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y)
    return vectorizer, model

# ------------------------------
# Streamlit App
# ------------------------------
st.title("📰 Fake News Detection System")
st.write("This app predicts whether a news article is Fake or Real using Machine Learning.")

# Input text from user
user_input = st.text_area("Enter the news text here:")

# Load data and train model
data = load_data()
vectorizer, model = train_model(data)

# Prediction button
if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter some news text first!")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]

        if prediction == 0:
            st.error("❌ This news is FAKE")
        else:
            st.success("✅ This news is REAL")
