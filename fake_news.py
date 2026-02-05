import pandas as pd
import re

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0   # Fake
true["label"] = 1   # Real

# Combine datasets
data = pd.concat([fake, true])

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

data["text"] = data["text"].apply(clean_text)

print("Data loaded and cleaned successfully!")
print(data.head())
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Separate input and output
X = data['text']   # news content
y = data['label']  # 0 = Fake, 1 = True

# Convert text to numbers using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(X)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Feature extraction completed!")
print("Training data size:", X_train.shape)
print("Testing data size:", X_test.shape)
from sklearn.linear_model import LogisticRegression

# Create the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

print("Model training completed!")
from sklearn.metrics import accuracy_score, classification_report

# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Detailed performance report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
# -------- STEP 8: User Input Prediction --------

while True:
    news = input("\nEnter news text (or type 'exit' to stop): ")

    if news.lower() == 'exit':
        print("Exiting Fake News Detector.")
        break

    # Convert input text to vector
    news_vector = vectorizer.transform([news])

    # Predict
    prediction = model.predict(news_vector)

    if prediction[0] == 0:
        print("🟥 This news is FAKE")
    else:
        print("🟩 This news is REAL")