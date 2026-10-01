import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords
nltk.download("stopwords")

# ---------------------------------------
# 1. Load datasets
# ---------------------------------------

fake = pd.read_csv("archive/Fake.csv")
true = pd.read_csv("archive/True.csv")

# ---------------------------------------
# 2. Add labels
# ---------------------------------------

fake["label"] = 1
true["label"] = 0

# ---------------------------------------
# 3. Combine datasets
# ---------------------------------------

data = pd.concat([fake, true], ignore_index=True)

# Shuffle
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# ---------------------------------------
# 4. Combine title + text
# ---------------------------------------

data["content"] = (
    data["title"].fillna("") + " " +
    data["text"].fillna("")
)

# ---------------------------------------
# 5. Text preprocessing
# ---------------------------------------

stop_words = set(stopwords.words("english"))

def clean_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove HTML
    text = re.sub(r"<.*?>", "", text)

    # Keep only letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove stopwords
    words = text.split()
    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)


print("Cleaning text...")

data["clean_content"] = data["content"].apply(clean_text)

# ---------------------------------------
# 6. Input and target
# ---------------------------------------

X = data["clean_content"]
y = data["label"]

# ---------------------------------------
# 7. Train/Test split
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ---------------------------------------
# 8. TF-IDF
# ---------------------------------------

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2)
)

print("\nCreating TF-IDF features...")

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF completed!")

print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------
# 9. Train Logistic Regression
# ---------------------------------------

print("\nTraining Logistic Regression model...")

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

print("Model training completed!")

# ---------------------------------------
# 10. Make predictions
# ---------------------------------------

print("\nMaking predictions...")

y_pred = model.predict(X_test_tfidf)

# ---------------------------------------
# 11. Evaluate model
# ---------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Accuracy percentage:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Real", "Fake"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
from sklearn.naive_bayes import MultinomialNB

# ---------------------------------------
# 12. Train Naive Bayes
# ---------------------------------------

print("\nTraining Naive Bayes model...")

nb_model = MultinomialNB()

nb_model.fit(X_train_tfidf, y_train)

print("Naive Bayes training completed!")

# ---------------------------------------
# 13. Make predictions
# ---------------------------------------

nb_pred = nb_model.predict(X_test_tfidf)

# ---------------------------------------
# 14. Evaluate Naive Bayes
# ---------------------------------------

nb_accuracy = accuracy_score(y_test, nb_pred)

print("\nNaive Bayes Accuracy:", nb_accuracy)
print("Naive Bayes Accuracy percentage:", nb_accuracy * 100, "%")

print("\nNaive Bayes Classification Report:")
print(classification_report(
    y_test,
    nb_pred,
    target_names=["Real", "Fake"]
))

print("\nNaive Bayes Confusion Matrix:")
print(confusion_matrix(y_test, nb_pred))
# ---------------------------------------
# 15. Test the model with new news
# ---------------------------------------

print("\n--------------------------------")
print("FAKE NEWS DETECTION SYSTEM")
print("--------------------------------")

news = input("\nEnter a news article: ")

# Clean the user's news
clean_news = clean_text(news)

# Convert to TF-IDF
news_tfidf = vectorizer.transform([clean_news])

# Predict
prediction = model.predict(news_tfidf)[0]

# Get probability
probability = model.predict_proba(news_tfidf)[0]

# Display result
if prediction == 1:
    print("\nPrediction: FAKE NEWS")
    print("Confidence:", round(probability[1] * 100, 2), "%")
else:
    print("\nPrediction: REAL NEWS")
    print("Confidence:", round(probability[0] * 100, 2), "%")
# ---------------------------------------
# Save trained model and vectorizer
# ---------------------------------------

import joblib

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")