import pandas as pd
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the CSV file
data = pd.read_csv("train_dataset.csv")  # Replace with the path to your CSV file

# Convert labels to numeric: OR -> 1 (legit), CG -> 0 (fake)
data['label'] = data['label'].map({'OR': 1, 'CG': 0})

# Drop rows where 'label' is NaN
data = data.dropna(subset=['label'])

# Clean text data: lowercase, remove special characters
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
    return text

# Apply cleaning function to the review column
data['review'] = data['review'].apply(clean_text)

# Separate features and target
X = data['review']
y = data['label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Preprocessing complete. Ready for model training.")

# Initialize the Naive Bayes model
model = MultinomialNB()

# Train the model on the TF-IDF vectors
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy and display a classification report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Fake', 'Legit'])

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# # Save the model and vectorizer
# joblib.dump(model, "fake_review_detector.pkl")
# joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
