import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, TensorDataset

class FakeReviewClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FakeReviewClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x.unsqueeze(1))
        x = self.fc1(h_n[-1])
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load the CSV file
data = pd.read_csv("train_dataset.csv")  # Replace with your CSV file path

data = data.dropna(subset=['label'])

# Map labels to 1 for legitimate (OR) and 0 for fake (CG)
data['label'] = data['label'].map({'OR': 1, 'CG': 0}).dropna()

# Text preprocessing
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
    return text

data['review'] = data['review'].apply(clean_text)

# Split data into features and labels
X = data['review']
y = data['label']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF for simplicity in this setup
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_tfidf, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_tfidf, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Create DataLoader for batching
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# Check unique values in y_train and y_test
print("Unique values in y_train:", y_train.unique())
print("Unique values in y_test:", y_test.unique())

# Initialize the model, loss function, and optimizer
input_size = X_train_tfidf.shape[1]  # Adjust input size according to your TF-IDF features
hidden_size = 128
output_size = 2  # Binary classification

model = FakeReviewClassifier(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early Stopping parameters
patience = 3
best_val_loss = float("inf")
no_improve_epochs = 0
best_model_state = None

# Training loop with early stopping
num_epochs = 20  # Set a high max number of epochs, early stopping will halt if needed
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Validation step with accuracy calculation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    val_loss /= len(test_loader)
    val_accuracy = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {total_loss/len(train_loader):.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_epochs = 0
        best_model_state = model.state_dict()  # Save the best model state
        print("Validation loss improved, saving model.")
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print("Early stopping triggered.")
            break

# Save the best model after training completes
torch.save(best_model_state, "best_fake_review_lstm_model.pth")
print("Best model saved as 'best_fake_review_lstm_model.pth'")