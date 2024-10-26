import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import contractions
import nltk

# Download NLTK data files (only need to run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the CSV file
data = pd.read_csv("train_dataset.csv")
data = data.dropna(subset=['label'])

# Map labels to 1 for legitimate (OR) and 0 for fake (CG)
data['label'] = data['label'].map({'OR': 1, 'CG': 0}).dropna()

# Text preprocessing
def clean_text(text):
    text = contractions.fix(text)  # Expand contractions
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
    return text

data['review'] = data['review'].apply(lambda x: clean_text(str(x)))

# Tokenization
data['tokens'] = data['review'].apply(word_tokenize)

# Remove stop words
stop_words = set(stopwords.words('english'))
data['tokens'] = data['tokens'].apply(lambda tokens: [word for word in tokens if word not in stop_words])

# Lemmatization
lemmatizer = WordNetLemmatizer()
data['tokens'] = data['tokens'].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])

# Build vocabulary
all_tokens = [token for sublist in data['tokens'] for token in sublist]
word_counts = Counter(all_tokens)
vocab_size = 5000  # Adjust as needed
most_common_words = word_counts.most_common(vocab_size - 2)  # Reserve indices for PAD and UNK
word_to_idx = {'PAD': 0, 'UNK': 1}
for idx, (word, count) in enumerate(most_common_words, start=2):
    word_to_idx[word] = idx

# Convert tokens to indices
def tokens_to_indices(tokens, word_to_idx):
    return [word_to_idx.get(token, word_to_idx['UNK']) for token in tokens]

data['indices'] = data['tokens'].apply(lambda x: tokens_to_indices(x, word_to_idx))

# Pad sequences
max_seq_length = 100  # Adjust as needed
def pad_sequence_custom(seq, max_length, padding_value=0):
    if len(seq) < max_length:
        seq = seq + [padding_value] * (max_length - len(seq))
    else:
        seq = seq[:max_length]
    return seq

data['padded_indices'] = data['indices'].apply(lambda x: pad_sequence_custom(x, max_seq_length))

# Split data into features and labels
X = data['padded_indices'].tolist()
y = data['label'].values

# Split the data into train and test sets
X_train, X_val_test, y_train, y_val_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_val_test, y_val_test, test_size=0.5, random_state=42
)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader for batching
train_data = TensorDataset(X_train_tensor, y_train_tensor)
val_data = TensorDataset(X_val_tensor, y_val_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

# Check unique values in y_train and y_test
print("Unique values in y_train:", set(y_train))
print("Unique values in y_val:", set(y_val))
print("Unique values in y_test:", set(y_test))

# Load pre-trained GloVe embeddings
print("Loading pre-trained embeddings...")
embeddings_index = {}
embed_size = 100  # Using GloVe 100d embeddings

with open('glove.6B.100d.txt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        if word in word_to_idx:
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

print(f"Found {len(embeddings_index)} word vectors.")

# Create embedding matrix
embedding_matrix = np.zeros((len(word_to_idx), embed_size))
for word, idx in word_to_idx.items():
    if word in embeddings_index:
        embedding_matrix[idx] = embeddings_index[word]
    else:
        embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embed_size, ))

# Define the model with an Embedding layer and Bidirectional LSTM
class FakeReviewClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, embedding_matrix):
        super(FakeReviewClassifier, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix), freeze=False, padding_idx=0
        )
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        output, (h_n, _) = self.lstm(x)
        h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)  # Concatenate final states from both directions
        x = self.dropout(h_n)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
hidden_size = 128
output_size = 2  # Binary classification
vocab_size = len(word_to_idx)

model = FakeReviewClassifier(vocab_size, embed_size, hidden_size, output_size, embedding_matrix)

# Handle class imbalance using class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early Stopping parameters
patience = 3
best_val_loss = float("inf")
no_improve_epochs = 0
best_model_state = None

# Training loop with early stopping
num_epochs = 20
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
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    val_loss /= len(val_loader)
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

# Evaluate on the test set
model.load_state_dict(best_model_state)
model.eval()
test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        test_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

test_loss /= len(test_loader)
test_accuracy = correct / total
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Function to preprocess new reviews
def preprocess_new_reviews(reviews):
    reviews = [clean_text(str(review)) for review in reviews]
    tokens = [word_tokenize(review) for review in reviews]
    tokens = [[word for word in review_tokens if word not in stop_words] for review_tokens in tokens]
    tokens = [[lemmatizer.lemmatize(word) for word in review_tokens] for review_tokens in tokens]
    indices = [tokens_to_indices(review_tokens, word_to_idx) for review_tokens in tokens]
    padded_indices = [pad_sequence_custom(index_list, max_seq_length) for index_list in indices]
    return torch.tensor(padded_indices, dtype=torch.long)

# Example of using the model on new test data
test_data = pd.read_csv('test-dataset.csv')

# Preprocess the test data
X_test_new_tensor = preprocess_new_reviews(test_data['review'].tolist())

# Predict using the model
model.eval()
with torch.no_grad():
    outputs = model(X_test_new_tensor)
    _, predicted = torch.max(outputs, 1)

test_data['predicted_label'] = predicted.numpy()
test_data['predicted_label'] = test_data['predicted_label'].map({1: 'OR', 0: 'CG'})

# Display predictions
print(test_data[['review', 'predicted_label']])
