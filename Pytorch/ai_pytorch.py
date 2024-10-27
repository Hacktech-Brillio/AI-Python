# Import necessary libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import contractions
import ssl

# NLTK Setup
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Load the CSV file
data = pd.read_csv("Data/train_dataset.csv")
data = data.dropna(subset=['label', 'review'])

# Map labels to 1 for legitimate (OR) and 0 for fake (CG)
data['label'] = data['label'].map({'OR': 1, 'CG': 0}).dropna()

# Text preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = contractions.fix(text)  # Expand contractions
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
    return text

def get_wordnet_pos(treebank_tag):
    """Map POS tag to first character lemmatize() accepts"""
    from nltk.corpus import wordnet
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text):
    text = clean_text(text)
    tokens = nltk.word_tokenize(text)
    # Remove stop words
    tokens = [word for word in tokens if word not in stop_words]
    # POS tagging
    pos_tags = nltk.pos_tag(tokens)
    # Lemmatization with POS tags
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]
    return ' '.join(tokens)

data['cleaned_review'] = data['review'].apply(preprocess_text)

# Split the data into train and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    data['cleaned_review'], data['label'], test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1, random_state=42
)

# Load DistilBERT tokenizer and model
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Tokenize input texts
max_seq_length = 128  # Reduced sequence length

def tokenize_function(texts):
    return tokenizer(
        texts.tolist(),
        padding='max_length',
        truncation=True,
        max_length=max_seq_length,
        return_tensors='pt'
    )

train_encodings = tokenize_function(X_train)
val_encodings = tokenize_function(X_val)
test_encodings = tokenize_function(X_test)

# Create custom dataset
class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.values

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ReviewDataset(train_encodings, y_train)
val_dataset = ReviewDataset(val_encodings, y_val)
test_dataset = ReviewDataset(test_encodings, y_test)

# DataLoaders
batch_size = 8  # Reduced batch size

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Use GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Optimizer and scheduler with mixed precision support
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
num_epochs = 2  # Fewer epochs
total_steps = len(train_loader) * num_epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# Training loop with mixed precision
from tqdm.auto import tqdm

best_val_accuracy = 0
patience = 2
no_improve_epochs = 0
best_model_state = None

for epoch in range(num_epochs):
    # Training
    model.train()
    total_train_loss = 0
    train_correct = 0
    train_total = 0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            logits = outputs.logits

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_train_loss += loss.item()
        _, preds = torch.max(logits, dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    avg_train_loss = total_train_loss / len(train_loader)
    train_accuracy = train_correct / train_total

    # Validation
    model.eval()
    total_val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                logits = outputs.logits

            total_val_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = val_correct / val_total

    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Early stopping
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        no_improve_epochs = 0
        best_model_state = model.state_dict()
        print("Validation accuracy improved, saving model.")
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print("Early stopping triggered.")
            break

# Save the best model
torch.save(best_model_state, "best_fake_review_distilbert_model.pth")
print("Best model saved as 'best_fake_review_distilbert_model.pth'")

# Load the best model for testing
model.load_state_dict(best_model_state)
model.eval()

# Evaluation on the test set
test_correct = 0
test_total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

test_accuracy = test_correct / test_total
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=['CG', 'OR']))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
