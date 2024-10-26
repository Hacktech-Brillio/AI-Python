import numpy as np
import torch
import pickle
import coremltools as ct
import torch.nn as nn

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
        x = self.embedding(x.long())  # Ensure input is LongTensor
        output, (h_n, _) = self.lstm(x)
        h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)  # Concatenate final states from both directions
        x = self.dropout(h_n)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load word_to_idx, max_seq_length, and embedding_matrix
with open('word_to_idx.pkl', 'rb') as f:
    word_to_idx = pickle.load(f)
with open('max_seq_length.pkl', 'rb') as f:
    max_seq_length = pickle.load(f)
with open('embedding_matrix.pkl', 'rb') as f:
    embedding_matrix = pickle.load(f)

# Model Parameters
vocab_size = len(word_to_idx)
embed_size = 100  # Set to match GloVe embedding dimensions used during training
hidden_size = 128
output_size = 2  # Binary classification

# Load the trained PyTorch model
model = FakeReviewClassifier(vocab_size, embed_size, hidden_size, output_size, embedding_matrix)
model.load_state_dict(torch.load("best_fake_review_lstm_model.pth"))
model.eval()

# Trace the model with TorchScript using a dummy input
dummy_input = torch.zeros((1, max_seq_length), dtype=torch.long)  # Update to ensure correct type for LSTM
traced_model = torch.jit.trace(model, dummy_input)

# Define your class labels
class_labels = ["CG", "OR"]

# Create a ClassifierConfig with your class labels
classifier_config = ct.ClassifierConfig(class_labels)

# Convert the traced model to Core ML, including the classifier configuration
mlmodel = ct.convert(
    traced_model,
    source="pytorch",
    inputs=[ct.TensorType(name="input", shape=dummy_input.shape, dtype=np.int32)],
    classifier_config=classifier_config
)


# Optional: Set additional metadata
mlmodel.author = 'Vlad Marian'
mlmodel.short_description = 'Fake Review Classifier using LSTM'
mlmodel.license = 'Your License Information'
mlmodel.input_description['input'] = 'Sequence of word indices representing the review text'
mlmodel.output_description['classLabel'] = 'Predicted class label (CG or OR)'

# Save the converted Core ML model with the .mlpackage extension
mlmodel.save("FakeReviewClassifier_with_labels.mlpackage")
print("Model successfully converted and saved as 'FakeReviewClassifier_with_labels.mlpackage'")
