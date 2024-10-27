# 🧠 Brillio Fake Review Classifier AI Model

This project provides an LSTM-based AI model to detect fake reviews, built with PyTorch and converted to Core ML for iOS integration. The model preprocesses review text, embeds it, and then classifies it as either legitimate (OR) or fake (CG).

## 🚀 Features

- **Text Preprocessing**: Cleans and tokenizes text, removes stopwords, lemmatizes tokens, and maps them to indices.
- **Embedding with GloVe**: Utilizes pre-trained GloVe embeddings to enhance text representation.
- **Bi-directional LSTM**: Trained for binary classification, leveraging both past and future context.
- **Core ML Conversion**: Transformed for iOS compatibility, enabling on-device AI inference.

## 📂 Project Structure

- `train_dataset.csv`: Training dataset with labeled reviews (labelled as "OR" for legitimate and "CG" for fake).
- `ai_pytorch_old.py`: Contains the model architecture, preprocessing, and training script.
- `FakeReviewClassifier_with_labels.mlpackage`: The converted Core ML model ready for iOS deployment.
- Additional files:
  - `word_to_idx.pkl`, `max_seq_length.pkl`, `embedding_matrix.pkl`: Required to ensure compatibility when loading the Core ML model for iOS.


## 📝 Preprocessing

1. **Text Cleaning**: Expands contractions, removes special characters, and converts text to lowercase.
2. **Tokenization**: Splits text into tokens, removes stopwords, and lemmatizes.
3. **Indexing and Padding**: Maps tokens to indices using a vocabulary of the most common words and pads sequences to a specified length.

## 🧪 Model Evaluation

After training, the model is evaluated on a test set to compute accuracy and loss.

## 📦 Core ML Conversion

The trained model is converted to Core ML format using TorchScript and CoreMLTools, allowing seamless integration with iOS.

- **Conversion Script**: 
  - Uses `ClassifierConfig` to include class labels for `CG` (fake) and `OR` (legitimate).
- **Saving**:
  - The final model is saved as `FakeReviewClassifier_with_labels.mlpackage`.

## 📱 Using the Model on iOS

The converted `.mlpackage` model can be directly loaded in an iOS app using Core ML. This setup allows iOS apps to classify reviews in real-time, providing users with an authenticity score.