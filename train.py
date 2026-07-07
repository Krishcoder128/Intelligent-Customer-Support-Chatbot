"""
===========================================================
AI Customer Support Chatbot
Train DistilBERT Intent Classification Model

Author : Your Name
Project: Intelligent Customer Support Chatbot
===========================================================
"""

# ==========================
# Import Required Libraries
# ==========================

import os
import json
import pandas as pd
import numpy as np
import pickle
import random
from sklearn.metrics import classification_report
import torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)

from torch.optim import AdamW

from tqdm import tqdm


# ==========================
# Configuration
# ==========================

# Location of the training dataset
DATA_PATH = "data/customer_support_dataset.csv"

# Folder where the trained model will be saved
MODEL_PATH = "models/distilbert_customer_support"

# Pre-trained transformer model from Hugging Face
MODEL_NAME = "distilbert-base-uncased"

# Training parameters
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 8
LEARNING_RATE = 2e-5

# Create the models folder if it doesn't exist
os.makedirs(MODEL_PATH, exist_ok=True)

# ==========================
# Load Dataset
# ==========================

print("=" * 50)
print("Loading training dataset...")
print("=" * 50)

df = pd.read_csv(DATA_PATH)

print("\nDataset Loaded Successfully!\n")

print(df.head())


# ==========================
# Basic Data Cleaning
# ==========================

# Remove empty rows
df.dropna(inplace=True)

# Convert text to lowercase
df["text"] = df["text"].str.lower()

# Remove duplicate rows
df.drop_duplicates(inplace=True)

print(f"\nTotal Samples : {len(df)}")


# ==========================
# Encode Intent Labels
# ==========================

label_encoder = LabelEncoder()

df["label"] = label_encoder.fit_transform(df["intent"])

# Save all labels for chatbot.py
labels = list(label_encoder.classes_)

print("\nIntent Labels\n")

for idx, label in enumerate(labels):
    print(f"{idx} --> {label}")


# ==========================
# Train/Test Split
# ==========================

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.20,
    random_state=42,
    shuffle=True,
    stratify=df["label"]
)

print("\nTraining Samples :", len(train_texts))
print("Testing Samples  :", len(test_texts))


# ==========================
# Load DistilBERT Tokenizer
# ==========================

print("\nLoading DistilBERT Tokenizer...\n")

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)


# ==========================
# Tokenization
# ==========================

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding="max_length",
    max_length=MAX_LENGTH
)

test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding="max_length",
    max_length=MAX_LENGTH
)


print("\nTokenization Completed Successfully!")

# =====================================================
# Create Custom PyTorch Dataset
# =====================================================

class CustomerSupportDataset(Dataset):
    """
    This class converts our tokenized data into a format
    that PyTorch can understand.
    """

    def __init__(self, encodings, labels):

        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        # Returns total number of samples
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns one training sample at a time.
        """

        item = {}

        # Convert every token into a PyTorch tensor
        for key, value in self.encodings.items():
            item[key] = torch.tensor(value[idx])

        # Add the corresponding label
        item["labels"] = torch.tensor(self.labels[idx])

        return item


# =====================================================
# Create Training & Testing Dataset
# =====================================================

train_dataset = CustomerSupportDataset(
    train_encodings,
    train_labels
)

test_dataset = CustomerSupportDataset(
    test_encodings,
    test_labels
)


# =====================================================
# Create DataLoader
# =====================================================

"""
Instead of sending the whole dataset to the model,
DataLoader sends small batches.
"""

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nDataLoader Created Successfully!")


# =====================================================
# Detect CPU or GPU
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing Device : {device}")


# =====================================================
# Load DistilBERT Model
# =====================================================

"""
Load the pre-trained DistilBERT model and replace
its final classification layer according to
the number of intents in our dataset.
"""

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels)
)

# Move model to CPU/GPU
model.to(device)

print("\nDistilBERT Loaded Successfully!")


# =====================================================
# Optimizer
# =====================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_loader) * EPOCHS
)

# =====================================================
# Training Loop
# =====================================================
print("\nStarting Training...\n")

model.train()

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    total_loss = 0

    progress_bar = tqdm(train_loader)

    for batch in progress_bar:

        # Move every tensor to GPU/CPU
        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels_tensor = batch["labels"].to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward Pass
        outputs = model(

            input_ids=input_ids,

            attention_mask=attention_mask,

            labels=labels_tensor

        )

        loss = outputs.loss

        total_loss += loss.item()

        # Backpropagation
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update model weights
        optimizer.step()
        scheduler.step()

        progress_bar.set_description(
            f"Loss : {loss.item():.4f}"
        )

    average_loss = total_loss / len(train_loader)

    print(f"\nAverage Loss : {average_loss:.4f}")

    # =====================================================
# Model Evaluation
# =====================================================

print("\n" + "=" * 50)
print("Evaluating Model...")
print("=" * 50)

model.eval()

correct_predictions = 0
total_predictions = 0

# Disable gradient calculation during evaluation
with torch.no_grad():

    for batch in test_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_tensor = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        predictions = torch.argmax(outputs.logits, dim=1)

        correct_predictions += (predictions == labels_tensor).sum().item()
        total_predictions += labels_tensor.size(0)

accuracy = (correct_predictions / total_predictions) * 100

print(f"\nModel Accuracy : {accuracy:.2f}%")

# =====================================================
# Save Model
# =====================================================

print("\nSaving Model...")





# =====================================================
# Save Label Encoder
# =====================================================


label_encoder_path = os.path.join(
    MODEL_PATH,
    "label_encoder.pkl"
)

with open(label_encoder_path, "wb") as file:
    pickle.dump(label_encoder, file)



# =====================================================
# Save Intent Labels
# =====================================================

labels_path = os.path.join(
    MODEL_PATH,
    "intent_labels.json"
)

with open(labels_path, "w") as file:
    json.dump(labels, file, indent=4)



# =====================================================
# Training Summary
# =====================================================

print("\n" + "=" * 50)
print("Training Completed Successfully!")
print("=" * 50)

print(f"\nModel Saved To : {MODEL_PATH}")

print(f"\nTotal Intents : {len(labels)}")

print("\nIntent List:")

for label in labels:
    print(f"✔ {label}")

print("\nYour AI Customer Support Model is Ready!")