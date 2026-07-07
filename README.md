# Intelligent Customer Support Chatbot

An AI-powered Customer Support Chatbot developed using **Python**, **DistilBERT**, **PyTorch**, and **Hugging Face Transformers**. The chatbot is designed to understand customer queries using Natural Language Processing (NLP) and provide accurate, context-aware responses by retrieving information from structured datasets such as customer orders, products, refunds, and FAQs.

This project was developed as a **Capstone Project** to demonstrate the practical application of transformer-based language models in customer support automation.

---

## Project Overview

Customer support is an essential part of every business, but handling repetitive customer queries manually can be time-consuming and inefficient. This project aims to automate common customer support tasks by combining Artificial Intelligence with structured data retrieval.

The chatbot uses a fine-tuned **DistilBERT** model to identify the user's intent and then retrieves the required information from CSV-based databases. It also implements confidence-based fallback logic to avoid incorrect responses when the model is uncertain.

---

## Features

- Transformer-based intent classification using DistilBERT
- Natural language understanding
- Customer order tracking
- Product information retrieval
- Refund status lookup
- Shipping and delivery information
- Billing and payment assistance
- Order cancellation support
- Account management support
- Promotions and offers information
- Confidence score prediction
- Confidence-based fallback mechanism
- Human handoff simulation
- Hybrid AI + Rule-based architecture
- Flexible Order ID and Product ID detection

---

## Intent Categories

The chatbot currently supports the following customer support intents:

- Greeting
- Order Tracking
- Product Inquiry
- Refund
- Shipping Information
- Billing Inquiry
- Cancel Order
- Promotions
- Account Creation
- Account Recovery
- Account Management
- Human Handoff

---

## Project Structure

```text
CustomerSupportChatbot/
│
├── data/
│   ├── customer_support_dataset.csv
│   ├── customer_orders.csv
│   ├── products.csv
│   ├── refunds.csv
│   └── faq.csv
│
├── models/
│   └── distilbert_customer_support/
│       ├── model.safetensors
│       ├── config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── special_tokens_map.json
│       ├── label_encoder.pkl
│       └── intent_labels.json
│
├── chatbot.py
├── train.py
├── generate_dataset.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- DistilBERT
- Pandas
- NumPy
- Scikit-learn
- Regular Expressions (Regex)

---

## AI Model

The chatbot uses **DistilBERT**, a lightweight transformer model based on BERT.

DistilBERT was selected because it provides:

- High intent classification accuracy
- Faster inference compared to BERT
- Lower memory consumption
- Efficient performance on CPU systems

The model was fine-tuned on a custom customer support dataset for multi-class intent classification.

---

## Dataset

The chatbot was trained on a custom dataset containing approximately **2,000+ customer support conversations** covering twelve different customer support intents.

Additional datasets include:

- Customer Orders
- Product Database
- Refund Records
- Frequently Asked Questions (FAQ)

These datasets allow the chatbot to provide dynamic responses instead of static replies.

---

## System Workflow

```text
                User Query
                     │
                     ▼
        DistilBERT Intent Classification
                     │
                     ▼
          Confidence Score Prediction
                     │
        ┌────────────┴─────────────┐
        │                          │
 High Confidence           Low Confidence
        │                          │
        ▼                          ▼
 Database Search          Fallback Response
        │
        ▼
 Intelligent Customer Reply
```

---

## Example Queries

### Order Tracking

```
Track ORD1001
Track order 1001
Where is my order?
Order status ORD1015
```

### Product Inquiry

```
P001
Show product details
Tell me about the gaming monitor
Laptop warranty
```

### Refund

```
Refund status ORD1010
How do I request a refund?
```

### Billing

```
What payment methods do you accept?
Can I pay using UPI?
```

---

## Performance

| Metric | Result |
|---------|---------|
| Training Samples | ~2005 |
| Supported Intents | 12 |
| Model | DistilBERT |
| Validation Accuracy | ~93–95% |

---

## Key Highlights

- Uses a transformer-based NLP model instead of keyword matching.
- Combines Artificial Intelligence with structured database retrieval.
- Uses confidence scores to reduce incorrect responses.
- Implements fallback logic for uncertain predictions.
- Supports flexible Order ID and Product ID recognition.
- Modular code structure for easy future enhancements.

---

## Future Improvements

Possible future enhancements include:

- Web-based interface using Flask or Django
- SQL database integration
- Chat history storage
- Voice-based interaction
- Multi-language support
- Sentiment analysis
- Cloud deployment
- REST API integration

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Intelligent-Customer-Support-Chatbot.git
```

Move into the project directory:

```bash
cd Intelligent-Customer-Support-Chatbot
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the chatbot:

```bash
python chatbot.py
```

To retrain the model:

```bash
python train.py
```

---

## Learning Outcomes

This project demonstrates practical implementation of:

- Natural Language Processing (NLP)
- Transformer-based Language Models
- Intent Classification
- Machine Learning Model Training
- Confidence-based Decision Making
- AI-powered Customer Support Automation
- Python Application Development
- Database Integration

---

## License

This project is released under the **MIT License**.

---

## Author

**SNAYUSH Behera**

Capstone Project

Department of Computer Science & Engineering

Lovely Professional University
