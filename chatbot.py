# ==========================================================
# AI Customer Support Chatbot
# Phase 3 - Part 1
# Load trained model, tokenizer, labels and datasets
# ==========================================================

import os
import json
import pickle
import re
import pandas as pd
import torch

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

# ----------------------------------------------------------
# Project Paths
# ----------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "distilbert_customer_support"
)

DATA_PATH = os.path.join(BASE_DIR, "data")

# ----------------------------------------------------------
# Load AI Model
# ----------------------------------------------------------

print("Loading trained model...")

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)

model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

print("Model loaded successfully.")

# ----------------------------------------------------------
# Load Intent Labels
# ----------------------------------------------------------

with open(
    os.path.join(MODEL_PATH, "intent_labels.json"),
    "r"
) as file:
    intent_labels = json.load(file)

with open(
    os.path.join(MODEL_PATH, "label_encoder.pkl"),
    "rb"
) as file:
    label_encoder = pickle.load(file)

# ----------------------------------------------------------
# Load CSV Databases
# ----------------------------------------------------------

print("Loading datasets...")

orders_df = pd.read_csv(
    os.path.join(DATA_PATH, "customer_orders.csv")
)

products_df = pd.read_csv(
    os.path.join(DATA_PATH, "products.csv")
)

refunds_df = pd.read_csv(
    os.path.join(DATA_PATH, "refunds.csv")
)

faq_df = pd.read_csv(
    os.path.join(DATA_PATH, "faq.csv")
)

support_df = pd.read_csv(
    os.path.join(DATA_PATH, "customer_support_dataset.csv")
)

print("All datasets loaded successfully.")

# ----------------------------------------------------------
# Display Dataset Information
# ----------------------------------------------------------

print("\n========== DATA SUMMARY ==========")

print(f"Orders    : {len(orders_df)} records")
print(f"Products  : {len(products_df)} records")
print(f"Refunds   : {len(refunds_df)} records")
print(f"FAQs       : {len(faq_df)} records")
print(f"Training   : {len(support_df)} samples")

print("==================================")

# ----------------------------------------------------------
# Confidence Threshold
# If prediction confidence falls below this value,
# the chatbot will transfer the user to a human agent.
# ----------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.55



# -----------------------------
# Predict user intent
# -----------------------------
def predict_intent(text):

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"]
        )

    probabilities = torch.softmax(outputs.logits, dim=1)

    confidence, prediction = torch.max(probabilities, dim=1)

    intent = label_encoder.inverse_transform(
        [prediction.item()]
    )[0]

    return intent, confidence.item()


# -----------------------------
# Search Orders CSV
# -----------------------------


def search_orders(query):

    order = re.search(r"(?:ORD)?\s*(\d{4})", query.upper())

    if not order:
        return None

    order_id = "ORD" + order.group(1)

    result = orders_df[
        orders_df["Order_ID"].str.upper() == order_id
    ]

    if not result.empty:
        return result.iloc[0].to_dict()

    return None

# -----------------------------
# Search Products CSV
# -----------------------------
def search_products(query):

    query = query.lower()

    for _, row in products_df.iterrows():

        product = str(row["Product_Name"]).lower()
        brand = str(row["Brand"]).lower()
        category = str(row["Category"]).lower()
        product_id = str(row["Product_ID"]).lower()

        if (
            product in query
            or brand in query
            or category in query
            or product_id in query
        ):
            return row.to_dict()

    return None

# -----------------------------
# Search Refund CSV
# -----------------------------

def search_refunds(query):

    order = re.search(r"(?:ORD)?\s*(\d{4})", query.upper())

    if not order:
        return None

    order_id = "ORD" + order.group(1)

    result = refunds_df[
        refunds_df["Order_ID"].str.upper() == order_id
    ]

    if not result.empty:
        return result.iloc[0].to_dict()

    return None

# -----------------------------
# Search FAQ CSV
# -----------------------------
def search_faq(query):

    query = query.lower()

    # Shipping
    if "shipping" in query or "delivery" in query:
        return faq_df.iloc[1].to_dict()

    # Return
    if "return" in query:
        return faq_df.iloc[0].to_dict()

    # Refund
    if "refund" in query:
        return faq_df.iloc[6].to_dict()

    # Track Order
    if "track" in query or "order" in query:
        return faq_df.iloc[3].to_dict()

    # Payment
    if "payment" in query or "upi" in query or "card" in query:
        return faq_df.iloc[4].to_dict()

    # Cancel
    if "cancel" in query:
        return faq_df.iloc[5].to_dict()

    # COD
    if "cash" in query or "cod" in query:
        return faq_df.iloc[7].to_dict()

    # Support
    if "support" in query or "contact" in query:
        return faq_df.iloc[8].to_dict()

    # Warranty
    if "warranty" in query:
        return faq_df.iloc[9].to_dict()

    return None
    


def search_database(intent, query):

    if intent == "order_tracking":

        result = search_orders(query)

        if result:
            return (
                f"Order ID       : {result['Order_ID']}\n"
                f"Customer       : {result['Name']}\n"
                f"Product        : {result['Product']}\n"
                f"Amount         : ₹{result['Amount (₹)']}\n"
                f"Payment Status : {result['Payment_Status']}\n"
                f"Payment Method : {result['Payment_Method']}\n"
                f"Order Date     : {result['Order_Date']}"
            )
        else:
            return "Order not found. Please enter a valid Order ID (e.g., ORD1001)."
    elif intent == "cancel_order":

        faq = search_faq(query)

        if faq:
            return faq["Answer"]

        return "Orders can only be cancelled before they are shipped."    
    
    elif intent == "product_inquiry":

        result = search_products(query)

        if result:
            return (
                f"Product ID : {result['Product_ID']}\n"
                f"Product    : {result['Product_Name']}\n"
                f"Brand      : {result['Brand']}\n"
                f"Category   : {result['Category']}\n"
                f"Price      : ₹{result['Price']}\n"
                f"Stock      : {result['Stock']}\n"
                f"Warranty   : {result['Warranty']}"
            )

        return "Product not found. Please enter a valid product name."

    elif intent == "refund":

        result = search_refunds(query)

        if result:
            return (
                f"Refund ID     : {result['Refund_ID']}\n"
                f"Order ID      : {result['Order_ID']}\n"
                f"Reason        : {result['Reason']}\n"
                f"Refund Status : {result['Refund_Status']}\n"
                f"Amount        : ₹{result['Refund_Amount']}\n"
                f"Requested On  : {result['Request_Date']}"
            )

        # No Order ID found -> Search FAQ
        faq = search_faq(query)

        if faq:
            return faq["Answer"]

        return "Refund record not found. Please provide a valid Order ID."

    elif intent == "greeting":
        return "Hello! 👋 How can I help you today?"

    elif intent == "human_handoff":
        return "Sure! I'm connecting you with one of our customer support executives."

    # Search FAQ if nothing matched above
    result = search_faq(query)

    if result:
        return result["Answer"]
        

    return (
    "I'm sorry, I couldn't find the required information. "
    "Please contact customer support."
        )

def chatbot():
    print("\n" + "=" * 60)
    print("      AI CUSTOMER SUPPORT CHATBOT")
    print("=" * 60)
    print("Type 'exit' anytime to quit.\n")

    while True:
        user_input = input("You : ").strip()

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nBot : Thank you for contacting Customer Support.")
            print("Have a great day!")
            break
# -----------------------------------------
# Direct Product ID Detection
# -----------------------------------------

        if re.search(r"P\d{3}", user_input.upper()):

            response = search_database(
            "product_inquiry",
            user_input
        )

            print("\nDetected Intent : product_inquiry")
            print("Confidence      : 100.00%")

            print("\nBot :", response)
            print("-" * 60)

            continue
# -------------------------------------------------
# Direct Order ID Detection
# -------------------------------------------------

        if re.search(r"(?:ORD)?\s*\d{4}", user_input.upper()):
            response = search_database("order_tracking", user_input)

            print("\nDetected Intent : order_tracking")
            print("Confidence      : 100.00%")

            print("\nBot :", response)
            print("-" * 60)

            continue        

        intent, confidence = predict_intent(user_input)

        print(f"\nDetected Intent : {intent}")
        print(f"Confidence      : {confidence:.2%}")

        # Confidence too low
        if confidence < CONFIDENCE_THRESHOLD:
            print("\nBot : I'm not completely sure I understood your question.")

            refund_words = [
                "refund",
                "return",
                "money",
                "cancel",
                "replace"
            ]

            if any(word in user_input.lower() for word in refund_words):
                print("Bot : It looks like your question is related to refunds or returns.")
                print("Bot : Please check the Refund Policy or contact our support team.")
            else:
                print("Bot : Could you please rephrase your question?")
                print("Bot : If your issue continues, I'll connect you with customer support.")

            print("-" * 60)
            continue

        response = search_database(intent, user_input)

        print("\nBot :", response)
        print("-" * 60)


if __name__ == "__main__":
    chatbot()