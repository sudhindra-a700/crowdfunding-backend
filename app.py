import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import shap  # Keep SHAP for XAI
import numpy as np
import pandas as pd
import os
import re
import requests
import json
from datetime import datetime
import warnings
import matplotlib.pyplot as plt  # Keep matplotlib for SHAP plots
import random  # For random numbers in mock data
from sklearn.model_selection import KFold  # Added missing import
from typing import Optional  # Added missing import

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Global variables for model, tokenizer, and NGO Darpan data ---
# Initialize as None, to be loaded lazily
model = None
tokenizer = None
ngo_darpan_data = None # This will hold the DataFrame
DEVICE = "cpu"  # Default to CPU. Will be updated if CUDA is available.

# Define model output directory
MODEL_OUTPUT_DIR = "./distilbert-fraud-finetuned"
BASE_MODEL_NAME = "distilbert-base-uncased"

# Determine the base directory for data files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NGO_DARPAN_CSV_PATH = os.path.join(BASE_DIR, "DEhli.csv")
NGO_FRAUD_CSV_PATH = os.path.join(BASE_DIR, "ngo_fraud.csv")

def load_fraud_detection_model():
    """
    Loads the fine-tuned fraud detection model and tokenizer.
    If not found, loads the base model. This function is designed for lazy loading.
    It will only load if the global 'model' is None.
    """
    global model, tokenizer, DEVICE

    if model is None or tokenizer is None:
        print("Lazily loading fraud detection model...")
        try:
            # Check for GPU availability
            if torch.cuda.is_available():
                DEVICE = "cuda"
                print("CUDA (GPU) is available. Using GPU for fraud detection model.")
            else:
                DEVICE = "cpu"
                print("CUDA (GPU) is not available. Using CPU for fraud detection model.")

            if os.path.exists(MODEL_OUTPUT_DIR) and os.listdir(MODEL_OUTPUT_DIR):
                print(f"Loading fine-tuned model from {MODEL_OUTPUT_DIR}")
                tokenizer = AutoTokenizer.from_pretrained(MODEL_OUTPUT_DIR)
                model = AutoModelForSequenceClassification.from_pretrained(MODEL_OUTPUT_DIR, num_labels=2)
            else:
                print(f"No fine-tuned model found at {MODEL_OUTPUT_DIR}. Loading base model: {BASE_MODEL_NAME}")
                tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
                model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=2)

            model.to(DEVICE)
            print("Fraud detection model loaded successfully (lazy).")
        except Exception as e:
            print(f"FATAL ERROR: Could not load fraud detection model: {e}")
            model = None
            tokenizer = None
            raise RuntimeError("Fraud detection model initialization failed.")  # Propagate error


def load_ngo_darpan_data(file_path: str):
    """
    Loads NGO Darpan data from a CSV file into a pandas DataFrame.
    Caches the DataFrame for subsequent calls. This function is designed for lazy loading.
    It will only load if the global 'ngo_darpan_data' is None.
    """
    global ngo_darpan_data
    if ngo_darpan_data is None:
        print(f"Lazily loading NGO Darpan data from {file_path}...")
        try:
            if not os.path.exists(file_path):
                print(f"Warning: NGO Darpan data file not found at {file_path}. NGO Darpan lookup will be unavailable.")
                ngo_darpan_data = pd.DataFrame() # Set to empty DataFrame to avoid None
                return ngo_darpan_data

            ngo_darpan_data = pd.read_csv(file_path)
            print(f"NGO Darpan data loaded successfully from {file_path}. Shape: {ngo_darpan_data.shape}")
        except Exception as e:
            print(f"Error loading NGO Darpan data from {file_path}: {e}")
            ngo_darpan_data = pd.DataFrame()  # Ensure it's an empty DataFrame on error
            raise RuntimeError(f"NGO Darpan data loading failed: {e}")  # Propagate error
    return ngo_darpan_data


def search_ngo_darpan_csv(ngo_darpan_id: str) -> dict:
    """
    Searches the loaded NGO Darpan CSV data for a given NGO Darpan ID.
    Returns a dictionary of relevant details if found, otherwise an empty dict.
    Assumes ngo_darpan_data is already loaded globally.
    """
    global ngo_darpan_data
    if ngo_darpan_data is None or ngo_darpan_data.empty or 'Unique ID of VO/ NGO' not in ngo_darpan_data.columns:
        print("Warning: NGO Darpan data not loaded or empty. Cannot perform search.")
        return {}

    search_id = str(ngo_darpan_id).strip()
    result = ngo_darpan_data[ngo_darpan_data['Unique ID of VO/ NGO'].astype(str).str.strip() == search_id]

    if not result.empty:
        return result.iloc[0].to_dict()
    return {}


# Custom Dataset for fine-tuning
class FraudDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def fine_tune_model(dataset_path: str = NGO_FRAUD_CSV_PATH, k_folds: int = 1):
    """
    Fine-tunes the DistilBERT model for fraud detection.
    Uses k-fold cross-validation. Only runs if fine-tuned model is not found.
    """
    global model, tokenizer  # Ensure global model/tokenizer are used/updated

    if os.path.exists(MODEL_OUTPUT_DIR) and os.listdir(MODEL_OUTPUT_DIR):
        print("Model already fine-tuned. Skipping fine-tuning.")
        # Ensure the global model and tokenizer are set to the fine-tuned version
        load_fraud_detection_model()  # This will load the fine-tuned model if not already
        return

    print(f"Loading dataset from {dataset_path} for fine-tuning...")
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}. Cannot fine-tune model.")
        return

    if 'description' not in df.columns or 'label' not in df.columns:
        if 'text' in df.columns:
            df['description'] = df['text']
        else:
            print("Error: Dataset must contain 'description' (or 'text') and 'label' columns. Creating dummy data.")
            dummy_data = {
                'description': [
                    "Legitimate charity raising funds for education.",
                    "Invest now for guaranteed 100% returns in 24 hours!",
                    "Building sustainable communities through local initiatives.",
                    "Urgent! Send money to this account for a secret prize.",
                    "Non-profit organization dedicated to environmental conservation.",
                ],
                'label': [0, 1, 0, 1, 0]
            }
            df = pd.DataFrame(dummy_data)

    dataset = Dataset.from_pandas(df)

    # Re-initialize tokenizer and model to ensure they are clean for fine-tuning
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=2)
    model.to(DEVICE)  # Ensure model is on the correct device

    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(examples["description"], padding="max_length", truncation=True, max_length=128),
        batched=True
    )
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": (predictions == labels).mean()}

    for fold, (train_index, val_index) in enumerate(kf.split(tokenized_dataset)):
        print(f"--- Training Fold {fold + 1}/{k_folds} ---")
        train_dataset = tokenized_dataset.select(train_index)
        val_dataset = tokenized_dataset.select(val_index)

        training_args = TrainingArguments(
            output_dir=f"./results_fold_{fold}",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"./logs_fold_{fold}",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        print(f"Finished training fold {fold + 1}.")

    model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"Fine-tuned model saved to {MODEL_OUTPUT_DIR}")


def predict_fraud(organization_data: dict, api_key_trustcheckr: Optional[str] = None) -> tuple:
    """
    Predicts fraud score and provides explanation for an organization.
    Integrates with mock TrustCheckr and other verification services, including NGO Darpan CSV lookup.
    Returns fraud_score, explanation, plot_path, and verification_details.
    """
    global model, tokenizer, ngo_darpan_data # Ensure global model/tokenizer/data are used

    # Ensure model is loaded before prediction (this call will only load if model is None)
    # This is kept for robustness, although it should be loaded at module import now.
    load_fraud_detection_model()

    # If model is still None after attempting to load, raise an error
    if model is None or tokenizer is None:
        raise RuntimeError("Fraud detection model is not loaded. Cannot perform prediction.")

    text = organization_data.get('recent_posts', '') + " " + organization_data.get('bio',
                                                                                   '') + " " + organization_data.get(
        'description', '')

    # Model prediction
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128).to(DEVICE)
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=1)[0]
    fraud_score = probabilities[1].item()  # Probability of being fraud (label 1)

    # --- Generate Explanation (keeping XAI) ---
    explanation = "AI analysis based on text content and organizational data. "
    if fraud_score > 0.7:
        explanation += "High likelihood of fraud detected. Suspicious language patterns and/or lack of verifiable organizational details contribute to this score."
    elif fraud_score > 0.4:
        explanation += "Moderate risk of fraud detected. Some inconsistencies or less verifiable information were found. Manual review is recommended."
    else:
        explanation += "Low likelihood of fraud. Information appears consistent and verifiable."

    # --- Generate SHAP Plot (keeping XAI) ---
    plot_path = None  # Default to None if plot generation fails or is not desired
    try:
        # For text classification, SHAP on token embeddings is more complex.
        # A simplified approach for demonstration:
        # Create a dummy plot to satisfy the return signature
        plt.figure(figsize=(2, 2))
        plt.text(0.5, 0.5, f"SHAP Plot for {organization_data.get('org_name', 'Campaign')}", ha='center', va='center',
                 wrap=True)
        plt.axis('off')

        # Ensure the directory exists relative to the script
        script_dir = os.path.dirname(__file__)
        shap_plots_dir = os.path.join(script_dir, "static", "shap_plots")
        os.makedirs(shap_plots_dir, exist_ok=True)

        plot_filename = f"shap_plot_{organization_data.get('org_name', 'unknown').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        plot_path = os.path.join("/static/shap_plots", plot_filename)  # Path for web access (relative to base URL)
        full_plot_path = os.path.join(shap_plots_dir, plot_filename)  # Absolute path for saving

        plt.savefig(full_plot_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"Dummy SHAP plot saved to {full_plot_path}")

    except Exception as e:
        print(f"Warning: Could not generate SHAP plot: {e}")
        plot_path = "/static/dummy_shap_plot.png"  # Fallback placeholder for web access

    # --- Verification Details (based on provided data and NGO Darpan lookup) ---
    # Ensure NGO Darpan data is loaded before search (this call will only load if ngo_darpan_data is None)
    # This is kept for robustness, although it should be loaded at module import now.
    load_ngo_darpan_data(NGO_DARPAN_CSV_PATH) # Use the global path

    # If NGO Darpan data is still None/empty after attempting to load, handle gracefully
    if ngo_darpan_data is None or ngo_darpan_data.empty:
        print("Warning: NGO Darpan data not available for lookup.")


    verification_details = {
        'org_name': organization_data.get('org_name', 'N/A'),
        'pan_provided': bool(organization_data.get('pan')),
        'reg_number_provided': bool(organization_data.get('reg_number')),
        'registration_type_provided': bool(organization_data.get('registration_type')),
        'ngo_darpan_id_provided': bool(organization_data.get('ngo_darpan_id')),
        'fcra_number_provided': bool(organization_data.get('fcra_number')),
        'ngo_darpan_lookup_status': 'Not Performed',
        'ngo_darpan_details': {},
        'trustcheckr_score': None,
        'social_media_verified': False,  # Placeholder
        'issues': []
    }

    # Perform NGO Darpan lookup
    if organization_data.get('ngo_darpan_id'):
        # Call search_ngo_darpan_csv without passing file_path, as data is global
        darpan_details = search_ngo_darpan_csv(organization_data['ngo_darpan_id'])
        if darpan_details:
            verification_details['ngo_darpan_lookup_status'] = 'Found in CSV'
            verification_details['ngo_darpan_details'] = darpan_details
        else:
            verification_details['ngo_darpan_lookup_status'] = 'Not Found in CSV'
            verification_details['issues'].append('NGO Darpan ID not found in local data')
    else:
        verification_details['ngo_darpan_lookup_status'] = 'No ID Provided'

    # Mock TrustCheckr API call (keeping for consistency)
    if api_key_trustcheckr and api_key_trustcheckr != "mock_trustcheckr_key":
        mock_score = 0.5  # Default
        if verification_details['pan_provided']: mock_score -= 0.1
        if verification_details['ngo_darpan_lookup_status'] == 'Found in CSV': mock_score -= 0.2
        if len(verification_details['issues']) > 0: mock_score += 0.3
        verification_details['trustcheckr_score'] = max(0.0, min(1.0, mock_score))
    else:
        verification_details['trustcheckr_score'] = random.uniform(0.1, 0.9)  # Random score for mock

    # Adjust fraud score based on verification results (mock logic)
    if verification_details['ngo_darpan_lookup_status'] == 'Not Found in CSV':
        fraud_score = min(1.0, fraud_score + 0.15)  # Increase fraud score if NGO Darpan ID is not found

    if verification_details['trustcheckr_score'] is not None:
        fraud_score = (fraud_score * 0.6) + (verification_details['trustcheckr_score'] * 0.4)

    # Final fraud score clamping
    fraud_score = max(0.0, min(1.0, fraud_score))

    return fraud_score, explanation, plot_path, verification_details

# --- Module-level loading to ensure models/data are loaded once per worker ---
# This code runs when the fraud_detection.py module is imported by each Gunicorn worker.
# It ensures lazy loading happens at worker boot, not on first request.
try:
    load_fraud_detection_model()
    print("Fraud detection model pre-loaded at module import.")
except RuntimeError as e:
    print(f"Warning: Fraud detection model pre-loading failed at module import: {e}")
    # The model will remain None, and predict_fraud will raise an error if called.

try:
    load_ngo_darpan_data(NGO_DARPAN_CSV_PATH)
    print("NGO Darpan data pre-loaded at module import.")
except RuntimeError as e:
    print(f"Warning: NGO Darpan data pre-loading failed at module import: {e}")
    # The data will remain None/empty, and search_ngo_darpan_csv will handle it.


if __name__ == "__main__":
    # For local testing, ensure paths are correct
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    ngo_darpan_test_path = os.path.join(current_script_dir, "DEhli.csv")
    ngo_fraud_test_path = os.path.join(current_script_dir, "ngo_fraud.csv")

    # Fine-tune model (only if not already fine-tuned)
    # This will use the pre-loaded model/tokenizer if available, or load base.
    # If fine-tuning is needed, it will save to MODEL_OUTPUT_DIR.
    fine_tune_model(dataset_path=ngo_fraud_test_path, k_folds=1)  # Reduced folds for quicker local test

    # Example prediction (will use the pre-loaded model/data)
    sample_org_legit = {
        'org_name': 'Shiksha Foundation',
        'bio': 'Dedicated to providing education for underprivileged children in rural India.',
        'follower_count': 10000,
        'post_count': 200,
        'account_age_days': 730,
        'engagement_rate': 0.03,
        'recent_posts': 'Join our tree planting drive next month!',
        'pan': 'ABCDE1234F',
        'registration_type': 'Trust',
        'registration_number': 'T1234567890',
        'ngo_darpan_id': 'DL/2017/0165260',  # Example from DEhli.csv
        'fcra_number': '1234567890'
    }
    fraud_score, explanation, plot_path, verification = predict_fraud(sample_org_legit, api_key_trustcheckr="test_key")
    print("\n--- Legit Organization Prediction ---")
    print(f"Fraud Score: {fraud_score:.2f}")
    print(f"Explanation: {explanation}")
    print(f"SHAP Plot Path: {plot_path}")
    print(f"Verification: {json.dumps(verification, indent=2)}")

    sample_org_fraud = {
        'org_name': 'CryptoGold Investments',
        'bio': 'Guaranteed daily returns! Invest in our exclusive crypto arbitrage bots.',
        'follower_count': 50,
        'post_count': 5,
        'account_age_days': 10,
        'engagement_rate': 0.1,
        'recent_posts': 'Last chance to get rich! DM us now!',
        'pan': 'INVALIDPAN12',
        'registration_type': '',  # No specific type
        'registration_number': '',
        'ngo_darpan_id': 'INVALIDNGOID',  # Will not be found in CSV
        'fcra_number': ''
    }
    fraud_score, explanation, plot_path, verification = predict_fraud(sample_org_fraud, api_key_trustcheckr="test_key")
    print("\n--- Fraudulent Organization Prediction ---")
    print(f"Fraud Score: {fraud_score:.2f}")
    print(f"Explanation: {explanation}")
    print(f"SHAP Plot Path: {plot_path}")
    print(f"Verification: {json.dumps(verification, indent=2)}")
