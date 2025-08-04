import torch
import lightgbm as lgb
import xgboost as xgb
import shap
import numpy as np
import pandas as pd
import os
import re
import json
import uuid
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Optional, Dict, Any, Tuple
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Global variables and constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
NGO_DARPAN_CSV_PATH = os.path.join(BASE_DIR, "DEhli.csv")
NGO_FRAUD_CSV_PATH = os.path.join(BASE_DIR, "ngo_fraud.csv")
MODEL_LGBM_PATH = os.path.join(BASE_DIR, "lgbm_fraud_model.joblib")
MODEL_XGB_PATH = os.path.join(BASE_DIR, "xgb_fraud_model.joblib")

# Global instances, to be loaded lazily
ngo_darpan_data = None
lgbm_model = None
xgb_model = None
feature_columns = []
label_encoder_dict = {}

def load_ngo_darpan_data() -> pd.DataFrame:
    """
    Loads and preprocesses the NGO Darpan data from a CSV file.
    Designed for lazy loading and singleton pattern.
    """
    global ngo_darpan_data
    if ngo_darpan_data is not None:
        return ngo_darpan_data

    try:
        ngo_darpan_data = pd.read_csv(NGO_DARPAN_CSV_PATH)
        ngo_darpan_data.rename(columns={'NGO Darpan ID': 'ngo_darpan_id', 'FCRA Number': 'fcra_number'}, inplace=True)
        print(f"Successfully loaded NGO Darpan data from {NGO_DARPAN_CSV_PATH}")
        return ngo_darpan_data
    except FileNotFoundError:
        print(f"NGO Darpan data not found at {NGO_DARPAN_CSV_PATH}. Mocking data.")
        mock_data = {
            'ngo_darpan_id': [f'NGO{i:05d}' for i in range(100)],
            'Name of VO/NGO': [f'Mock NGO {i}' for i in range(100)],
            'fcra_number': [f'FCRA{random.randint(1000000000, 9999999999)}' for _ in range(100)]
        }
        ngo_darpan_data = pd.DataFrame(mock_data)
        return ngo_darpan_data

def load_and_train_models(force_retrain: bool = False):
    """
    Loads trained models from disk, or trains them if they don't exist.
    """
    global lgbm_model, xgb_model, feature_columns, label_encoder_dict
    
    if lgbm_model is not None and xgb_model is not None and not force_retrain:
        print("Models already loaded. Skipping training.")
        return

    # Check if pre-trained models exist
    if not force_retrain and os.path.exists(MODEL_LGBM_PATH) and os.path.exists(MODEL_XGB_PATH):
        try:
            lgbm_model = joblib.load(MODEL_LGBM_PATH)
            xgb_model = joblib.load(MODEL_XGB_PATH)
            print("Successfully loaded pre-trained LightGBM and XGBoost models.")
            # Load feature columns and encoders (assuming they were saved)
            # For this example, we will regenerate them based on the data.
        except Exception as e:
            print(f"Failed to load models: {e}. Retraining...")
            force_retrain = True

    if force_retrain or lgbm_model is None or xgb_model is None:
        print("Loading data and training new models...")
        try:
            df = pd.read_csv(NGO_FRAUD_CSV_PATH)
        except FileNotFoundError:
            print(f"Training data not found at {NGO_FRAUD_CSV_PATH}. Mocking data for training.")
            mock_data = {
                'campaign_name': [f'Campaign {i}' for i in range(500)],
                'description': [f'Description for campaign {i}' for i in range(500)],
                'org_name': [f'Org {i}' for i in range(500)],
                'category': [random.choice(['Education', 'Health', 'Community', 'Technology']) for _ in range(500)],
                'ngo_darpan_id': [f'NGO{random.randint(0, 1000):05d}' for _ in range(500)],
                'pan': [f'ABCDE{random.randint(1000, 9999)}F' for _ in range(500)],
                'has_certificate': [random.choice([True, False]) for _ in range(500)],
                'account_age': [random.randint(1, 365) for _ in range(500)],
                'label': [0] * 450 + [1] * 50
            }
            df = pd.DataFrame(mock_data)
        
        # Feature Engineering: Combine text features (a simple approach)
        df['text_features'] = df['campaign_name'] + " " + df['description']
        
        # Select features to use for the model
        feature_columns = ['account_age', 'has_certificate']
        categorical_columns = ['category']
        
        # Label Encoding for categorical features
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoder_dict[col] = le
            feature_columns.append(col)

        X = df[feature_columns]
        y = df['label']
        
        # Split data for training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train LightGBM model
        print("Training LightGBM model...")
        lgbm_model = lgb.LGBMClassifier(random_state=42)
        lgbm_model.fit(X_train, y_train)
        joblib.dump(lgbm_model, MODEL_LGBM_PATH)

        # Train XGBoost model
        print("Training XGBoost model...")
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        xgb_model.fit(X_train, y_train)
        joblib.dump(xgb_model, MODEL_XGB_PATH)
        
        print("Model training complete.")

class AdvancedFraudModerationSystem:
    def __init__(self):
        """Initializes the fraud moderation system."""
        load_and_train_models()
        self.lgbm_model = lgbm_model
        self.xgb_model = xgb_model
        self.ngo_data = load_ngo_darpan_data()
        self.feature_columns = feature_columns
        self.label_encoder_dict = label_encoder_dict

    def preprocess_campaign_data(self, campaign_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocesses a single campaign dictionary into a DataFrame row
        that the model can use for prediction.
        """
        # Create a dictionary with features the model expects
        data_for_df = {
            'account_age': [campaign_data.get('account_age', 30)],  # Default to 30 days
            'has_certificate': [campaign_data.get('has_certificate', False)],
            'category': [campaign_data.get('category', 'Community')]
        }
        
        df = pd.DataFrame(data_for_df)
        
        # Apply the same label encoding as during training
        for col, le in self.label_encoder_dict.items():
            if col in df.columns:
                # Handle unseen labels by assigning a new, default value
                try:
                    df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
                except ValueError:
                    df[col] = -1
        
        return df[self.feature_columns]

    def predict_fraud(self, campaign_data: Dict[str, Any]) -> Tuple[float, str, Optional[str], Dict]:
        """
        Predicts a fraud score for a given campaign and provides an explanation.
        Uses the trained LightGBM model and SHAP.
        """
        # Preprocess the campaign data
        preprocessed_data = self.preprocess_campaign_data(campaign_data)
        
        # Get fraud score from LightGBM model
        fraud_score = self.lgbm_model.predict_proba(preprocessed_data)[:, 1][0]
        
        # Generate SHAP explanation
        explanation = "Fraud score explanation not available."
        shap_plot_path = None
        try:
            explainer = shap.TreeExplainer(self.lgbm_model)
            shap_values = explainer.shap_values(preprocessed_data)
            
            # Create a more detailed explanation
            if isinstance(shap_values, list): # For multiclass output
                shap_values = shap_values[1] # Take the values for the "fraud" class
            
            # Get feature contributions
            feature_contributions = {}
            for i, feature in enumerate(self.feature_columns):
                feature_contributions[feature] = shap_values[0, i]
            
            # Sort contributions to find the most important features
            sorted_contributions = sorted(feature_contributions.items(), key=lambda item: abs(item[1]), reverse=True)
            
            explanation_parts = []
            for feature, contribution in sorted_contributions[:3]: # Top 3 features
                if contribution > 0:
                    explanation_parts.append(f"High '{feature}' value increased the fraud score.")
                else:
                    explanation_parts.append(f"Low '{feature}' value decreased the fraud score.")

            explanation = " ".join(explanation_parts)
            
            # Create a mock SHAP plot (as the environment may not have a display)
            plt.figure()
            shap.summary_plot(shap_values, preprocessed_data, feature_names=self.feature_columns, show=False)
            plot_filename = f"shap_plot_{uuid.uuid4().hex}.png"
            shap_plot_path = os.path.join("/tmp", plot_filename)
            plt.savefig(shap_plot_path, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Could not generate SHAP explanation or plot: {e}")
            explanation = "Failed to generate detailed explanation."
            shap_plot_path = None

        # Check NGO Darpan ID separately
        ngo_darpan_id = campaign_data.get('ngo_darpan_id', '')
        ngo_verification = {"is_verified": False, "details": "No NGO Darpan ID provided."}
        if ngo_darpan_id:
            if ngo_darpan_id in self.ngo_data['ngo_darpan_id'].values:
                ngo_verification = {"is_verified": True, "details": f"NGO Darpan ID '{ngo_darpan_id}' found in records."}
            else:
                ngo_verification = {"is_verified": False, "details": f"NGO Darpan ID '{ngo_darpan_id}' not found."}
        
        return fraud_score, explanation, shap_plot_path, ngo_verification

    def process_new_campaign(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes a new campaign for fraud and returns a moderation status.
        """
        print(f"Processing new campaign: {campaign_data.get('title', 'Untitled')}")
        fraud_score, explanation, _, verification_details = self.predict_fraud(campaign_data)
        
        moderation_status = "pending"
        if fraud_score < 0.3:
            moderation_status = "approved"
        elif fraud_score >= 0.3 and fraud_score < 0.7:
            moderation_status = "review_needed"
        else:
            moderation_status = "rejected"
            
        return {
            "status": moderation_status,
            "fraud_score": float(fraud_score), # Ensure it's a standard float
            "explanation": explanation,
            "verification_details": verification_details
        }

if __name__ == "__main__":
    # --- Example Usage ---
    
    # Force re-training the models for a fresh start
    load_and_train_models(force_retrain=True)

    # Initialize the system
    moderation_system = AdvancedFraudModerationSystem()

    # Sample a legitimate-looking campaign
    sample_legit = {
        'title': 'Fundraising for local school supplies',
        'description': 'Providing books and stationary to underprivileged students in the community.',
        'organization': 'Hope for All Foundation',
        'ngo_darpan_id': 'NGO00010',  # Assuming this exists in the mock data
        'pan_number': 'ABCDE1234F',
        'has_certificate': True,
        'donors_count': 150,
        'created_at': '2024-01-15T12:00:00',
        'category': 'Education',
        'account_age': 150
    }
    print("--- Processing a legitimate campaign ---")
    result_legit = moderation_system.process_new_campaign(sample_legit)
    print(json.dumps(result_legit, indent=2))
    print("-" * 30)

    # Sample a suspicious-looking campaign
    sample_suspicious = {
        'title': 'High-yield investment for a quick profit',
        'description': 'Guaranteed daily returns! Invest in our exclusive crypto arbitrage bots. Limited time offer.',
        'organization': 'CryptoGold Investments',
        'ngo_darpan_id': 'INVALIDNGOID123',
        'pan_number': 'PANINVALID',
        'has_certificate': False,
        'donors_count': 5,
        'created_at': '2024-07-25T08:30:00',
        'category': 'Technology',
        'account_age': 5
    }
    print("--- Processing a suspicious campaign ---")
    result_suspicious = moderation_system.process_new_campaign(sample_suspicious)
    print(json.dumps(result_suspicious, indent=2))
    print("-" * 30)
