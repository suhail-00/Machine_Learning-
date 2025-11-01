import joblib
import pandas as pd
import numpy as np
import random
import os
from collections import OrderedDict
import time 

# --- Configuration for Simulation ---
MODEL_FILE = "final_malware_model.joblib"
SCHEMA_FILE = "deployment_features_schema.txt"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# --- GLOBAL MOCK OBJECT ---
class MockModel:
    """A mock classifier used when joblib.load fails in the simulation."""
    def predict(self, X):
        # X[0, 2] is the Entropy_Section_Score in the input vector
        if X[0, 2] > 7.5:
            return ["boatnet.ppc"]
        else:
            return ["clean_file"]

    def predict_proba(self, X):
        return np.array([[0.9, 0.1]]) 

# --- PHASE 18: PRODUCTION MONITORING AND DRIFT CLASS ---

class DataDriftMonitor:
    """Monitors incoming feature statistics against historical training data."""
    def __init__(self, feature_name, training_mean, training_std, drift_threshold_std=3.0):
        self.feature_name = feature_name
        self.training_mean = training_mean
        self.training_std = training_std
        self.drift_threshold_std = drift_threshold_std
        self.incoming_values = []
        self.monitoring_window_size = 5 

    def log_value(self, value):
        """Logs a new feature value and checks for drift."""
        self.incoming_values.append(value)
        if len(self.incoming_values) >= self.monitoring_window_size:
            self._check_drift()
            # Clear window to allow for a fresh check (or pop(0) for sliding window)
            self.incoming_values = [] 
    
    def _check_drift(self):
        """Calculates the current mean and compares it to the training mean."""
        current_mean = np.mean(self.incoming_values)
        
        # Calculate Z-score: How many standard deviations is the current mean from the training mean?
        # Handle division by zero just in case
        z_score = abs(current_mean - self.training_mean) / self.training_std if self.training_std > 0 else 0
        
        print(f"\n[*] Drift Check: {self.feature_name} (Window Mean: {current_mean:.4f})")
        print(f"    - Z-Score (Std Deviations): {z_score:.2f} / {self.drift_threshold_std:.1f}")

        if z_score > self.drift_threshold_std:
            print(f"\n[!!!] DATA DRIFT ALERT TRIGGERED [!!!]")
            print(f"The average {self.feature_name} has shifted by >{self.drift_threshold_std} standard deviations.")
            print("Action Required: Retrain model or investigate new attack patterns.")
            print("-" * 50)
        else:
            print("[+] Drift Check: All clear.")


# Simulating training statistics for TextSection_Entropy (from Phase 2)
TRAINING_STATS = {
    'TextSection_Entropy': {
        'mean': 6.5, 
        'std': 0.5 
    }
}


# --- PHASE 17: VALIDATION SCHEMA ---
VALIDATION_SCHEMA = {
    'TimeDateStamp': {'dtype': np.number, 'min': 1000000000, 'max': 2000000000, 'required': True},
    'TextSection_Entropy': {'dtype': np.number, 'min': 0.0, 'max': 8.0, 'required': True},
    'Entropy_Section_Score': {'dtype': np.number, 'min': 0.0, 'max': 20.0, 'required': True}
}

def validate_features(raw_features):
    """
    Validates the raw feature dictionary against the predefined schema.
    Returns True and None if valid, or False and an error message if invalid.
    """
    errors = []
    
    for name, spec in VALIDATION_SCHEMA.items():
        value = raw_features.get(name)
        
        if spec['required'] and value is None:
            errors.append(f"Missing required feature: {name}")
            continue

        if not np.issubdtype(type(value), spec['dtype']):
            errors.append(f"Invalid type for {name}: Expected {spec['dtype']}, got {type(value)}")

        if 'min' in spec and value < spec['min']:
            errors.append(f"Value too low for {name}: {value} < {spec['min']}")
        
        if 'max' in spec and value > spec['max']:
            errors.append(f"Value too high for {name}: {value} > {spec['max']} (Possible Data Drift or Packing)")
            
    if errors:
        return False, "; ".join(errors)
    return True, None


# --- Utility Functions ---

def load_deployment_schema(schema_file):
    """Loads the ordered list of feature names that the model expects."""
    feature_names = ['TimeDateStamp', 'TextSection_Entropy', 'Entropy_Section_Score']
    return feature_names

def parse_pe_file_for_deployment(file_path, entropy_value):
    """Simulates feature extraction with a fixed, provided entropy value."""
    tds = random.randint(1600000001, 1900000000)
    entropy = entropy_value
    entropy_section_score = entropy * 2 + np.sin(tds * 1e-9)
    
    raw_features = OrderedDict([
        ('TimeDateStamp', tds),
        ('TextSection_Entropy', entropy),
        ('Entropy_Section_Score', entropy_section_score)
    ])
    
    print(f"    - Extracted Features: TextSection_Entropy={raw_features['TextSection_Entropy']:.4f}")
    return raw_features


def run_deployment_service():
    print("=" * 70)
    print(" PHASE 18: PRODUCTION MONITORING AND DRIFT SIMULATION ")
    print("=" * 70)

    # Initialize the Drift Monitor for the critical entropy feature
    monitor = DataDriftMonitor(
        feature_name='TextSection_Entropy',
        training_mean=TRAINING_STATS['TextSection_Entropy']['mean'],
        training_std=TRAINING_STATS['TextSection_Entropy']['std']
    )

    # Step 1: Mock Load Artifacts
    try:
        model = joblib.load(MODEL_FILE) 
    except FileNotFoundError:
        model = MockModel() 
    except Exception as e:
        print(f"[FATAL ERROR] Could not load model: {e}")
        return
        
    feature_schema = load_deployment_schema("deployment_features_schema.txt")
    print(f"[*] Monitoring {monitor.feature_name} (Mean: {monitor.training_mean:.2f}, Std: {monitor.training_std:.2f})")
    
    # --- SIMULATION LOOP: CRITICAL DRIFT TEST ---
    print("\n" + "=" * 20 + " CRITICAL DRIFT TEST " + "=" * 20)
    
    # We simulate an extreme sequence of high-entropy files (7.8 to 8.0)
    # This will dramatically raise the mean and cross the 3.0 sigma threshold.
    entropy_sequence_drift = [
        random.uniform(7.8, 8.0), 
        random.uniform(7.9, 8.0), 
        random.uniform(7.9, 8.0), 
        random.uniform(7.9, 8.0), 
        random.uniform(7.9, 8.0), # This fifth sample will trigger the drift check
    ]
    
    for i, entropy_val in enumerate(entropy_sequence_drift):
        file_name = f"critical_sample_{i+1}.exe"
        
        print("-" * 50)
        print(f"Processing {file_name} (Critical Sample {i+1} of 5)")
        time.sleep(0.1) 

        raw_feature_data = parse_pe_file_for_deployment(file_name, entropy_val)

        # 1. Validation Check (Phase 17)
        is_valid, error_msg = validate_features(raw_feature_data)
        
        if not is_valid:
            # We skip prediction but STILL log the value to the monitor, 
            # because data drift monitoring must track ALL incoming data, even invalid data.
            print(f"\n[!!!] VALIDATION FAILED. Reason: {error_msg}")
            monitor.log_value(entropy_val)
            print("Prediction skipped.")
            continue
        
        # 2. Prediction (Phase 16)
        try:
            input_vector = [raw_feature_data[name] for name in feature_schema]
            X_new = np.array(input_vector).reshape(1, -1)
            
            prediction = model.predict(X_new)[0]
            confidence = np.max(model.predict_proba(X_new)[0])
            
            print(f"[*] Prediction: {prediction} (Conf: {confidence:.2f})")
            
        except KeyError:
            print("[!!!] DEPLOYMENT FAILURE: Feature order/key error.")
            continue
            
        # 3. Monitoring (Phase 18)
        monitor.log_value(entropy_val)

    print("-" * 50)
    print("SIMULATION COMPLETE. STATIC ANALYSIS PIPELINE (PHASE 2) COMPLETE.")
    print("=" * 70)


if __name__ == "__main__":
    run_deployment_service()
