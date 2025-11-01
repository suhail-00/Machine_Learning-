import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
import os
from collections import Counter

# --- Configuration ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

DATA_FILE = 'combined_feature_data_v1.csv'

def run_pipeline():
    print("=" * 70)
    print(" PHASE 4: ENSEMBLE MODEL TRAINING AND EVALUATION ")
    print("=" * 70)

    # Step 1: Load the combined dataset
    try:
        df = pd.read_csv(DATA_FILE)
        print(f"[*] Loaded combined data from: {DATA_FILE}")
    except FileNotFoundError:
        print(f"[FATAL] Data file '{DATA_FILE}' not found. Please ensure Phase 3 was run successfully.")
        return

    # Check for sufficient samples after all the cleaning
    if len(df) < 10:
        print(f"[FATAL] Insufficient clean samples ({len(df)} found). Cannot train model reliably. Aborting.")
        return

    # Separate features (X) and target (y)
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    # Standardize/Normalize the features (recommended, but skipped for simplicity on simulated data)
    print(f"[*] Total Features for training: {X.shape[1]} (Static + Dynamic)")
    
    # Step 2: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )
    print(f"[*] Split data: Train samples={len(X_train)}, Test samples={len(X_test)}")
    
    # Step 3: Handle Class Imbalance using SMOTE
    print(f"[*] Applying SMOTE to balance training classes (Initial: {Counter(y_train)})")
    
    # If using a highly reduced dataset (like 16 samples), SMOTE needs few neighbors
    try:
        # Check if the smallest class has enough samples for SMOTE (min_samples=k_neighbors)
        min_class_size = min(Counter(y_train).values())
        k_neighbors = min(5, max(1, min_class_size - 1)) # Use k=1 if min_size is 2
        
        if k_neighbors < 1:
            print("[WARNING] Not enough unique samples for SMOTE in the smallest class. Skipping SMOTE.")
            X_resampled, y_resampled = X_train, y_train
        else:
            smote = SMOTE(sampling_strategy='auto', random_state=RANDOM_SEED, k_neighbors=k_neighbors)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            print(f"[+] SMOTE applied. Training samples increased to: {len(X_resampled)}")
            print(f"    - Resampled Class Counts: {Counter(y_resampled)}")

    except Exception as e:
        print(f"[WARNING] SMOTE failed due to data sparsity ({e}). Proceeding without oversampling.")
        X_resampled, y_resampled = X_train, y_train


    # Step 4: Define Base Classifiers
    print("[*] Training Base Classifiers for the Ensemble...")
    
    # Model 1: Random Forest (Excellent for non-linear data and speed)
    clf1 = RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED, class_weight='balanced_subsample')
    
    # Model 2: Gradient Boosting (Excellent for complex relationships and high prediction power)
    clf2 = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=RANDOM_SEED)

    # Step 5: Create the Ensemble (Voting Classifier)
    ensemble = VotingClassifier(
        estimators=[('rf', clf1), ('gb', clf2)],
        voting='soft'  # 'soft' uses weighted probabilities for better results
    )

    # Step 6: Train and Predict
    print("[*] Training Ensemble Model (Random Forest + Gradient Boosting)...")
    ensemble.fit(X_resampled, y_resampled)
    
    y_pred = ensemble.predict(X_test)
    
    # Step 7: Evaluate
    final_accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "=" * 50)
    print("         FINAL ENSEMBLE MODEL RESULTS")
    print("=" * 50)
    print(f"Accuracy Score: {final_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("=" * 50)
    print("[+] Ensemble Training and Evaluation Complete.")


if __name__ == "__main__":
    run_pipeline()
