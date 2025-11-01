import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# --- CONFIGURATION ---
INPUT_FILE = "malware_data_final.csv"
TARGET_COLUMN = 'malware_family'
RANDOM_STATE = 42
MIN_SAMPLES_PER_CLASS = 5 # Threshold to resolve train_test_split ValueError

# !!! ACTION REQUIRED: Ensure these column names EXACTLY match the cleaned headers !!!
FEATURES_TO_KEEP = [
    'apk',
    'application/zip',
    'mohit',
]

def run_static_classification():
    print("="* 60)
    print(" STATIC MALWARE CLASSIFICATION PIPELINE (FINAL) ")
    print("="* 60)

    #------1. Data Loading and Initial Cleaning------
    try:
        print(f"[*] Phase 1: Loading data from: {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE)
        print(f"[+] Data loaded. Initial shape: {df.shape}")
    except FileNotFoundError:
        print(f"[ERROR] Input file '{INPUT_FILE}' not found. Please run the data collection script first.")
        sys.exit(1)

    # Select relevant columns (Target + Features)
    current_cols = df.columns.tolist()
    cols_to_use = [col for col in FEATURES_TO_KEEP if col in current_cols]
    
    if TARGET_COLUMN not in cols_to_use:
        cols_to_use.append(TARGET_COLUMN)
        
    df_processed = df[cols_to_use].copy()
    print(f"[+] Features selected for initial model: {df_processed.columns.tolist()}")

    # Drop rows where the target label is missing
    df_processed.dropna(subset=[TARGET_COLUMN], inplace=True)
    print(f"[+] Shape after dropping missing labels: {df_processed.shape}")

    #------2. Feature Encoding & Filtering------

    # A. Encode Target Variable (Y)
    le = LabelEncoder()
    df_processed['target_encoded'] = le.fit_transform(df_processed[TARGET_COLUMN])
    print(f"[+] Target '{TARGET_COLUMN}' encoded into {len(le.classes_)} classes.")
    
    # B. Filter Rare Classes (To resolve the ValueError)
    class_counts = df_processed['target_encoded'].value_counts()
    frequent_classes = class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index
    
    if len(frequent_classes) < 2:
        print(f"\n[FATAL ERROR] Not enough classes remain after filtering (Min required: 2, Found: {len(frequent_classes)}). Cannot train model.")
        return

    # Filter the DataFrame based on frequent encoded IDs
    df_filtered = df_processed[df_processed['target_encoded'].isin(frequent_classes)].copy()
    
    print(f"[*] Filtered data: Reduced from {len(df_processed)} to {len(df_filtered)} samples.")
    print(f"[*] Retained {len(frequent_classes)} classes.")

    # C. Define X and Y strictly from the filtered set
    Y = df_filtered['target_encoded']
    X_raw = df_filtered.drop(columns=[TARGET_COLUMN, 'target_encoded'])
    
    # D. Prepare Features (X) - One-Hot Encode categorical features
    X_encoded = pd.get_dummies(X_raw, drop_first=True)
    
    # E. Handle case where no features remain after encoding
    if X_encoded.shape[1] == 0:
        print("[WARNING] No features remain after encoding/selection. Adding a dummy feature to proceed.")
        X_features = pd.DataFrame({'dummy_feature': 1}, index=X_encoded.index)
    else:
        X_features = X_encoded

    #------3. Split Data------
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_features, Y, test_size=0.2, random_state=RANDOM_STATE, stratify=Y
    )
    print(f"[+] Data split successful: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

    #------4. Model Training------
    print("\n[*] Phase 4: Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, Y_train)

    #------5. Evaluation------
    Y_pred = model.predict(X_test)

    print("\n--- CLASSIFICATION REPORT ---")
    print(f"Accuracy: {accuracy_score(Y_test, Y_pred):.4f}")

    # Get class names from the encoder fitted on the *original* data for reporting consistency
    unique_labels_in_test = np.unique(Y_test)
    report_labels = le.inverse_transform(unique_labels_in_test)
    print("\nClassification Report:\n", classification_report(Y_test, Y_pred, target_names=report_labels))

    # Plot Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(Y_test, Y_pred)
    # Use report_labels for class names in confusion matrix
    annotate = len(report_labels) <= 10 
    sns.heatmap(cm, annot=annotate, fmt='d', cmap='Blues', xticklabels=report_labels, yticklabels=report_labels)
    plt.title('Confusion Matrix (Static Features)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

    print("=" * 60)

if __name__ == "__main__":
    run_static_classification()