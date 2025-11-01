import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# THIS IS THE CRITICAL IMPORT!
from imblearn.over_sampling import RandomOverSampler
import random

# --- Configuration ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def generate_simulated_data(n_samples=1462, n_features=6):
    """
    Simulates the original malware dataset with severe class imbalance
    to match the output characteristics (38 classes, many small).
    (NOTE: In a real scenario, you would replace this with your actual CSV loading.)
    """
    print(f"[*] Phase 1: Loading data from: malware_data_final.csv (Simulated)")

    # 1. Simulate class labels (38 classes, heavily imbalanced)
    base_class = ["boatnet.ppc"] * 100
    common_class = ["boatnet.x86"] * 80
    mid_class = ["debug"] * 50
    rare_classes = [
        "aarch64", "arc", "arm", "arm5", "arm7", "dvr.jaws.sh", "file", "mips",
        "morte.i686", "morte.ppc", "sora.ppc", "space.x86", "x86_64", "Class_14", "Class_15", 
        "Class_16", "Class_17", "Class_18", "Class_19", "Class_20", "Class_21", "Class_22",
        "Class_23", "Class_24", "Class_25", "Class_26", "Class_27", "Class_28", "Class_29",
        "Class_30", "Class_31", "Class_32", "Class_33", "Class_34", "Class_35", "Class_36",
        "Class_37"
    ]
    # Ensure 38 classes total are generated in the raw data
    all_unique_labels = rare_classes[:38] 
    
    # Create an imbalanced list of 1462 labels, prioritizing the small classes to hit 38 unique families
    y_raw_list = base_class + common_class + mid_class
    
    # Add a single instance for the remaining rare classes
    remaining_classes = [c for c in all_unique_labels if c not in base_class and c not in common_class and c not in mid_class]
    y_raw_list.extend(remaining_classes)
    
    # Fill the rest with a random distribution
    num_needed = n_samples - len(y_raw_list)
    y_raw_list.extend(np.random.choice(all_unique_labels, size=num_needed, replace=True).tolist())
    
    y_labels = y_raw_list[:n_samples]
    
    # 2. Simulate feature matrix (14 columns)
    X_raw = np.random.rand(n_samples, n_features)
    
    df = pd.DataFrame(X_raw, columns=[f'raw_feature_{i}' for i in range(n_features)])
    df['Label'] = y_labels
    
    print(f"[+] Data loaded. Initial shape: {df.shape}")
    
    # Simulate filtering to 663 samples, ensuring 38 unique classes are retained
    df_filtered = df.groupby('Label').filter(lambda x: len(x) >= 1).sample(n=663, replace=True, random_state=RANDOM_SEED)
    
    print(f"[*] Filtered data: Reduced from 1462 to {df_filtered.shape[0]} samples.")
    print(f"[*] Retained {df_filtered['Label'].nunique()} classes.")
    
    return df_filtered

def generate_deep_static_features(df):
    """
    Simulates the generation of 8 rich numerical features.
    """
    print(f"[*] Phase 2: Simulating deep static PE features...")
    
    num_new_features = 8
    
    # Generate 8 new features
    for i in range(num_new_features):
        df[f'rich_feature_{i}'] = np.log1p(df[df.columns[i % 6]] * (i + 1))
        
    print(f"[+] {num_new_features} rich numerical features generated.")
    return df

def run_pipeline():
    print("=" * 60)
    print(" DEEP STATIC FEATURE PIPELINE (PHASE 2 - IMBALANCE FIXED) ")
    print("=" * 60)

    # 1. Load and Filter Data
    data_df = generate_simulated_data()

    # 2. Feature Generation
    data_df = generate_deep_static_features(data_df)

    # 3. Prepare Features and Labels
    X = data_df.drop('Label', axis=1)
    y = data_df['Label']

    # 4. Data Split
    # Stratified split is crucial for imbalanced data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    print(f"[+] Final Feature Matrix X shape: {X.shape}")
    print(f"[+] Data split successful: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    
    # ==========================================================
    # --- CRITICAL FIX: ADDRESSING CLASS IMBALANCE (PHASE 3) ---
    # ==========================================================
    print("\n[*] Phase 3: Applying Random Over-Sampler to Training Data...")
    
    # Initialize the RandomOverSampler
    ros = RandomOverSampler(random_state=RANDOM_SEED)
    
    # Resample the training data
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
    
    print(f"[+] Training data before resampling: {X_train.shape[0]} samples.")
    print(f"[+] Training data AFTER resampling: {X_train_res.shape[0]} samples.")
    print(f"[+] Resampling complete. All {y.nunique()} classes are now equally represented in the training set.")
    # ==========================================================

    # 5. Training Model
    print("\n[*] Phase 5: Training Random Forest Classifier on RESAMPLED data...")

    # Using class_weight='balanced' in addition to oversampling can sometimes help,
    # but oversampling is the primary fix here.
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=RANDOM_SEED, 
        class_weight='balanced'
    )
    
    # Train on the resampled data
    model.fit(X_train_res, y_train_res)

    # 6. Evaluation
    # Evaluate against the original, UNTOUCHED test set (X_test, y_test)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    # Set zero_division=0 to handle classes not present in the test set (though stratification helps)
    report = classification_report(y_test, y_pred, zero_division=0)
    
    print("\n--- CLASSIFICATION REPORT (Deep Static Features - IMBALANCE FIXED) ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    print("=" * 60)

if __name__ == "__main__":
    # Check for imblearn availability (This check is no longer strictly necessary 
    # since you fixed the import error, but is good practice)
    try:
        from imblearn.over_sampling import RandomOverSampler
    except ImportError:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("ERROR: 'imbalanced-learn' library not found.")
        print("Please install it using: pip install imbalanced-learn")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        exit()
        
    run_pipeline()
