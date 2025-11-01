import pandas as pd
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import OrderedDict

# --- Configuration ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Define file paths
STATIC_DATA_FILE = 'malware_data_final.csv'
DYNAMIC_DATA_OUTPUT_FILE = 'combined_feature_data_v1.csv'
MIN_CLASS_THRESHOLD = 2 # Lowered to 2 to maximize the sample pool.

# --- Core Dynamic Feature Generation ---

def generate_dynamic_features(df_static):
    """
    Simulates the extraction of behavioral features from a sandbox environment.
    """
    print("[*] Phase 3A: Simulating Dynamic Sandbox Execution and Feature Extraction...")
    
    num_samples = len(df_static)
    
    # 1. API Call Density (High-Value Feature)
    api_density = np.random.uniform(low=0.1, high=0.9, size=num_samples)
    entropy_score = df_static['Entropy_Section_Score'].values
    # Introduce correlation: high static entropy often correlates with high dynamic activity
    api_density = api_density + (entropy_score * 0.05)
    
    # 2. Registry Key Modifications (Medium-Value Feature)
    registry_mods = np.random.randint(low=0, high=15, size=num_samples)
    
    # 3. File Write Count (Low-Value Feature)
    file_writes = np.random.randint(low=0, high=8, size=num_samples)
    
    # 4. Network Activity Score (Simulated Network Traffic)
    network_score = np.random.uniform(low=0.0, high=1.0, size=num_samples)
    
    # Create a DataFrame for dynamic features
    df_dynamic = pd.DataFrame({
        'API_Call_Density': api_density,
        'Registry_Key_Modifications': registry_mods,
        'File_Write_Count': file_writes,
        'Network_Activity_Score': network_score,
    }, index=df_static.index)
    
    print(f"[+] Generated {len(df_dynamic.columns)} dynamic features.")
    return df_dynamic

def run_pipeline():
    print("=" * 70)
    print(" DYNAMIC ANALYSIS PIPELINE (PHASE 3) ")
    print("=" * 70)

    # Step 1: Load the initial (simulated) static data
    try:
        # Use sep=',' parameter to handle the quoted strings better during load
        df = pd.read_csv(STATIC_DATA_FILE, sep=',')
        print(f"[*] Loaded data from: {STATIC_DATA_FILE}")
        
        # --- CRITICAL FIX 1: Find the Label column (string column) and rename it ---
        label_candidate_cols = [col for col in df.columns if df[col].dtype == 'object' or df[col].nunique() > len(df) * 0.9]
        
        if label_candidate_cols:
            label_col_name = label_candidate_cols[-1]
            df.rename(columns={label_col_name: 'Label'}, inplace=True)
            print(f"[!] Identified and renamed column '{label_col_name}' to 'Label' for consistency.")
        else:
            print("[FATAL] Could not identify a suitable Label column. Aborting.")
            return

        # --- CRITICAL FIX 2: Rename numeric features starting at index 2 (skipping 0:ID and 1:Hash) ---
        expected_raw_names = [f'raw_feature_{i}' for i in range(5)]
        rename_dict = {}
        START_INDEX = 2 # Skip ID/Date (0) and Hash/ID (1). Numeric features start here.
        
        for i in range(5):
            current_col_index = START_INDEX + i
            current_col_name = df.columns[current_col_index]
            new_raw_name = expected_raw_names[i]
            
            if current_col_name not in expected_raw_names:
                rename_dict[current_col_name] = new_raw_name
        
        if rename_dict:
            df.rename(columns=rename_dict, inplace=True)
            print("[!] Renamed first 5 NUMERIC feature columns (starting at index 2) to raw_feature_0...4 for regeneration.")


        # Apply the class filtering logic
        if 'Label' not in df.columns:
             print("[FATAL] 'Label' column lost during initial renaming/cleaning. Cannot proceed.")
             return
             
        df_filtered = df.groupby('Label').filter(lambda x: len(x) >= MIN_CLASS_THRESHOLD)
        print(f"    - Samples after filtering: {len(df_filtered)}")
    
    except FileNotFoundError:
        print(f"[FATAL] Static data file '{STATIC_DATA_FILE}' not found. Please ensure Phase 1 was run successfully.")
        return
    except IndexError:
        print("[FATAL] Data structure error: The input CSV does not contain the expected number of columns (at least 7).")
        return
    
    # --- CRITICAL FIX 3: REGENERATE STATIC FEATURES & BULLETPROOF TYPE CONVERSION & IMPUTATION ---
    static_features_needed = ['TimeDateStamp', 'TextSection_Entropy', 'Entropy_Section_Score']
    
    if any(col not in df_filtered.columns for col in static_features_needed):
        print("[*] Regenerating missing high-value static features from raw features...")
        
        if 'raw_feature_0' not in df_filtered.columns or 'raw_feature_1' not in df_filtered.columns:
            print("[FATAL] Cannot regenerate features. Required raw columns are missing.")
            return

        df_filtered = df_filtered.copy()

        # Aggressive cleaning function using pd.to_numeric(errors='coerce')
        def clean_and_convert_numeric(series):
            # 1. Convert to string, strip whitespace, remove quotes
            cleaned_series = series.astype(str).str.strip().str.replace('"', '', regex=False)
            # 2. Force conversion to numeric, turning any remaining trash into NaN
            return pd.to_numeric(cleaned_series, errors='coerce')

        # 1. Regenerate Simple Static Features and apply Median Imputation
        df_filtered['TimeDateStamp'] = clean_and_convert_numeric(df_filtered['raw_feature_0'])
        df_filtered['TextSection_Entropy'] = clean_and_convert_numeric(df_filtered['raw_feature_1'])
        
        # --- MEDIAN IMPUTATION CORE LOGIC ---
        initial_nan_count = df_filtered[['TimeDateStamp', 'TextSection_Entropy']].isna().sum().sum()
        
        for col in ['TimeDateStamp', 'TextSection_Entropy']:
            # Calculate median of the column (skipping NaNs)
            median_value = df_filtered[col].median()
            # If the entire column is NaN (as happened when 100% was dropped), use 0 as a default
            if pd.isna(median_value):
                median_value = 0.0
            
            # Fill the NaNs with the calculated median
            df_filtered[col].fillna(median_value, inplace=True)

        if initial_nan_count > 0:
            print(f"[IMPUTED] Replaced {initial_nan_count} corrupted value(s) with feature median.")

        # 2. Regenerate the Breakthrough Feature
        df_filtered['Entropy_Section_Score'] = (df_filtered['TextSection_Entropy'] * 2) + (df_filtered['TimeDateStamp'] * 1e-10)
        print(f"[+] Static features successfully regenerated and imputed. Samples ready: {len(df_filtered)}")
    
    # Step 2: Generate Dynamic Features
    df_dynamic = generate_dynamic_features(df_filtered)
    
    # Step 3: Combine Static and Dynamic Features
    print("[*] Phase 3B: Merging Static and Dynamic Feature Sets...")
    
    # Define which features to keep for the final combined dataset
    static_features_to_keep = static_features_needed + ['Label']
    
    # Filter the DataFrame to only include the required static features and the label
    df_static_ready = df_filtered[static_features_to_keep]

    # Merge based on index (assuming 1-to-1 mapping of sample IDs)
    df_combined = pd.merge(df_static_ready, df_dynamic, left_index=True, right_index=True)
    
    # Verify the combined dataset
    print(f"[+] Combined dataset shape: {df_combined.shape}")
    print(f"[+] Total features in V1: {len(df_combined.columns) - 1}")
    
    # Step 4: Save the Combined Dataset
    print(f"[*] Phase 3C: Saving combined dataset to {DYNAMIC_DATA_OUTPUT_FILE}")
    df_combined.to_csv(DYNAMIC_DATA_OUTPUT_FILE, index=False)
    print("[+] Dynamic Analysis Feature Generation Complete.")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
