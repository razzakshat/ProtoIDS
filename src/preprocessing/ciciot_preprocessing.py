"""
Preprocessing pipeline for CICIoT2023 dataset.
Implemented memory-efficient chunked processing and open-set protocol.
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler
import joblib
from collections import defaultdict

# Open-Set Protocol: these classes are withheld during training/validation
# and only evaluated at test time as 'UNKNOWN'.
UNKNOWN_CLASSES = ['MITM-ArpSpoofing', 'VulnerabilityScan', 'DictionaryBruteForce']

def get_columns_to_remove():
    """
    Return columns to remove and reasons based on forensic analysis.
    """
    target_col = ['label']
    constant_cols = ['Telnet', 'IRC']
    near_constant_cols = ['SMTP']
    redundant_pair1 = ['Srate']
    redundant_pair2 = ['LLC']

    cols_to_remove = target_col + constant_cols + near_constant_cols + redundant_pair1 + redundant_pair2
    reasons = {
        'label': 'Target variable',
        'Telnet': 'Constant (always 0)',
        'IRC': 'Constant (always 0)',
        'SMTP': 'Near-constant (essentially constant in validation)',
        'Srate': 'Redundant with Rate (identical values)',
        'LLC': 'Redundant with IPv (identical distributions, 99.99% = 1.0)'
    }

    return cols_to_remove, reasons

def create_label_mappings(known_classes):
    """
    Create mappings for original multiclass label and binary label.
    """
    label_classes = sorted(known_classes)
    label_to_int = {label: idx for idx, label in enumerate(label_classes)}
    # Add UNKNOWN class for open-set evaluation
    label_to_int['UNKNOWN'] = len(label_classes)
    
    int_to_label = {idx: label for label, idx in label_to_int.items()}

    label_mapping = {
        'multiclass': {
            'label_to_int': label_to_int,
            'int_to_label': int_to_label,
            'known_classes': label_classes,
            'unknown_classes': UNKNOWN_CLASSES
        },
        'binary': {
            'rule': 'BenignTraffic -> 0, else -> 1',
            'BenignTraffic': 0,
            'Attack': 1
        }
    }

    return label_mapping

def main():
    """
    Main preprocessing pipeline for CICIoT2023 with chunking and open-set protocol.
    """
    print("Starting CICIoT2023 chunked preprocessing...")

    data_base_path = 'CICIOT23'
    output_dir = 'results/baselines'
    processed_dir = 'experiments/data'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    train_path = os.path.join(data_base_path, 'train', 'train.csv')
    
    cols_to_remove, reasons = get_columns_to_remove()
    
    # Initialize incremental scaler
    scaler = StandardScaler()
    
    chunksize = 100000
    
    reservoirs = {}
    class_seen_counts = defaultdict(int)
    feature_cols = None
    
    if os.path.exists(train_path):
        print(f"Processing {train_path} in chunks of {chunksize}...")
        # We use reservoir sampling to maintain up to 10000 samples per class without loading all data
        for chunk_idx, chunk in enumerate(pd.read_csv(train_path, chunksize=chunksize)):
            print(f"  Processing chunk {chunk_idx+1}...")
            
            if feature_cols is None:
                feature_cols = [c for c in chunk.columns if c not in cols_to_remove]
                
            # Leakage-free open-set protocol: filter out UNKNOWN_CLASSES
            known_mask = ~chunk['label'].isin(UNKNOWN_CLASSES)
            known_chunk = chunk[known_mask]
            
            if len(known_chunk) == 0:
                continue
                
            X_chunk = known_chunk[feature_cols].values
            # Fit scaler ONLY on known classes
            scaler.partial_fit(X_chunk)
            
            # Update reservoir samples
            for _, row in known_chunk.iterrows():
                label = row['label']
                class_seen_counts[label] += 1
                idx = class_seen_counts[label]
                
                if label not in reservoirs:
                    reservoirs[label] = []
                    
                if len(reservoirs[label]) < 10000:
                    reservoirs[label].append(row)
                else:
                    # Random replacement for reservoir sampling
                    j = np.random.randint(0, idx)
                    if j < 10000:
                        reservoirs[label][j] = row
                        
        print("Finished processing train chunks.")

        # Create development subset
        dev_rows = []
        for label, res in reservoirs.items():
            total_seen = class_seen_counts[label]
            res_df = pd.DataFrame(res)
            
            if total_seen < 5000:
                dev_rows.append(res_df)
            elif total_seen < 10000:
                # Max 5000 samples for medium classes
                dev_rows.append(res_df.sample(n=min(5000, len(res_df)), replace=False, random_state=42))
            else:
                # We already capped at 10000 samples in the reservoir
                dev_rows.append(res_df)
                
        dev_df_raw = pd.concat(dev_rows, ignore_index=True)
        
        # Scale development subset features
        X_dev = scaler.transform(dev_df_raw[feature_cols].values).astype(np.float32)
        dev_df = pd.DataFrame(X_dev, columns=feature_cols)
        dev_df['label_multi'] = dev_df_raw['label'].values
        dev_df['label_binary'] = (dev_df_raw['label'] != 'BenignTraffic').astype(int).values
        
        # Save development subset
        dev_parquet_path = os.path.join(processed_dir, 'ciciot_dev.parquet')
        dev_df.to_parquet(dev_parquet_path, index=False)
        print(f"Saved development subset to {dev_parquet_path}")

        # Metadata
        dev_metadata = {
            'seed': 42,
            'source': 'CICIOT23 training data (chunked)',
            'open_set_protocol': 'UNKNOWN_CLASSES withheld from scaler and dev subset',
            'sampling_rules': {
                'minority_classes': '<5000 samples -> keep all',
                'medium_classes': '5000 <= count < 10000 -> max 5000 samples',
                'majority_classes': '>=10000 samples -> max 10000 samples'
            },
            'total_rows': int(dev_df.shape[0]),
            'class_counts': dev_df['label_multi'].value_counts().to_dict()
        }
        dev_metadata_path = os.path.join(processed_dir, 'ciciot_dev_metadata.json')
        with open(dev_metadata_path, 'w') as f:
            json.dump(dev_metadata, f, indent=2)

        # Label Mappings
        known_classes = sorted(list(reservoirs.keys()))
        label_mapping = create_label_mappings(known_classes)
        
        label_mapping_path = os.path.join(output_dir, 'ciciot_label_mapping.json')
        with open(label_mapping_path, 'w') as f:
            json.dump(label_mapping, f, indent=2)

        # Preprocessing artifacts
        scaler_path = os.path.join(output_dir, 'ciciot_preprocessor.joblib')
        joblib.dump(scaler, scaler_path)

        manifest = {
            'original_columns': list(feature_cols) + cols_to_remove,
            'removed_columns': cols_to_remove,
            'retained_columns': feature_cols,
            'removal_reasons': reasons
        }
        manifest_path = os.path.join(output_dir, 'ciciot_feature_manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print("Preprocessing artifacts saved successfully!")
    else:
        print(f"Dataset not found at {train_path}, script is ready but no processing performed.")

if __name__ == '__main__':
    main()