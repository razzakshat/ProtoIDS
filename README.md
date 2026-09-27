# ProtoIDS: Prototype-Based Open-Set IoT Intrusion Detection

ProtoIDS is an experimental deep learning approach for IoT intrusion detection designed to tackle the critical problem of zero-day attacks. Traditional classifiers draw rigid decision boundaries between known attacks, leading them to confidently misclassify unseen or zero-day attacks as "normal" traffic or known attack types.

ProtoIDS solves this by learning "prototypes" (representative embeddings) for known network behaviors. When new traffic is analyzed, the model measures its cosine distance to these prototypes. If the traffic is too far from any known prototype, it is reliably flagged as **UNKNOWN** (a potential zero-day attack), rather than being misclassified.

## Key Features

- **Leakage-Free Preprocessing**: Comprehensive forensic analysis ensures that identifiers like IP addresses, timestamps, and raw payloads are removed, guaranteeing the model learns true behavioral patterns rather than memorizing dataset artifacts.
- **Memory-Efficient Processing**: Implements chunked data processing and reservoir sampling to handle massive datasets like CICIoT2023 (5.5M+ rows) gracefully without exceeding RAM limits.
- **Rigorous Open-Set Protocol**: Explicitly withholds specific attack classes (e.g., `MITM-ArpSpoofing`, `VulnerabilityScan`, `DictionaryBruteForce`) during training to rigorously evaluate the model's ability to detect novel, zero-day attacks at inference time.
- **Cross-Dataset Validation**: Built to evaluate model generalization across completely different environments (CICIoT2023 and Edge-IIoTset).

## Project Structure

- `src/preprocessing/`: Memory-efficient chunked data processing, feature scaling, and open-set protocol enforcement.
- `src/baselines/`: Scripts to run traditional ML baselines (Random Forest, Gradient Boosting, MLP) and ablation studies.
- `src/protoids/`: Core implementation of the ProtoIDS architecture (Encoder, Prototype Layer, Distance thresholds).
- `experiments/data/`: Processed development subsets (balanced via reservoir sampling) for rapid prototyping.
- `reports/`: Comprehensive dataset forensic reports documenting feature statistics, class imbalances, and leakage analysis.

## Setup Instructions

### 1. Environment Setup

It is recommended to use a virtual environment. Install dependencies with:

```bash
pip install -r requirements.txt
```

### 2. Datasets
The project uses two massive datasets which are **NOT** tracked in version control due to their size (multi-GB). You will need to download them and place them in the root directory:
- [CICIoT2023](https://www.kaggle.com/datasets/himadri07/ciciot2023) -> `CICIOT23/`
- Edge-IIoTset -> `DNN-EdgeIIoT-dataset.csv/`

*(Note: Processed, highly-balanced development subsets are generated locally for rapid prototyping.)*

### 3. Usage

**1. Run Preprocessing:**
Process the raw datasets into leakage-free, normalized, and balanced development subsets. This step also enforces the open-set protocol.
```bash
python src/preprocessing/ciciot_preprocessing.py
```

**2. Run Baselines & Ablation Studies:**
Evaluate traditional ML models and feature ablation studies on the known classes.
```bash
python src/baselines/ablation_quick.py
python src/baselines/run_baselines.py
```

**3. Train ProtoIDS:**
Train the ProtoIDS neural network.
```bash
python src/protoids/train_protoids.py
```

## Recent Milestones

- ✅ Completed rigorous dataset forensic analysis to identify and eliminate data leakage.
- ✅ Implemented memory-efficient (chunked) preprocessing for CICIoT2023.
- ✅ Integrated the rigorous open-set evaluation protocol (withholding zero-day attacks from the training/validation subsets).
- ✅ Fixed baseline ablation scripts to seamlessly integrate with the open-set mapping logic.
- ✅ Fully implemented the ProtoIDS deep learning architecture (Encoder and Prototype layer).

## Acknowledgments
Developed for research into robust, open-set Intrusion Detection Systems (IDS) for IoT networks. 
