# Installation and Usage

## Requirements

The package requirements for this repository follow the dependencies of the Aeon time-series toolkit and associated deep learning libraries. Please install the required packages before running the scripts.

Core dependencies include:

* Python 3.10+
* NumPy
* SciPy
* Scikit-learn
* PyTorch
* TensorFlow / Keras
* Matplotlib
* Seaborn
* tslearn
* Aeon toolkit

Install dependencies using:

```bash
pip install -r requirements.txt
```

---
# Requirements 
numpy>=1.23
scipy>=1.10
scikit-learn>=1.3

torch>=2.0
tensorflow>=2.12

matplotlib>=3.7
seaborn>=0.12

tslearn>=0.6

aeon>=0.10

pandas>=2.0
joblib>=1.3
tqdm>=4.65
h5py>=3.9

## Dataset Preparation

### ABIDE Dataset

The Autism Brain Imaging Data Exchange (ABIDE) dataset can be downloaded from:

https://fcon_1000.projects.nitrc.org/indi/abide/

For preprocessing, rs-fMRI data should be processed using the **C-PAC pipeline**, ensuring that the **CC200 parcellation (Craddock CC-200 atlas)** is enabled during preprocessing. The repository uses ROI-level time series generated from the CC200 atlas.

The corresponding ROI time-series data and behavioral/IQ information are available through the ABIDE release and phenotypic files.

---

### HCP Dataset

The Human Connectome Project (HCP) dataset can be downloaded from:

https://www.humanconnectome.org/study/hcp-young-adult

After downloading the HCP dataset, the rs-fMRI data must be mapped into **200 Regions of Interest (ROIs)** using the provided script:

```bash
python hcp_200_rois_mapping.py
```

This script converts the HCP grayordinate-level rs-fMRI data into **CC200 ROI-based time series** and saves the processed output in the local directory for downstream prediction.

---

## Running the Models

### 1. Similarity-based IQ Prediction Model

Run:

```bash
python Similarity_based_prediction.py
```

This script:

* Computes similarity measures (Correlation, Coherence, PLI, PLV, DTW, and LCSS)
* Trains the CNN-based similarity prediction model
* Performs 5-fold cross-validation
* Saves prediction plots and evaluation statistics
* Extracts and saves CNN similarity features required for the Hybrid model

Saved CNN features are later used in the Hybrid model.

---

### 2. Temporal (HYDRA-based) IQ Prediction Model

Run:

```bash
python Temporal_based.py
```

This script:

* Extracts multivariate temporal features using HYDRA
* Applies Ridge regression for prediction
* Performs 5-fold cross-validation
* Computes statistical evaluation metrics and prediction plots

---

### 3. Hybrid IQ Prediction Model

Run:

```bash
python hybrid.py
```

This script:

* Combines HYDRA temporal features with CNN-derived similarity features
* Performs intermediate feature fusion
* Applies Ridge regression on the combined representation
* Performs 5-fold cross-validation
* Reports prediction performance, confidence intervals, permutation statistics, and scatter plots

The Hybrid model represents the final prediction framework and combines both temporal and similarity-based information for IQ prediction.

