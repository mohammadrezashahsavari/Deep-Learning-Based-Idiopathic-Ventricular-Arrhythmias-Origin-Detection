# Project Overview

This project implements a fully automatic, end-to-end deep learning model to localize the origin of IVAs from 12-lead ECG data. The model is designed to differentiate between the two main locations where these abnormal signals originate: the right ventricular outflow tract (RVOT) and the left ventricular outflow tract (LVOT). The framework uses a combination of a Convolutional Neural Network (CNN) for spatial feature extraction, a recurrent neural network (RNN) for temporal analysis, and an attention mechanism to focus on the most critical parts of the ECG signal. Validated on a real-world dataset of 334 patients, the model provides a cost-effective and low-risk approach to aid in the evaluation of IVA patients before they undergo cardiac catheter ablation.

---

## Directory Structure

The repository is organized as follows:

* **`Dataset Generators/`**: Scripts for preparing the dataset, including data conversion, padding, and augmentation.
* **`experiments/`**: Contains the main Python scripts for running the supervised and semi-supervised learning experiments.
* **`Filters/`**: Contains filter files used in the MATLAB data preprocessing script.
* **`Models/`**: Defines the neural network architectures used in the experiments.
* **`output/`**: Stores the results of the experiments, including performance metrics and model predictions.
* **`tools/`**: A collection of utility scripts for tasks such as implementing callbacks, plotting training history, defining custom loss functions, and calculating performance metrics.
* **`Experiment.py`**: The main script to run the experiments.
* **`utils.py`**: A collection of helper functions for data loading, preprocessing, and other common tasks.

---

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3.x**
* **MATLAB** (for running the `DownsampleDataset.m` script)
* **TensorFlow**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **SciPy**
* **Matplotlib**
* **Librosa**
* **tqdm**

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/mohammadrezashahsavari/Deep-Learning-Based-Idiopathic-Ventricular-Arrhythmias-Origin-Detection.git](https://github.com/mohammadrezashahsavari/Deep-Learning-Based-Idiopathic-Ventricular-Arrhythmias-Origin-Detection.git)
    cd Deep-Learning-Based-Idiopathic-Ventricular-Arrhythmias-Origin-Detection
    ```

2.  **Install Python dependencies:**
    It is recommended to use a virtual environment to manage the project's dependencies.

    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

### 1. Data Preparation

The first step is to prepare the ECG dataset.

1.  **Convert to `.mat` format:**
    Place your raw ECG data in a directory (e.g., `Data/PVCVTECGData`). Then, run the `MatDatasetGenerator.py` script to convert the data into `.mat` files.

    ```bash
    python "Dataset Generators/MatDatasetGenerator.py"
    ```

2.  **Pad the signals:**
    Use `DatasetPadder.py` to pad the signals to a uniform length. This script provides options for zero-padding and self-padding.

    ```bash
    python "Dataset Generators/DatasetPadder.py"
    ```

3.  **Downsample and filter the signals:**
    Run the `DownsampleDataset.m` script in MATLAB to filter and downsample the ECG signals. Make sure the paths in the script are set correctly. This will create a new directory with the preprocessed data (e.g., `Data/ZeroPaddedFiltered50HzDataset`).

### 2. Running the Experiments

The main experiments can be run using the `Experiment.py` script. You can configure the experiment parameters within this file, such as the model architecture, and whether to use data augmentation.

To run the 10-fold cross-validation for the VGG-BiLSTM model:

1.  **Configure the experiment:**
    Open `Experiment.py` and ensure the `dataset_path` and `network_structure` variables are set as desired. For example:

    ```python
    dataset_path = os.path.join(base_project_dir, 'Data', 'ZeroPaddedFiltered50HzDataset')
    model = Experiment(dataset_path, network_structure='VGG_BiLSTM', seed=300)
    ```

2.  **Run the script:**

    ```bash
    python Experiment.py
    ```

The training progress and results will be printed to the console, and the detailed output will be saved in the `output/` directory.

### 3. Reproducing Results

To reproduce the results from the trained models, you can use the `reproduce_results_on_10fold()` method in `supervised.py`. This will load the saved model weights and evaluate them on the test set.

---

## Models

The `Models/` directory contains the definitions for the different neural network architectures used in this project:

* **`models.py`**: Defines various models, including:
    * `VGG_Model`
    * `VGG_LSTM_Model`
    * `VGG_BiLSTM_Model`
    * `VGG_LSTM_Attn_Model`
    * `VGG_BiLSTM_Attn_Model`
* **`SemisupervisedModels.py`**: Contains models for semi-supervised learning experiments (if applicable).

---

## Citing this Work

If you use this code or the findings from our paper in your research, please consider citing our publication:

```bibtex
@article{shahsavari2024localizing,
  title={Localizing the origin of idiopathic ventricular arrhythmias from ecg using a recurrent convolutional neural network with attention},
  author={Shahsavari, Mohammadreza and Delfan, Niloufar and Forouzanfar, Mohamad},
  journal={IEEE Transactions on Instrumentation and Measurement},
  volume={73},
  pages={1--10},
  year={2024},
  publisher={IEEE}
}
