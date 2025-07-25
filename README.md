
# 🎧 Audio MNIST Classification using MFCC and Neural Networks

This project demonstrates end-to-end classification of spoken digit audio samples from the [Audio MNIST dataset](https://www.kaggle.com/datasets/sripaadsrinivasan/audio-mnist). It involves feature extraction using MFCCs, data preprocessing, and training neural networks using three different frameworks: **Scikit-learn**, **TensorFlow Keras**, and **PyTorch**.

---

## 📂 Project Structure

- `mnist.py`: Main script for feature extraction, preprocessing, and classification using MLPs.
- `mnist_features.csv`: Generated CSV file containing extracted MFCC features and labels (created during runtime).
- `data/`: Directory containing the Audio MNIST `.wav` files (expected structure: `data/<speaker_id>/<digit>_<speaker_id>_<index>.wav`).

---

## 📊 Dataset

The dataset contains spoken digits (0–9) in `.wav` format sampled at 16kHz. Each file is labeled according to the digit spoken.

📥 Download the dataset from Kaggle: [Audio MNIST](https://www.kaggle.com/datasets/sripaadsrinivasan/audio-mnist)

---

## 📈 Pipeline Overview

### 1. Feature Extraction
- MFCC (13 coefficients) are extracted using `python_speech_features` and `librosa`.
- Each audio file is converted into a single 13-dimensional vector using the mean of its MFCCs.
- Features and corresponding labels are stored in a CSV file.

### 2. Data Splitting
- Train-test split: 80% training, 20% testing.
- Within training data, a further 90-10 split is used for validation during training.

---

## 🧠 Model Architectures

### ✅ Scikit-learn MLP
- Framework: `sklearn.neural_network.MLPClassifier`
- Architecture: [256, 128, 64, 32] fully connected layers
- Reports: Accuracy, Precision, Recall, F1 Score, Confusion Matrix

### ✅ TensorFlow Keras MLP
- Framework: `tensorflow.keras.Sequential`
- Architecture: [128, 64, 32] + softmax output
- Activation: SiLU (Swish)
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Reports: Accuracy, Precision, Recall, F1 Score, Confusion Matrix

### ✅ PyTorch MLP
- Framework: `torch.nn`
- Architecture: [128, 64] + linear output
- Activation: SiLU
- Training: Manual loop with `DataLoader` and GPU support
- Reports: Per-epoch training/validation metrics + full test evaluation

---

## 🧪 Evaluation Metrics

Each model reports the following on the **test set**:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

🎯 **Target Accuracy**: ≥ 87%

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/audio-mnist-mlp.git
cd audio-mnist-mlp
```

### 2. Install Dependencies
```bash
pip install numpy pandas librosa python_speech_features scikit-learn tensorflow torch tqdm
```

### 3. Download Dataset
Place the downloaded `Audio MNIST` `.wav` files in a folder named `data/`.

### 4. Run the Script
```bash
python mnist.py
```

---

## 🛠️ Requirements

- Python 3.7+
- `numpy`, `pandas`
- `librosa`, `python_speech_features`
- `scikit-learn`, `tensorflow`, `torch`
- `tqdm`, `matplotlib` (optional for visualization)

---

## 📌 Notes

- MFCC extraction assumes uniform audio format (16kHz sampling).
- Normalize input features before training.
- GPU acceleration is automatically used in PyTorch if available.
- Outputs include detailed logs and evaluation summaries for each framework.

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 👨‍💻 Author

**Waqar Ahmed**  
For inquiries or collaborations, feel free to reach out via GitHub or LinkedIn.
