# 🎤 Speech Emotion Recognition (SER)

An end-to-end **Speech Emotion Recognition** system that identifies human emotions from speech using a deep learning **LSTM-based neural network**.  
This project covers **audio feature extraction, model training, evaluation, and deployment** through a Streamlit web application.

---

## 📌 Project Overview

The goal of this project is to classify emotions from speech signals. The system processes raw audio, extracts meaningful acoustic features, and feeds them into a trained LSTM model to predict the underlying emotion.

Key highlights:
- Robust audio feature extraction using **Librosa**
- Optimized **multi-layer LSTM** architecture
- Detailed evaluation with accuracy, loss curves, and confusion matrix
- Interactive **Streamlit app** for real-time inference

---

## 🎯 Emotion Classes

The model predicts the following **8 emotions**:

- Angry 😠  
- Calm 😌  
- Disgust 🤢  
- Fear 😨  
- Happy 😊  
- Neutral 😐  
- Sad 😢  
- Surprise 😲  

---

## 📊 Model Performance

### ✅ Test Results
- **Test Accuracy:** **83.26%**
- **Test Loss:** **0.5243**
- **Total Test Samples:** 2336

---

## 📈 Training & Validation Trends

- Training accuracy exceeds **90%**
- Validation accuracy stabilizes around **83–84%**
- Smooth convergence with no major overfitting

---

## 🔍 Confusion Matrix Analysis

- Strong diagonal dominance indicates high class-wise accuracy
- Most confusion observed between:
  - Sad ↔ Neutral
  - Fear ↔ Sad
  - Disgust ↔ Neutral

These overlaps are expected due to similar acoustic patterns in emotional speech.

---

## ⚙️ Best Hyperparameters

Best-performing hyperparameters obtained after tuning:

- LSTM Units (Layer 1): 256  
- LSTM Units (Layer 2): 256  
- LSTM Units (Layer 3): 128  
- LSTM Dropout: 0.4  
- Dense Dropout: 0.5  
- Dense Units: 256  
- Learning Rate: 0.001  
- Batch Size: 128  

---

## 🧠 Model Architecture

- Input: Time-series audio features  
- Three stacked **LSTM layers**
- Dropout layers for regularization
- Fully connected dense layer
- Softmax output layer for multi-class classification

---

## 🧪 Feature Extraction

Each audio sample is processed with:
- Sampling rate: **22,050 Hz**
- Duration: **3 seconds**

Extracted features:
- MFCC (40 coefficients)
- Delta MFCC
- Chroma STFT
- Log Mel Spectrogram
- Spectral Contrast
- Tonnetz

All features are:
- Concatenated and time-aligned
- Normalized using a saved scaler
- Reshaped to match LSTM input format

---

## 🖥️ Streamlit Web Application

An interactive web app is included for real-time emotion prediction.

### Features:
- Upload audio files (WAV, MP3, OGG, FLAC, M4A)
- Play uploaded audio
- Predict emotion with confidence score
- Visualizations:
  - Emotion probability distribution
  - Audio waveform
  - Mel spectrogram

### ▶️ Run the App

Run the following command from the project directory:

streamlit run project.py

Ensure the following files are present in the same directory:
- emotion_model.h5  
- best_model_label_encoder.joblib  
- best_model_feature_scaler.joblib  

---

## 📂 Project Structure

.
├── project.py                         # Streamlit application  
├── Model_Code.ipynb                   # Model training and experiments  
├── emotion_model.h5                   # Trained LSTM model  
├── best_model_label_encoder.joblib    # Label encoder  
├── best_model_feature_scaler.joblib   # Feature scaler  
├── best_params_acc_0_8326_*.json      # Best hyperparameters  
├── confusion_matrix_*.png             # Confusion matrix image  
├── training_history_*.png             # Accuracy and loss curves  
└── README.md  

---

## 🛠️ Technologies Used

- Python  
- TensorFlow / Keras  
- Librosa  
- NumPy  
- Scikit-learn  
- Streamlit  
- Plotly  

---

## 👥 Team Details

**Project Title:** Implementation of a Research Paper  

**Team Members:**
- Aditya Upendra Gupta (AD24B1003)  
- Kartavya Gupta (AD24B1028)  
- Anshika Agarwal (AD24B1007)  


**Institute:**  
Indian Institute of Information Technology, Raichur  

---

## 🚀 Future Scope

- Attention-based LSTM or Transformer models  
- Data augmentation for better generalization  
- Real-time microphone input  
- Multilingual speech emotion recognition  

---

Built with ❤️ using Deep Learning and Streamlit
