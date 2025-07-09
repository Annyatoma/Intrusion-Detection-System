# 🔐 Intrusion Detection System using Machine Learning

This project is aimed at developing an Intrusion Detection System (IDS) using Machine Learning techniques to detect malicious network activities. It leverages the NSL-KDD dataset and implements both Decision Tree and Artificial Neural Network (ANN) models for classification.

---

## 📌 Problem Statement

With increasing cyber threats and network traffic, traditional rule-based IDS systems are becoming inefficient. There is a need for an intelligent, automated system that can detect and classify suspicious behavior in real-time with high accuracy.

## Project Objectives

- Detect network intrusions using ML algorithms.

- Train models using real-world network data (NSL-KDD).

- Evaluate models using classification metrics.

- Provide a baseline for intelligent, adaptive IDS systems.

## 🛠️ Technologies Used

- Python 3.x

- Google Colab (T4 GPU)

- Libraries:pandas, numpy, scikit-learn, keras, matplotlib

## 📂 Dataset

- Name: NSL-KDD Dataset

- 41 features per connection, includes both numeric and categorical attributes.

## 📈 Model Overview
1. Decision Tree Classifier

    - Pros: Fast, interpretable
    
    - Accuracy: ~88%

2. Artificial Neural Network (ANN)
   
    - Deep learning model trained using Keras
    
    - Accuracy: 95%+

     - Architecture: Input layer → Hidden layers (ReLU) → Output layer (Softmax)

## 🧪 Steps Performed
1.Data Preprocessing

   - Label encoding of categorical variables
    
   - Feature normalization using StandardScaler

2.Model Training

   - Train-test split: 95:5
    
   - Training on Google Colab with GPU support

3.Evaluation

    - Accuracy, Loss vs Epoch plots
    
    - Confusion matrix, precision, recall

4.Deployment

    - Saved model in .keras format

## 🖼️ Results

 - Accuracy vs Epoch plot- 0.9938
    
 - Loss vs Epoch plot- 0.0214

## 🤝 Acknowledgments

This project is an AICTE student research submission aimed at developing an Intrusion Detection System (IDS) using Machine Learning techniques to detect malicious network activities.

## 🔮 Future Scope

-Real-time traffic analysis using Wireshark or scapy

-Use advanced models (LSTM, Transformers)

-Cloud deployment (AWS/GCP)

-Integration with firewall/alert systems

-Support for encrypted traffic (DPI)

## 👩‍💻 Author

Annyatoma Das
📧 annyatoma@gmail.com


