🏦 Bank Customer Churn Prediction using ANN
📌 Project Overview

This project predicts whether a bank customer will leave the bank (churn) using an Artificial Neural Network (ANN) built with TensorFlow/Keras.

The model is trained on customer demographic and financial data.

📊 Dataset
Source: Bank Customer Churn Dataset
Features include:
. Credit Score
. Geography
. Gender
. Age
. Balance
. Number of Products
. Has Credit Card
. Is Active Member
. Estimated Salary

Target: Exited (0 = Retained, 1 = Churned)

🧠 Model Architecture
Input Layer: 10–12 features (after encoding)
Hidden Layer 1: 16 neurons (ReLU)
Hidden Layer 2: 8 neurons (ReLU)
Output Layer: 1 neuron (Sigmoid)

Loss Function:
. Binary Crossentropy

Optimizer:
. Adam

📈 Performance
Accuracy: ~86%
Precision (Class 1): ~0.71
Recall (Class 1): ~0.48

Note: Model shows imbalance sensitivity (common in churn datasets)

⚙️ Tech Stack
Python
TensorFlow / Keras
Pandas
NumPy
Scikit-learn
Matplotlib / Seaborn