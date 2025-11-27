## churn_model ## End to End project -- using deep learning 

🧠 Customer Churn Prediction using ANN

This project predicts whether a bank customer is likely to churn (leave the bank) using an Artificial Neural Network (ANN).
It includes data preprocessing, model training, and a Streamlit web app for making real-time predictions.

📌 Features

Preprocessing of customer data

One-Hot Encoding & Label Encoding

Feature Scaling using StandardScaler

ANN model built using TensorFlow/Keras

User-friendly Streamlit web interface

Predicts churn probability in real-time

🏗️ Tech Stack

Python

TensorFlow / Keras

Scikit-learn

Pandas

NumPy

Streamlit

📂 Project Structure
ANN-Churn-Prediction/
│
├── model.h5                     # Trained ANN model
├── scaler.pkl                   # StandardScaler object
├── onehot_encoder_geo.pkl       # OneHotEncoder for Geography
├── label_encoder_gender.pkl     # Label/OneHot encoder for Gender
├── app.py                       # Streamlit web app
├── README.md                    # Project documentation
└── dataset.csv                  # Original dataset (optional)

🚀 How to Run the Project
1️⃣ Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/Mac

2️⃣ Install dependencies
pip install -r requirements.txt


(Or install manually: Streamlit, TensorFlow, scikit-learn, pandas, numpy)

3️⃣ Run the Streamlit app
streamlit run app.py


The app will open in your browser at:

http://localhost:8501

📊 Model Details

Input Layer: 11 features

Hidden Layers: Dense layers with ReLU activation

Output Layer: Sigmoid (binary classification)

Loss Function: Binary Crossentropy

Optimizer: Adam

🧪 Prediction Logic in the App

User enters customer details

Geography is One-Hot Encoded

Gender is encoded (LabelEncoder or OneHotEncoder)

Numeric values are scaled

Model predicts churn probability

App shows:

🟢 Likely to stay

🔴 Likely to churn

📎 Example Output
🟢 The customer is unlikely to churn.

Churn Probability: 0.11

🙌 Acknowledgments

This project is inspired by real-world customer retention use cases.
Built for practice and educational learning of ANNs and ML deployment.
