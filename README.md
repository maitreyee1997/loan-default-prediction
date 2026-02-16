# 💳 Loan Default Prediction App

An end-to-end Machine Learning project that predicts whether a loan applicant is likely to default or not.

The model is trained on a credit risk dataset and deployed using **Flask** with an HTML frontend interface.

---

## 📌 Project Overview

This project helps financial institutions identify high-risk loan applicants in advance to reduce financial losses.

It follows a complete real-world Data Science workflow:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training
- Cross Validation
- Model Evaluation
- Model Saving
- Web App Deployment using Flask

---

## 🧠 Problem Statement

Loan defaults result in significant financial losses for banks and lending institutions.

The objective of this project is to:

- Predict high-risk applicants
- Reduce default rates
- Support data-driven loan approval decisions

Target Variable:

```
loan_status  
0 = Non-Default  
1 = Default
```

---

## 📂 Dataset Information

- Source: Kaggle – Credit Risk Dataset
- Includes:
  - Applicant income
  - Loan amount
  - Credit history
  - Employment details
  - Other financial indicators

---

## ⚙️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- Joblib / Pickle
- Flask
- HTML (Frontend UI)

---

## 🔄 Project Pipeline

1️⃣ Data Loading & Exploratory Data Analysis  
2️⃣ Missing Value Treatment  
3️⃣ Categorical Encoding  
4️⃣ Feature Scaling  
5️⃣ Train-Test Split  
6️⃣ Model Training  
   - Logistic Regression  
   - Random Forest  
   - XGBoost  
7️⃣ Cross Validation  
8️⃣ Model Evaluation  
9️⃣ Model Saving using Pickle  
🔟 Flask App Integration  

---

## 🤖 Best Model

### XGBoost Classifier (Final Model)

- Accuracy: ~94%
- Recall (Default Class): ~73%
- Cross-validated performance
- Selected based on best balance between precision and recall

---

## 💾 Model Saving

The trained model is saved using:

```python
import pickle

with open("models/loan_model.pkl", "wb") as file:
    pickle.dump(model, file)
```

---

## 🌐 Web Application Deployment

The project includes a Flask-based web application.

### 🔹 Backend
- Flask (app.py)

### 🔹 Frontend
- HTML template (index.html)
- User-friendly form for entering applicant details
- Real-time prediction output

The app loads the trained model and returns prediction results through a simple web interface.

---

## 🚀 How to Run the Project Locally

### 1️⃣ Clone the Repository

```
git clone <your-repo-link>
cd loan_default_prediction
```

### 2️⃣ Create Virtual Environment

```
conda create -n loan_env python=3.10
conda activate loan_env
```

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Run Flask Application

```
python app.py
```

App will run at:

```
http://127.0.0.1:5000/
```

---

## 📂 Project Structure

```
loan_default_prediction/
│
├── data/
├── models/
│   └── loan_model.pkl
├── notebook/
│   └── loan_model_training.ipynb
├── templates/
│   └── index.html
├── app.py
├── requirements.txt
└── README.md
```

---

## 📈 Future Improvements

- Hyperparameter tuning
- SHAP feature importance
- Model monitoring
- Docker containerization
- Cloud deployment (AWS / Azure)

---

## 👩‍💻 Author

Maitreyee  
Data Analyst | Aspiring Data Scientist  

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
