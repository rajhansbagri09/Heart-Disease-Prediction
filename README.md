# 💓 Heart Disease Risk Prediction

This is a **machine learning-based web app** that predicts the likelihood of developing heart disease using various medical parameters such as age, cholesterol, blood pressure, diabetes, and lifestyle habits.

The app is deployed using **Streamlit** and the model is built with a **Random Forest Classifier**.

---

## 🚀 Live Demo

🔗 [Click to try the live app](https://heart-disease-prediction-raj09.streamlit.app/)  
📦 [GitHub Repo](https://github.com/rajhansbagri09/Heart-Disease-Prediction)

---

## 📌 Features

- Clean, easy-to-use web UI built with **Streamlit**
- Predicts heart disease risk and displays a **confidence score**
- Automatically sets **0 cigarettes/day** if user selects non-smoker
- Scales input using `StandardScaler` for better model performance
- Trained on real-world dataset with missing values handled using median imputation
- Shows **binary classification output**: High Risk or Low Risk
