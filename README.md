# House Price Prediction using Linear Regression

## Project Overview

This project predicts the selling price of a house using a Linear Regression model. The model is trained on the Ames Housing dataset and estimates house prices based on important property features such as living area, number of bedrooms, and number of bathrooms.

The project demonstrates the complete machine learning workflow, including data preprocessing, feature selection, model training, evaluation, model serialization, and deployment through a Streamlit web application.

---

## Features

- Predict house prices using a trained Linear Regression model
- Interactive Streamlit web application
- Real-time predictions based on user input
- Model saved and loaded using Joblib
- Clean and easy-to-use interface

---

## Dataset

Dataset: Ames Housing Dataset

### Features Used

- Living Area (GrLivArea)
- Number of Bedrooms (BedroomAbvGr)
- Number of Bathrooms (FullBath)

### Target Variable

- SalePrice

---

## Machine Learning Algorithm

- Linear Regression

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook

---

## Project Workflow

1. Load the dataset
2. Perform data preprocessing
3. Select relevant features
4. Split the dataset into training and testing sets
5. Train the Linear Regression model
6. Evaluate model performance
7. Save the trained model using Joblib
8. Deploy the model using Streamlit

---

## Model Evaluation

The model was evaluated using the following metrics:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## Project Structure

```
House-Price-Prediction/
│
├── app.py
├── House_Price_Prediction.ipynb
├── models/
│   └── house_model.pkl
├── requirements.txt
└── README.md
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/your-username/House-Price-Prediction.git
```

### Navigate to the project folder

```bash
cd House-Price-Prediction
```

### Install the required dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

The application will open automatically in your default web browser.

To explore the complete machine learning workflow, open the notebook:

```bash
jupyter notebook
```

---

