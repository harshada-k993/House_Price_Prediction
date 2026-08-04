# House Price Prediction using Linear Regression

## Project Overview

This project predicts the selling price of a house using a **Linear Regression** model. The model is trained on housing data and estimates house prices based on important property features such as living area, number of bedrooms, and number of bathrooms.

The project demonstrates the complete machine learning workflow, including data preprocessing, feature selection, model training, evaluation, and prediction.

---

##  Dataset

**Dataset:** Ames Housing Dataset

The dataset contains various attributes describing residential homes.

### Features Used

- Living Area (GrLivArea)
- Number of Bedrooms (BedroomAbvGr)
- Number of Bathrooms (FullBath)

### Target Variable

- SalePrice

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Machine Learning Algorithm

- Linear Regression

---

## 📈 Project Workflow

1. Import the dataset
2. Perform data preprocessing
3. Select relevant features
4. Split the dataset into training and testing sets
5. Train the Linear Regression model
6. Evaluate model performance
7. Make predictions on new house data
8. Save the trained model using Joblib

---

##  Model Evaluation

Evaluation metrics used:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

##  Project Structure

```
House-Price-Prediction/
│
├── House_Price_Prediction.ipynb
├── house_model.pkl
├── README.md
└── requirements.txt
```

---

## How to Run

1. Clone the repository

```bash
git clone https://github.com/your-username/House-Price-Prediction.git
```

2. Install the required packages

```bash
pip install -r requirements.txt
```

3. Open the Jupyter Notebook

```bash
jupyter notebook
```

4. Run all the cells to train and test the model.

---

##  Sample Prediction

Input:

- Living Area: 1500 sq ft
- Bedrooms: 3
- Bathrooms: 2

Output:

```
Estimated House Price: $198,453
```

*(Sample output may vary depending on the trained model.)*

---

## Future Improvements

- Include more housing features
- Compare multiple regression algorithms
- Hyperparameter tuning
- Deploy as a web application using Streamlit

---

## 👩‍💻 Author

Harshada Kolekar

Machine Learning Internship Project
