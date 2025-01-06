# Chennai Land Price Prediction using Machine Learning

This project focuses on building a machine learning model to predict land prices in Chennai. The project includes data preprocessing, feature engineering, and model optimization to achieve accurate predictions. Additionally, the model is integrated into a Streamlit-based web application for easy user interaction and hosted on AWS EC2 for enhanced scalability and accessibility.

## Introduction

The **Chennai Land Price Prediction** project aims to assist users in predicting land prices based on specific attributes. This tool is designed for real estate buyers, sellers, and investors to evaluate land prices efficiently. Using machine learning models, a Streamlit web application, and AWS EC2 deployment, the project provides instant price estimates based on user inputs.

---

## Project Scope

The project utilizes historical data on land sales in Chennai, incorporating various features such as location, size, type, and surrounding infrastructure. The objective is to create a machine learning model that predicts land prices with high accuracy. The model is deployed in a user-friendly Streamlit interface hosted on AWS EC2 for robust performance and seamless user access.

---

## Packages Used

### Pandas
- Pandas is utilized for data manipulation and preprocessing.
- Learn more about Pandas [here](https://pandas.pydata.org/docs/).

### NumPy
- Used for numerical operations and handling missing data.
- Learn more about NumPy [here](https://numpy.org/doc/stable/).

### Scikit-learn
- Used for data preprocessing, feature encoding, and model evaluation.
- Learn more about Scikit-learn [here](https://scikit-learn.org/stable/).

### XGBoost
- XGBoost is the primary model for prediction, offering high performance and accuracy.
- Learn more about XGBoost [here](https://xgboost.readthedocs.io/en/stable/).

### Streamlit
- Used to develop the web-based interface for user interaction.
- Learn more about Streamlit [here](https://docs.streamlit.io/).

### Joblib
- Used for saving and loading the trained model and preprocessing pipelines.
- Learn more about Joblib [here](https://joblib.readthedocs.io/en/stable/).

### Matplotlib & Seaborn
- Used for data visualization and exploratory data analysis (EDA).
- Learn more about Matplotlib [here](https://matplotlib.org/) and Seaborn [here](https://seaborn.pydata.org/).

### CSS
- A custom CSS file enhances the Streamlit UI for a better user experience.

---

## Installation

To set up the environment, run the following commands to install the required libraries:

```bash
pip install numpy pandas scikit-learn xgboost streamlit matplotlib seaborn joblib
```

---

## File Structure

1. **`Data_Preprocessing.ipynb`**  
   - Prepares the raw data by handling missing values, encoding categorical features, and normalizing data. The cleaned data is saved as `Preprocessed_data.csv`.
     
2. **`Preprocessed_data.csv`**  
   - The cleaned dataset used for training and testing.
    
3. **`LandPrice_Prediction_Model.ipynb`**  
   - Trains multiple models (e.g., Random Forest, XGBoost) and evaluates them. The best-performing model (XGBoost) is saved as `xgboost_best_model.pkl`.
  
4. **`encode_categorical_columns.pkl`**  
   - Stores mappings for categorical feature encoding.

5. **`xgboost_best_model.pkl`**  
   - The trained machine learning model used for predictions.

6. **`StreamLit_UI.py`**  
   - A Streamlit script that provides an interface for users to input land details and get price predictions using the trained model.

7. **`style.css`**  
   - Custom CSS to style the Streamlit web application.

8. **`train-chennai-sale.csv`**  
   - The raw dataset containing land price information.

9. **`Chennai_Price_Prediction_logo.webp`**  
   - The project logo for branding purposes.

---

## How to Run the Code

1. **Preprocess the data:**
   - Run the `Data_Preprocessing.ipynb` notebook to clean and prepare the dataset.

2. **Train the model:**
   - Run the `LandPrice_Prediction_Model.ipynb` notebook to train and save the model.

3. **Launch the web application:**
   - Use the following command to run the Streamlit app:
     ```bash
     streamlit run StreamLit_UI.py
     ```

---

## Features of the Web Application

- User-friendly interface to input land details such as size, location, and type.
- Instant predictions of land prices using the trained XGBoost model.
- Interactive and visually appealing design enhanced with CSS styling.

---

## Model Evaluation

- **Pre-Tuning Score**: Initial performance metrics of the XGBoost model.
- **Post-Tuning Score**: Improved performance after hyperparameter tuning.

| Metric              | Before Tuning      | After Tuning |
|---------------------|--------------------|--------------|
| Mean Absolute Error | 317394.2           | 219982.95    |
| R² Score            | 0.987434           | 0.9939322    |

---

## Streamlit UI

The web application provides a seamless experience for users. Below are sample screenshots:

1. **Before Prediction**  
   ![StreamUI before Prediction-1](https://github.com/user-attachments/assets/f6edd8aa-a0e9-4175-9c72-f0abfec4b350)
   ![StreamUI before Prediction-2](https://github.com/user-attachments/assets/05c3fc88-de58-4bf3-bfb6-4de37cb2b3cf)

2. **After Prediction**  
   ![StreamUI After Prediction](https://github.com/user-attachments/assets/28cee394-861b-4cc9-ab2d-589c742733b2)

---

## Deployment on AWS EC2

The project is deployed on an AWS EC2 instance to ensure reliability, scalability, and accessibility. Users can interact with the web application through a public URL, accessing the prediction tool from anywhere without local setup. 

**Key benefits of AWS EC2 deployment:**
1. Scalable infrastructure to handle user requests efficiently.
2. Secure and reliable hosting for the application.
3. Easy accessibility via the internet.

**Steps for deployment:**
1. Set up an AWS EC2 instance with the required specifications.
2. Install the necessary libraries and dependencies on the instance.
3. Deploy the Streamlit application and configure it for public access.

---


