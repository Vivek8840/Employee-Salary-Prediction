import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import sys

# Load the trained model and preprocessor (ColumnTransformer and StandardScaler)
# We assume the entire pipeline including preprocessing is saved in 'model_2'
try:
    model_pipeline = joblib.load('salary_prediction_model.pkl')
except FileNotFoundError:
    print("Error: 'salary_prediction_model.pkl' not found. Please ensure the model is trained and saved.", file=sys.stderr)
    sys.exit(1)

def predict_salary(age, gender, education_level, job_title, years_of_experience):
    """
    Predicts salary based on input features.

    Args:
        age (int): Age of the employee.
        gender (str): Gender of the employee ('Male' or 'Female').
        education_level (str): Education level (e.g., 'Bachelor's', 'Master's', 'PhD').
        job_title (str): Job title (e.g., 'Software Engineer', 'Data Analyst').
        years_of_experience (int): Years of experience.

    Returns:
        float: Predicted salary.
    """
    input_data = pd.DataFrame([[age, gender, education_level, job_title, years_of_experience]],
                              columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])

    # Make prediction using the loaded pipeline
    predicted_salary = model_pipeline.predict(input_data)[0]
    return predicted_salary

if __name__ == "__main__":
    print("Welcome to the Salary Prediction Tool!")
    print("Please enter the details to predict salary.")

    try:
        age = int(input("Enter Age: "))
        gender = input("Enter Gender (Male/Female): ").strip()
        education_level = input("Enter Education Level (e.g., Bachelor's, Master's, PhD): ").strip()
        job_title = input("Enter Job Title: ").strip()
        years_of_experience = int(input("Enter Years of Experience: "))

        if gender not in ['Male', 'Female']:
            raise ValueError("Gender must be 'Male' or 'Female'.")
        # You might want to add more validation for education_level and job_title
        # based on the unique values in your training data.

        predicted = predict_salary(age, gender, education_level, job_title, years_of_experience)
        print(f"\nThe predicted salary is: ${predicted:,.2f}")

    except ValueError as ve:
        print(f"Invalid input: {ve}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)