# Importing essential libraries
import numpy as np
import pandas as pd
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Suppress any warnings
warnings.filterwarnings("ignore")

# Function to handle missing values imputation
def impute_missing_values(df, columns):
    df['Glucose'].fillna(df['Glucose'].mean(), inplace=True)
    df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace=True)
    df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace=True)
    df['Insulin'].fillna(df['Insulin'].median(), inplace=True)
    df['BMI'].fillna(df['BMI'].median(), inplace=True)
    return df

# Function to train and evaluate the RandomForest model
def train_evaluate_model(X_train, y_train, X_test, y_test):
    classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    return classifier

# Main script
if __name__ == "__main__":
    # Loading the dataset
    df = pd.read_csv('kaggle_diabetes.csv')

    # Renaming 'DiabetesPedigreeFunction' as 'DPF'
    df.rename(columns={'DiabetesPedigreeFunction': 'DPF'}, inplace=True)

    # Replace 0 values with NaN for specific columns
    columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[columns_to_impute] = df[columns_to_impute].replace(0, np.NaN)

    # Splitting the dataset into features and target variable
    X = df.drop(columns='Outcome')
    y = df['Outcome']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Impute missing values for both train and test sets
    X_train = impute_missing_values(X_train, columns_to_impute)
    X_test = impute_missing_values(X_test, columns_to_impute)

    # Train and evaluate the model
    classifier = train_evaluate_model(X_train, y_train, X_test, y_test)

    # Save the trained model as a pickle file
    filename = 'diabetes_classifier.pkl'
    with open(filename, 'wb') as model_file:
        pickle.dump(classifier, model_file)

    print(f"Model saved as {filename}")
