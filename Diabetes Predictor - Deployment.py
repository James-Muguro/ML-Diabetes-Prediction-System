# Importing essential libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Loading the dataset
df = pd.read_csv('kaggle_diabetes.csv')

# Renaming DiabetesPedigreeFunction as DPF
df.rename(columns={'DiabetesPedigreeFunction': 'DPF'}, inplace=True)

# Replacing the 0 values from ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'] with NaN
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)

# Splitting the dataset into features and target variable
X = df.drop(columns='Outcome')
y = df['Outcome']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Imputing missing values after splitting
X_train['Glucose'].fillna(X_train['Glucose'].mean(), inplace=True)
X_train['BloodPressure'].fillna(X_train['BloodPressure'].mean(), inplace=True)
X_train['SkinThickness'].fillna(X_train['SkinThickness'].median(), inplace=True)
X_train['Insulin'].fillna(X_train['Insulin'].median(), inplace=True)
X_train['BMI'].fillna(X_train['BMI'].median(), inplace=True)

X_test['Glucose'].fillna(X_train['Glucose'].mean(), inplace=True)  # Use the training set mean
X_test['BloodPressure'].fillna(X_train['BloodPressure'].mean(), inplace=True)
X_test['SkinThickness'].fillna(X_train['SkinThickness'].median(), inplace=True)
X_test['Insulin'].fillna(X_train['Insulin'].median(), inplace=True)
X_test['BMI'].fillna(X_train['BMI'].median(), inplace=True)

# Creating Random Forest Model
classifier = RandomForestClassifier(n_estimators=100, random_state=0)  # Increased n_estimators for potentially better performance
classifier.fit(X_train, y_train)

# Predicting on the test set
y_pred = classifier.predict(X_test)

# Evaluating the model
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Creating a pickle file for the classifier
filename = 'diabetes-prediction-rfc-model.pkl'
with open(filename, 'wb') as model_file:
    pickle.dump(classifier, model_file)