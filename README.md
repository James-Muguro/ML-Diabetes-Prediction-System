# Diabetes Risk Predictor: An ML-Powered Web Application

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Learning Objectives](#learning-objectives)
- [Technical Aspects](#technical-aspects)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Run the Application](#run-the-application)
- [Future Work and Enhancements](#future-work-and-enhancements)

## Overview

The **Diabetes Risk Predictor** is an application designed to predict whether an individual has diabetes based on several health metrics. Utilizing a dataset sourced from the [Kaggle Diabetes dataset](https://www.kaggle.com/), this project implements a Random Forest Classifier to provide reliable predictions based on the following features:

- Number of Pregnancies
- Glucose Level
- Blood Pressure
- Skin Thickness
- Insulin Level
- Body Mass Index (BMI)
- Diabetes Pedigree Function (DPF)
- Age

The goal is to offer a web-based interface for users to input their health data and receive a prediction regarding their diabetes status.

## Motivation

Diabetes is a significant and growing health concern, exacerbated by modern sedentary lifestyles. Early detection is crucial for effective management and prevention of complications. This project seeks to harness machine learning technology to enable early detection of diabetes, thereby potentially improving health outcomes for individuals. By creating a user-friendly application, we aim to make predictive healthcare more accessible.

## Learning Objectives

This project aims to provide practical experience with the following concepts:

- Data Gathering
- Descriptive Analysis
- Data Visualization
- Data Preprocessing
- Model Training and Evaluation
- Deployment of Machine Learning Models
- Building a Flask Web Application

## Technical Aspects

- **Model Training**: A Random Forest Classifier is employed for training on the dataset using the `scikit-learn` library.
- **Web Application**: A Flask-based web application allows users to input their health metrics.
- **User Input**: Users can enter data like number of pregnancies, insulin level, age, BMI, etc.
- **Result Display**: Predictions are displayed on a new page after submission.

## Technologies Used

- **Machine Learning**: Scikit-Learn (Random Forest Classifier)
- **Web Framework**: Flask (lightweight web app framework)
- **Data Libraries**: NumPy (numerical data handling), Pandas (data manipulation)
- **Deployment**: Heroku (cloud platform for deployment)
- **Programming Language**: Python
- **Visualization**: Matplotlib (data visualization)

## Installation

- **Clone the Repository**: Download the project files to your local machine.

- **Navigate to the Project Directory**: Use your terminal or command prompt to change into the project folder.

- **Set Up a Virtual Environment**: Create a new virtual environment to manage dependencies separately. Activate it to ensure you’re working within this environment.

- **Install Dependencies**: Use a package manager to install the required libraries listed in the requirements.txt file.

## Run the Application

To run the application, execute the following command:

```bash
python app.py
```

This will start the Flask development server, and you can access the application in your web browser at `http://127.0.0.1:5000/`.

## Future Work and Enhancements

While the current application provides basic functionality for diabetes prediction, there are several areas for potential enhancement:

- **Model Improvement**: Experimenting with other machine learning algorithms (e.g., XGBoost, Neural Networks) to potentially improve prediction accuracy.
- **User Interface Enhancements**: Improving the user interface for better user experience, including responsive design for mobile devices.
- **Data Visualization**: Adding data visualization features to help users understand the factors influencing their prediction.
- **Real-time Predictions**: Implementing real-time predictions using live data from wearable devices or health apps.
- **Broader Health Assessments**: Expanding the model to predict other health conditions based on additional metrics.
