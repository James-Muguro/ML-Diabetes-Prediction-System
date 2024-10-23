# Diabetes Risk Predictor: An ML-Powered Web Application

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Demo](#demo)
- [Learning Objectives](#learning-objectives)
- [Technical Aspects](#technical-aspects)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Run the Application](#run-the-application)
- [Bug / Feature Request](#bug--feature-request)
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

## Demo

You can explore a live demonstration of the application [here](https://mldiabete.herokuapp.com/).

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

![Made with Python](https://forthebadge.com/images/badges/made-with-python.svg)

[![Scikit-Learn](https://github.com/scikit-learn/scikit-learn/blob/master/doc/logos/scikit-learn-logo-small.png)](https://github.com/scikit-learn/)
[![Flask](https://flask.palletsprojects.com/en/1.1.x/_images/flask-logo.png)](https://flask.palletsprojects.com/)
![Heroku](https://github.com/ditikrushna/End-to-End-Diabetes-Prediction-Application-Using-Machine-Learning/blob/master/Resource/heroku.png)
![NumPy](https://github.com/ditikrushna/End-to-End-Diabetes-Prediction-Application-Using-Machine-Learning/blob/master/Resource/numpy.png)
![Pandas](https://github.com/ditikrushna/End-to-End-Diabetes-Prediction-Application-Using-Machine-Learning/blob/master/Resource/pandas.jpeg)

## Installation

To set up the project locally, follow these steps:

1. Clone this repository:

   ```bash
   git clone <repository-url>
   ```

2. Navigate to the project directory:

   ```bash
   cd ML-Diabetes-Prediction-System
   ```

3. Create a virtual environment with Python 3 and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

4. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Run the Application

To run the application, execute the following command:

```bash
python app.py
```

This will start the Flask development server, and you can access the application in your web browser at `http://127.0.0.1:5000/`.

## Bug / Feature Request

If you encounter any issues or have feature requests, please report them by opening an issue [here](https://github.com/ditikrushna/End-to-End-Diabetes-Prediction-Application-Using-Machine-Learning/issues).

## Future Work and Enhancements

While the current application provides basic functionality for diabetes prediction, there are several areas for potential enhancement:

- **Model Improvement**: Experimenting with other machine learning algorithms (e.g., XGBoost, Neural Networks) to potentially improve prediction accuracy.
- **User Interface Enhancements**: Improving the user interface for better user experience, including responsive design for mobile devices.
- **Data Visualization**: Adding data visualization features to help users understand the factors influencing their prediction.
- **Real-time Predictions**: Implementing real-time predictions using live data from wearable devices or health apps.
- **Broader Health Assessments**: Expanding the model to predict other health conditions based on additional metrics.
