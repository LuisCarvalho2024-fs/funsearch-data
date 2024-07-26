import itertools

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

"""
A model to detect higher chance of heart attack is being trained on a dataset.
These are the attributes of the dataset:

- age : Age of the patient
- sex : Sex of the patient
- cp : Chest pain type ~ 0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal Pain, 3 = Asymptomatic
- trtbps : Resting blood pressure (in mm Hg)
- chol : Cholestoral in mg/dl fetched via BMI sensor
- fbs : (fasting blood sugar > 120 mg/dl) ~ 1 = True, 0 = False
- restecg : Resting electrocardiographic results ~ 0 = Normal, 1 = ST-T wave normality, 2 = Left ventricular hypertrophy
- thalachh : Maximum heart rate achieved
- oldpeak : Previous peak
- slp : Slope
- caa : Number of major vessels
- thall : Thalium Stress Test result ~ (0,3)
- exng : Exercise induced angina ~ 1 = Yes, 0 = No

The target column has been removed from the dataset.
Complete the function to preprocesses the dataset, returning a dataset with up to FIVE important columns.
Process each column in any way that seems plausible.
Import libraries as needed.
"""

@funsearch.run
def run_evaluate():
  path="dataset/heart.csv"
  target_label = 'output'
  df = pd.read_csv(path)
  
  y = df[target_label].values
  df = df.drop(columns=[target_label])
  df : pd.DataFrame = select_columns_and_return_dataframe(df)
  
  column_names: list[str] = df.columns.tolist()
  column_names.sort()
  df = df[column_names]
  
  X = df.values
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
  models = generate_models(X_train, y_train)
  return [evaluate(models, X_test, y_test), column_names]

def generate_models(X_train, y_train):
    logistic_regression = LogisticRegression()
    svm = SVC()
    k_neighbors = KNeighborsClassifier()
    decision_tree = DecisionTreeClassifier()
    naive_bayes = GaussianNB()

    logistic_regression.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    k_neighbors.fit(X_train, y_train)
    decision_tree.fit(X_train, y_train)
    naive_bayes.fit(X_train, y_train)

    return [logistic_regression, svm, k_neighbors, decision_tree, naive_bayes]

def evaluate(models, X_test, y_test) -> float:
  """Returns the model accuracy."""
  f1s = []
  for model in models:
    y_pred = model.predict(X_test)
    f1s.append(f1_score(y_test, y_pred))
  f1_avg = sum(f1s) / len(f1s)
  return f1_avg

@funsearch.evolve
def select_columns_and_return_dataframe(df: pd.DataFrame) -> pd.DataFrame:
  # Preprocess as needed, select the 5 best columns and return df
  df = df[['age']]
  return df
