import itertools

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

"""
This dataset consists of customer activity data from a telecom service. It will be used to develop predictive models to predict churn.
These are the attributes of the dataset:

State: string
Account length: integer
Area code: integer
International plan: string
Voice mail plan: string
Number vmail messages: integer
Total day minutes: double
Total day calls: integer
Total day charge: double
Total eve minutes: double
Total eve calls: integer
Total eve charge: double
Total night minutes: double
Total night calls: integer
Total night charge: double
Total intl minutes: double
Total intl calls: integer
Total intl charge: double
Customer service calls: integer

The target column has been removed from the dataset
Complete the function to preprocesses the dataset, returning a dataset with the 8 most important columns.
Process each column in any way that seems plausible, like one-hot-encoding or droping NA values.
Import libraries as needed.
"""

@funsearch.run
def run_evaluate():
  path="dataset/churn-bigml-80.csv"
  target_label = 'Churn'
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
  # Preprocess columns as needed and the return df with up to 8 columns
  df = df[['Total day minutes']]
  return df
