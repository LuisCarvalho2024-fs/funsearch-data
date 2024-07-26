import itertools

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB

"""
A model to detect higher chance of heart attack is being trained on a dataset.
These are the attributes of the dataset:

ID: Unique alphanumeric identifier for each entry in the dataset.
Customer_ID: Alphanumeric identifier for each customer.
Month: Month of data collection. Example value: "January".
Name: Name of the customer.
Age: Age of the customer.
SSN: Social Security Number of the customer.
Occupation: Occupation of the customer.
Annual_Income: Annual income of the customer.
Monthly_Inhand_Salary: Monthly salary after deductions.
Num_Bank_Accounts: Number of bank accounts the customer has.
Num_Credit_Card: Number of credit cards the customer has.
Interest_Rate: Interest rate applied on loans.
Num_of_Loan: Number of loans the customer has.
Type_of_Loan: Type of loans taken by the customer, separated by commas. Example column value: "Auto Loan, Credit-Builder Loan, Personal Loan, and Home Equity Loan".
Delay_from_due_date: Number of days delayed from due date for payments.
Num_of_Delayed_Payment: Number of delayed payments made by the customer.
Changed_Credit_Limit: Indicates if the credit limit has been changed.
Num_Credit_Inquiries: Number of credit inquiries made by the customer.
Credit_Mix: Mix of different types of credit accounts held by the customer. Possible column values: ["_", "Good", "Standard", "Bad"].
Outstanding_Debt: Amount of outstanding debt.
Credit_Utilization_Ratio: Ratio of credit used to credit available.
Credit_History_Age: Age of credit history. Example column value: "22 Years and 6 Months"
Payment_of_Min_Amount: Indicates if minimum payment amount is met.
Total_EMI_per_month: Total Equated Monthly Installment (EMI) paid by the customer.
Amount_invested_monthly: Amount invested monthly by the customer. Type string, potentially with bad format.
Payment_Behaviour: Payment behavior of the customer. Example column value: "Low_spent_Large_value_payments".
Monthly_Balance: Monthly balance in the account.

Complete the function to preprocesses the dataset, returning a dataset with the TEN most important columns.
Process the data in each chosen column in a way to improve the results.
Import libraries as needed.
"""

@funsearch.run
def run_evaluate():
  path_train = "dataset/credit_train_cleaned.csv"
  target_label = 'Credit_Score'
  df = pd.read_csv(path_train)
  
  df = select_columns_and_return_dataframe(df)
  
  if target_label in df:
    y = df[target_label].values
    df = df.drop(columns=[target_label])
  else:
    return [-1, []]
  
  column_names: list[str] = df.columns.tolist()
  column_names.sort()
  df = df[column_names]
  
  X = df.values
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
  models = generate_models(X_train, y_train)
  return [evaluate(models, X_test, y_test), column_names]

def generate_models(X_train, y_train):
    logistic_regression = LogisticRegression()
    k_neighbors = KNeighborsClassifier()
    decision_tree = DecisionTreeClassifier()
    naive_bayes = MultinomialNB()

    logistic_regression.fit(X_train, y_train)
    k_neighbors.fit(X_train, y_train)
    decision_tree.fit(X_train, y_train)
    naive_bayes.fit(X_train, y_train)

    return [logistic_regression, k_neighbors, decision_tree, naive_bayes]

def evaluate(models, X_test, y_test) -> float:
  """Returns the model accuracy."""
  f1s = []
  for model in models:
    y_pred = model.predict(X_test)
    f1s.append(f1_score(y_test, y_pred, average='micro'))
  f1_avg = sum(f1s) / len(f1s)
  return f1_avg

@funsearch.evolve
def select_columns_and_return_dataframe(df: pd.DataFrame) -> pd.DataFrame:
  # Preprocess as needed, select the 10 best columns and return df. Keep the target column "Credit_Score".
  # There should be no NaN values in the columns, they should either be dropped or filled.
  # Assume each column may contain bad data.
  df = df[['Outstanding_Debt', 'Credit_Score']].dropna()
  return df
