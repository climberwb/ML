from sklearn.datasets import make_classification
from torch import nn, optim
from skorch import NeuralNetClassifier
import numpy  as np
# from pyperch.neural.backprop_nn import BackpropModule
import matplotlib.pyplot as plt
import torch
import random
from ucimlrepo import fetch_ucirepo
from sklearn.metrics import accuracy_score, make_scorer, f1_score, confusion_matrix, log_loss
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from helpers import seed_decorator


CATEGORY_COLUMNS_UCML = [ 'Marital Status', 
                          'Application mode', 
                          'Course', 
                          'Previous qualification', 
                          'Nacionality', 
                          "Mother's qualification", 
                          "Father's qualification", 
                          "Mother's occupation", "Father's occupation" ]

def _get_ucml_data():
  # fetch dataset
  predict_students_dropout_and_academic_success = fetch_ucirepo(id=697)

  # data (as pandas dataframes)
  X = predict_students_dropout_and_academic_success.data.features
  y = predict_students_dropout_and_academic_success.data.targets
  
  return X, y

def _remove_ucml_columns(X):

  try:
    CATEGORY_COLUMNS_UCML.remove('Nacionality')
    CATEGORY_COLUMNS_UCML.remove('Previous qualification')
  except:
    pass
  
  X[(X["Marital Status"]  != 1) & (X["Marital Status"] != 2) & (X["Marital Status"] != 4)]["Marital Status"] = -1
  return X 


@seed_decorator(seed=42)
def get_and_preprocess_sdas_data():
    X, y = _get_ucml_data()
    X = _remove_ucml_columns(X)

    # Create a column transformer
    ct = ColumnTransformer(
        [('encoding', OneHotEncoder(drop='first'), CATEGORY_COLUMNS_UCML)],
        remainder='passthrough',
        verbose_feature_names_out=True
    )

    # Fit and transform the data
    X_transformed = ct.fit_transform(X)

    if not np.issubdtype(X_transformed.dtype, np.number):
        print("The transformed data is not numeric. Converting to numeric.")
        X_transformed = X_transformed.astype(np.float64)

    # Get the feature names
    feature_names = ct.get_feature_names_out()

    try:
        X_transformed_df = pd.DataFrame(X_transformed.toarray(), columns=feature_names)
    except:
        X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_transformed_df, y, test_size=0.2, random_state=42)

    # Remove columns that have 'encoding' in their name where the count in X_train is 0
    columns_to_remove = [col for col in X_train.columns if 'encoding' in col and X_train[col].sum() == 0]
    X_train = X_train.drop(columns=columns_to_remove)
    X_test = X_test.drop(columns=columns_to_remove)

    print(f"Removed columns: {columns_to_remove}")
    print(f"Number of features after removal: {len(X_train.columns)}")

    return X_train, X_test, y_train, y_test


def y_label_encode(y_train,y_test):
    # Initialize and fit LabelEncoder
  label_encoder = LabelEncoder()
  y_train_encoded = label_encoder.fit_transform(y_train.values.ravel())
  y_test_encoded = label_encoder.transform(y_test.values.ravel())
  return y_train_encoded, y_test_encoded, label_encoder

@seed_decorator(seed=42)
def get_and_preprocess_raisin_data():
  predict_raison= fetch_ucirepo(id=850)

  # data (as pandas dataframes)
  X = predict_raison.data.features
  y = predict_raison.data.targets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  return X_train, X_test, y_train, y_test