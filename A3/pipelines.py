from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
#import SVM model
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import pandas as pd
import time
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from helpers import seed_decorator, generate_seed
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection


def preprocess_pipeline(X_train, list_of_category_columns):
    # Create a column transformer for preprocessing
    if "columns" in dir(X_train):
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), list(set(X_train.columns) - set(list_of_category_columns)))
            ],
            remainder='passthrough'
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), np.arange(X_train.shape[1]))
            ],
            remainder='passthrough'
        )

    return preprocessor

class DataFrameToNumpyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.columns = X.columns
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            return X.to_numpy()
        return X


def GMM_pipeline(X_train, list_of_categories=[], **kwargs):
    preprocessor = preprocess_pipeline(X_train, list_of_categories)
    gmm_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        # ('to_numpy', DataFrameToNumpyTransformer()),  # Convert to NumPy array
        ('classifier', GaussianMixture(**kwargs))
    ])
    gmm_pipeline.__class__.__name__ = 'GaussianMixture'
    return gmm_pipeline


def KMeans_pipeline(X_train, list_of_categories=[],**kwargs):
  # overrid classname

    preprocessor = preprocess_pipeline(X_train, list_of_categories)
    kmeans_pipeline =  Pipeline(steps=[
               ('preprocessor', preprocessor),
              ('classifier', KMeans(**kwargs))
          ])

    kmeans_pipeline.__class__.__name__ = f'KMeans'
    return kmeans_pipeline


def PCA_pipeline(X_train, list_of_categories=[],**kwargs):
  # overrid classname

    preprocessor = preprocess_pipeline(X_train, list_of_categories)
    pca_pipeline =  Pipeline(steps=[
               ('preprocessor', preprocessor),
              ('classifier', PCA(**kwargs))
          ])

    pca_pipeline.__class__.__name__ = f'PCA'
    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    return pca_pipeline


def PCANoStandard_pipeline(X_train, list_of_categories=[],**kwargs):
  # overrid classname

    preprocessor = preprocess_pipeline(X_train, list_of_categories)
    pca_pipeline =  Pipeline(steps=[
            #    ('preprocessor', preprocessor),
              ('classifier', PCA(**kwargs))
          ])

    pca_pipeline.__class__.__name__ = f'PCA'
    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    return pca_pipeline


def GaussianRP_pipeline(X_train, list_of_categories=[], **kwargs):
    preprocessor = preprocess_pipeline(X_train, list_of_categories)
    rp_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GaussianRandomProjection(**kwargs))
    ])
    rp_pipeline.__class__.__name__ = 'GaussianRandomProjection'
    return rp_pipeline

def GaussianRPNoStandard_pipeline(X_train, list_of_categories=[], **kwargs):
    preprocessor = preprocess_pipeline(X_train, list_of_categories)
    rp_pipeline = Pipeline(steps=[
        # ('preprocessor', preprocessor),
        ('classifier', GaussianRandomProjection(**kwargs))
    ])
    rp_pipeline.__class__.__name__ = 'GaussianRandomProjection'
    return rp_pipeline

def ICA_pipeline(X_train, list_of_categories=[], **kwargs):
    preprocessor = preprocess_pipeline(X_train, list_of_categories)
    ica_pipeline = Pipeline(steps=[
        # ('preprocessor', preprocessor),
        ('classifier', FastICA(**kwargs))
    ])
    ica_pipeline.__class__.__name__ = 'FastICA'
    return ica_pipeline




################### TSNE PIPELINE ############################
def TSNE_pipeline(X_train, list_of_categories=[],**kwargs):
  # overrid classname

    preprocessor = preprocess_pipeline(X_train, list_of_categories)
    tsne_pipeline =  Pipeline(steps=[
               ('preprocessor', preprocessor),
              ('classifier', TSNE(**kwargs))
          ])

    tsne_pipeline.__class__.__name__ = f'TSNE'
    return tsne_pipeline



class TSNEWithKLDivergence(BaseEstimator, TransformerMixin):
    def __init__(self, perplexity=30.0, **kwargs):
        self.perplexity = perplexity
        self.kwargs = kwargs
        self.tsne = TSNE(perplexity=self.perplexity, **self.kwargs)
        self.kl_divergence_ = None

    def fit(self, X, y=None):
        self.tsne.fit_transform(X)
        self.kl_divergence_ = self.tsne.kl_divergence_
        return self

    def transform(self, X):
        return self.tsne.embedding_

    def transform(self, X):
        return self.tsne.embedding_

def TSNEKL_pipeline(X_train, list_of_categories=[], **kwargs):
    preprocessor = preprocess_pipeline(X_train, list_of_categories)
    tsne_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', TSNE_pipeline(**kwargs))
    ])
    tsne_pipeline.__class__.__name__ = 'TSNE'
    return tsne_pipeline



