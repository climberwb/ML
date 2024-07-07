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
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join('./', 'pyperch')))
from pyperch.neural.backprop_nn import BackpropModule
import helpers


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




class CustomBackpropModule(BackpropModule):
    
    def forward(self, X):
        # Ensure the input is float32
        X = X.to(torch.float32)
        return super().forward(X)

    def float(self):
        # Convert all parameters to float32
        for param in self.parameters():
            param.data = param.data.float()
        return self
    
    


def NN_pipeline(X_train, list_of_categories, dr_pipeline = None,**kwargs):
    """
    NN_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', NeuralNetClassifier(BackpropModule, **kwargs))
    ])
    MLPClassifier.__class__.__name__ = f'NN'
    https://github.com/jlm429/pyperch/blob/master/notebooks/backprop_network.ipynb
    """

    # Define a function that converts data to numpy arrays
    def to_numpy(X):
        return np.array(X)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    dict_of_params = {
 
        "module__hidden_units": 10,
        "module__hidden_layers": 1,
        "module__activation": "tanh",
        "max_epochs": 500,
        "verbose": 0,
        "lr": 0.05,
            "batch_size": 128,
        "iterator_train__shuffle": False
    }
    dict_of_params.update(kwargs)

    # Define the F1 score callback for training and validation
    train_f1 = EpochScoring(scoring='f1_macro', lower_is_better=False, on_train=True, name='train_f1')
    valid_f1 = EpochScoring(scoring='f1_macro', lower_is_better=False, on_train=False, name='valid_f1')
    generate_seed()
    net = NeuralNetClassifier(
        module=CustomBackpropModule,
        callbacks=[EpochScoring(scoring='accuracy', name='train_acc', on_train=True), ('train_f1', train_f1), ('valid_f1', valid_f1)],
        #  train_split=predefined_split(dataset_valid),
        criterion=torch.nn.CrossEntropyLoss,
        optimizer = torch.optim.Adam,
        device = device,
        **dict_of_params
    )
    preprocessor = preprocess_pipeline(X_train, list_of_categories)
    steps = [
        ('preprocessor', preprocessor),
        ('dr_pipeline', dr_pipeline),
        ('classifier', net)
    ]
    if dr_pipeline is None:
        steps.pop(1)
    # HACK Ensure the module's weights are converted to float32
    net.initialize()
    net.module_.float()
    
    net_pipeline = Pipeline(steps=steps)

    net_pipeline.__class__.__name__ = f'NN'
    return net_pipeline

# Custom transformer for clustering and one-hot encoding
class ClusterAndEncode(BaseEstimator, TransformerMixin):
    def __init__(self, clusterer, **kwargs):
        self.clusterer = clusterer
        self.encoder = OneHotEncoder(sparse_output=False)

    def fit(self, X, y=None):
        clusters = self.clusterer.fit_predict(X)
        self.encoder.fit(clusters.reshape(-1, 1))
        return self

    def transform(self, X, y=None):
        clusters = self.clusterer.predict(X)
        encoded_clusters = self.encoder.transform(clusters.reshape(-1, 1))
        
        return np.hstack((X, encoded_clusters))

    def fit_transform(self, X, y=None):
        
        clusters = self.clusterer.fit_predict(X)
        encoded_clusters = self.encoder.fit_transform(clusters.reshape(-1, 1))
        return np.hstack((X, encoded_clusters))
    
    
# Update the NN_pipeline to include clustering and one-hot encoding

def NN_pipeline_with_clustering(X_train, list_of_categories,  kwargs_nn, kwargs_cluster, cluster_method='kmeans'):
    def to_numpy(X):
        return np.array(X)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    dict_of_params = {
        "module__hidden_units": 10,
        "module__hidden_layers": 1,
        "module__activation": "tanh",
        "max_epochs": 500,
        "verbose": 0,
        "lr": 0.05,
        "batch_size": 128,
        "iterator_train__shuffle": False
    }
    dict_of_params.update(kwargs_nn)

    train_f1 = EpochScoring(scoring='f1_macro', lower_is_better=False, on_train=True, name='train_f1')
    valid_f1 = EpochScoring(scoring='f1_macro', lower_is_better=False, on_train=False, name='valid_f1')
    generate_seed()
    net = NeuralNetClassifier(
        module=CustomBackpropModule,
        callbacks=[EpochScoring(scoring='accuracy', name='train_acc', on_train=True), ('train_f1', train_f1), ('valid_f1', valid_f1)],
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        device=device,
        **dict_of_params
    )
    
    preprocessor = preprocess_pipeline(X_train, list_of_categories)

    if cluster_method == 'kmeans':
        clusterer = KMeans
    elif cluster_method == 'gmm':
        clusterer = GaussianMixture
    else:
        raise ValueError("Invalid cluster method. Choose 'kmeans' or 'gmm'.")
    cluster_and_encode_instance = clusterer(**kwargs_cluster)
    cluster_and_encode = ClusterAndEncode(cluster_and_encode_instance, **kwargs_cluster)

    steps = [
        ('preprocessor', preprocessor),
        ('cluster_and_encode', cluster_and_encode),
        ('classifier', net)
    ]

    net_pipeline = Pipeline(steps=steps)
    net_pipeline.__class__.__name__ = 'NN'
    
    return net_pipeline
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



