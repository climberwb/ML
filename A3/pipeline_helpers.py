import pipelines 
import pandas as pd
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import numpy as np
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import mean_squared_error

def kl_divergence_scorer(estimator, X):
    estimator.fit(X)
    return estimator.named_steps['classifier'].kl_divergence_

perplxity_grid = [5, 10, 20, 30, 40, 50 ]
param_grid = {
    'classifier__perplexity': perplxity_grid
}

# Function to perform GridSearchCV
def grid_search_tsne(X_train, category_columns, dataset_name, sns: sns, kwargs={"random_state": 42}):
    perplexity_values = perplxity_grid
    kl_divergences = []
    models = []
    for perplexity in perplexity_values:
        model = pipelines.TSNE_pipeline(X_train, list_of_categories=category_columns, perplexity=perplexity, **kwargs)
        model.fit(X_train)
        kl_divergences.append(model.named_steps['classifier'].kl_divergence_)
        models.append(model)
    results = pd.DataFrame({
        'Perplexity': perplexity_values,
        'KL Divergence': kl_divergences
    })
    
    sns.lineplot(data=results, x='Perplexity', y='KL Divergence', marker='o').set(
        title=f'TSNE Perplexity Tuning - {dataset_name}'
    )
    
    argmin_i = results['KL Divergence'].idxmin()
    best_perplexity = results.loc[argmin_i]['Perplexity']
    # get the best model 
    best_model = models[argmin_i]
    return best_perplexity, results, best_model


# Reconstruction error for PCA/ICA/RCA
def reconstruction_error(X_train, pipeline = pipelines.PCANoStandard_pipeline, list_of_categories=[], random_state=42, gather_correlation=False):
    components = []
    errors = []
    reconstruction_percentages = []
    avg_abs_correlations = []
    for n_components in range(1, X_train.shape[1]+1):
        # fit the preprocessing pipeline
        preprocessor_pipeline = pipelines.preprocess_pipeline(X_train, list_of_categories)
        
        # Create and fit the PCA pipeline
        X_standardized = preprocessor_pipeline.fit_transform( X_train)
        
        dr_pipeline = pipeline(X_train,  n_components=n_components, random_state=42)
        
        # Get the PCA-transformed data
        X_generated =dr_pipeline.fit_transform(X_standardized)
        
        # Reconstruct the data from the PCA-transformed data
        X_reconstructed = dr_pipeline.inverse_transform(X_generated)
        
        # Calculate reconstruction error
        error = np.mean(np.sum(np.square(X_standardized - X_reconstructed), axis=1))
        errors.append(error)
        percent_reconstructed = np.sqrt(mean_squared_error(X_standardized,X_reconstructed ))#np.sum(np.abs(1-np.sqrt(np.sum((X_standardized - X_reconstructed)**2 , axis=1))/ np.sqrt(np.mean(X_standardized**2,axis=1))))
        reconstruction_percentages.append(percent_reconstructed)
        #source https://stats.stackexchange.com/questions/194278/meaning-of-reconstruction-error-in-pca-and-lda
        components.append(n_components)
        
        if gather_correlation:
            # Calculate the correlation matrix
            avg_abs_correlations.append(average_abs_correlation(X_generated))
    if gather_correlation:
        return errors, reconstruction_percentages,components, avg_abs_correlations
    return errors, reconstruction_percentages,components





def average_abs_correlation(X_train):
    # remove  nan values
    X_train = X_train[~np.isnan(X_train).any(axis=1)]
    # Calculate the correlation matrix
    correlation_matrix = np.abs(np.corrcoef(X_train, rowvar=False))

    if type(correlation_matrix) == np.float64:
        return correlation_matrix
    # create nan mask
    mask = np.isnan(correlation_matrix)
    
    # Create a mask to get the upper triangular matrix
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    return (mask * correlation_matrix).sum().sum()/ np.sum(mask)
    
    
    
    