import pipelines 
import pandas as pd
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import numpy as np
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import mean_squared_error, silhouette_score
import matplotlib.pyplot as plt
from helpers import seed_decorator, increase_font_size

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
    
    
@seed_decorator(seed=42)
def get_silhouette_scores(X_datasets, list_of_categories=[],random_state=42, pipeline = pipelines.KMeans_pipeline):
    seeds = [42]
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

    # Initialize arrays to store silhouette scores for each seed
    silhouette_scores_sdas = {k: [] for k in range_n_clusters}
    silhouette_scores_raisin = {k: [] for k in range_n_clusters}
    X_train_sdas, X_train_raisin, = X_datasets[0], X_datasets[1]
    
    assert X_train_sdas.shape[0] > 1000, "SDAS data dims is not correct"
    assert X_train_raisin.shape[0] < 1000, "Raisin data dims is not correct"
    for seed in seeds:
        for num_clusters in range_n_clusters:
            # Initialise kmeans for SDAS
            kmean_kwargs = {'n_clusters': num_clusters, 'random_state': seed}
            kmeans_sdas =  pipeline(X_train_sdas, list_of_categories=list_of_categories, **kmean_kwargs)
            kmeans_sdas.fit(X_train_sdas)
            cluster_labels_sdas = kmeans_sdas['classifier'].labels_

            # Initialise kmeans for Raisin
            kmeans_raisin = pipeline(X_train_raisin, **kmean_kwargs)
            kmeans_raisin.fit(X_train_raisin)
            cluster_labels_raisin = kmeans_raisin['classifier'].labels_

            # Silhouette score
            silhouette_scores_sdas[num_clusters].append(silhouette_score(X_train_sdas, cluster_labels_sdas))
            silhouette_scores_raisin[num_clusters].append(silhouette_score(X_train_raisin, cluster_labels_raisin))


    # Calculate mean and standard deviation of silhouette scores across seeds
    mean_silhouette_sdas = [np.mean(silhouette_scores_sdas[k]) for k in range_n_clusters]
    std_silhouette_sdas = [np.std(silhouette_scores_sdas[k]) for k in range_n_clusters]

    mean_silhouette_raisin = [np.mean(silhouette_scores_raisin[k]) for k in range_n_clusters]
    std_silhouette_raisin = [np.std(silhouette_scores_raisin[k]) for k in range_n_clusters]
 
    # Highlight the maximum mean silhouette score points
    max_mean_sdas_score = max(mean_silhouette_sdas)
    max_mean_sdas_k = range_n_clusters[mean_silhouette_sdas.index(max_mean_sdas_score)]

    max_mean_raisin_score = max(mean_silhouette_raisin)
    max_mean_raisin_k = range_n_clusters[mean_silhouette_raisin.index(max_mean_raisin_score)]
    
    return range_n_clusters, mean_silhouette_sdas, std_silhouette_sdas, mean_silhouette_raisin, std_silhouette_raisin, max_mean_sdas_score, max_mean_sdas_k, max_mean_raisin_score, max_mean_raisin_k


@seed_decorator(seed=42)
def get_bic_scores(X_datasets, list_of_categories=[], random_state=42, pipeline=pipelines.GMM_pipeline):
    seeds = [42]
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

    # Initialize arrays to store BIC scores for each seed
    bic_scores_sdas = {k: [] for k in range_n_clusters}
    bic_scores_raisin = {k: [] for k in range_n_clusters}
    X_train_sdas, X_train_raisin = X_datasets[0], X_datasets[1]
    
    assert X_train_sdas.shape[0] > 1000, "SDAS data dims is not correct"
    assert X_train_raisin.shape[0] < 1000, "Raisin data dims is not correct"
    # scale the data
    X_train_raisin_scaled = pipelines.preprocess_pipeline(X_train_raisin, list_of_categories).fit_transform(X_train_raisin)
    X_train_sdas_scaled = pipelines.preprocess_pipeline(X_train_sdas, list_of_categories).fit_transform(X_train_sdas)
    
    for seed in seeds:
        for num_clusters in range_n_clusters:
            # Initialize GMM for SDAS
            gmm_kwargs = {'n_components': num_clusters, 'random_state': seed}
            gmm_sdas = pipeline(X_train_sdas, list_of_categories=list_of_categories, **gmm_kwargs)
            gmm_sdas.fit(X_train_sdas)
            try:
                bic_scores_sdas[num_clusters].append(gmm_sdas['classifier'].bic(X_train_sdas_scaled.to_numpy()))
            except:
                bic_scores_sdas[num_clusters].append(gmm_sdas['classifier'].bic(X_train_sdas_scaled))

            # Initialize GMM for Raisin
            gmm_raisin = pipeline(X_train_raisin, **gmm_kwargs)
            gmm_raisin.fit(X_train_raisin)
            try:
                bic_scores_raisin[num_clusters].append(gmm_raisin['classifier'].bic(X_train_raisin_scaled.to_numpy()))
            except:
                bic_scores_raisin[num_clusters].append(gmm_raisin['classifier'].bic(X_train_raisin_scaled))

    # Calculate mean and standard deviation of BIC scores across seeds
    mean_bic_sdas = [np.mean(bic_scores_sdas[k]) for k in range_n_clusters]
    std_bic_sdas = [np.std(bic_scores_sdas[k]) for k in range_n_clusters]

    mean_bic_raisin = [np.mean(bic_scores_raisin[k]) for k in range_n_clusters]
    std_bic_raisin = [np.std(bic_scores_raisin[k]) for k in range_n_clusters]

    # Highlight the minimum mean BIC score points
    min_mean_bic_sdas_score = min(mean_bic_sdas)
    min_mean_bic_sdas_k = range_n_clusters[mean_bic_sdas.index(min_mean_bic_sdas_score)]

    min_mean_bic_raisin_score = min(mean_bic_raisin)
    min_mean_bic_raisin_k = range_n_clusters[mean_bic_raisin.index(min_mean_bic_raisin_score)]
    
    return range_n_clusters, mean_bic_sdas, std_bic_sdas, mean_bic_raisin, std_bic_raisin, min_mean_bic_sdas_score, min_mean_bic_sdas_k, min_mean_bic_raisin_score, min_mean_bic_raisin_k



def plot_bic_scores(range_n_clusters, mean_bic_sdas, std_bic_sdas, mean_bic_raisin, std_bic_raisin, min_mean_bic_sdas_score, min_mean_bic_sdas_k, min_mean_bic_raisin_score, min_mean_bic_raisin_k, plt):
    # Plot the mean and standard deviation
    plt.errorbar(range_n_clusters, mean_bic_sdas, yerr=std_bic_sdas, fmt='bx-', label='SDAS data fit')
    plt.errorbar(range_n_clusters, mean_bic_raisin, yerr=std_bic_raisin, fmt='ro-', label='Raisin data fit')
    
    # Increase font size for readability
    plt.rcParams.update({'font.size': 12})
    
    # Highlight the minimum mean BIC score points
    plt.plot(min_mean_bic_sdas_k, min_mean_bic_sdas_score, "bo", markersize=10, label=f'SDAS min k={min_mean_bic_sdas_k}')
    plt.plot(min_mean_bic_raisin_k, min_mean_bic_raisin_score, "ro", markersize=10, label=f'Raisin min k={min_mean_bic_raisin_k}')

    plt.xlabel("Values of K")
    plt.ylabel("log(BIC) score")
    plt.title("EM: BIC Analysis For Optimal k seed=42")
    plt.legend()
    plt.yscale('log')  # Set y-axis to log scale
  

    print(f"Min BIC score for SDAS data: {min_mean_bic_sdas_score} at k={min_mean_bic_sdas_k}")
    print(f"Min BIC score for Raisin data: {min_mean_bic_raisin_score} at k={min_mean_bic_raisin_k}")



# Define the parameter grid for GridSearchCV
param_grid = {
    "classifier__n_components": range(1, 7),
    "classifier__covariance_type": ["spherical", "tied", "diag", "full"],
}

# Function to perform GridSearchCV and plot the results
@seed_decorator(seed=42)
def grid_search_gmm(X_train, category_columns, dataset_name):
    param_grid = {
        "classifier__n_components": range(1, 7),
        "classifier__covariance_type": ["spherical", "tied", "diag", "full"],
    }
    def gmm_bic_score(estimator, X):
        """Callable to pass to GridSearchCV that will use the BIC score."""
        return -estimator['classifier'].bic(estimator['preprocessor'].transform(X))  # Make it negative since GridSearchCV expects a score to maximize

    model = pipelines.GMM_pipeline(X_train, list_of_categories=category_columns, random_state=42)
    grid_search = GridSearchCV(
        model, param_grid=param_grid, scoring=gmm_bic_score
    )
    grid_search.fit(X_train)
    
    df = pd.DataFrame(grid_search.cv_results_)[
        ["param_classifier__n_components", "param_classifier__covariance_type", "mean_test_score"]
    ]
    df["mean_test_score"] = -df["mean_test_score"]
    df = df.rename(
        columns={
            "param_classifier__n_components": "Number of components",
            "param_classifier__covariance_type": "Type of covariance",
            "mean_test_score": "BIC score",
        }
    )
    
    min_bic_score = df["BIC score"].min()
    best_params = grid_search.best_params_
    
    sns.catplot(
        data=df,
        kind="bar",
        x="Number of components",
        y="BIC score",
        hue="Type of covariance",
    ).set(title=f'GMM Hyperparameter Tuning - {dataset_name}')
    
    return min_bic_score, best_params, df