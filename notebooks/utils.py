import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hdbscan 
import umap
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import time
<<<<<<< HEAD
=======


def summary_plotter(scores, features, color_palette, cluster_number):
    """
    Plot feature means for members in a cluster.
    """
    
    # Get select cluster
    for_plotting = scores[scores['cluster']==cluster_number]
    for_plotting = for_plotting[features] # Get only feature columns
    for_plotting = pd.DataFrame(for_plotting.mean()) # Get average z-score of each member
    for_plotting.reset_index(inplace=True)
    for_plotting.columns = ['variable', 'value']
    
    
    # Plot
    fig, ax = plt.subplots()
    fig.set_dpi(120)
    fig.set_size_inches(10,7)
    
    if cluster_number == -1:
        plt.barh(for_plotting['variable'], for_plotting['value'], color=(0.5,0.5,0.5))
    else:
        plt.barh(for_plotting['variable'], for_plotting['value'], color=color_palette[int(cluster_number)])
    
    plt.axvline(0, ymin=0, ymax=1, color='grey', linewidth=2)
    ax.invert_yaxis()
    plt.title("Cluster {}".format(cluster_number))
    
    
>>>>>>> 756445fb173b0ddd19c53c5f04e12cf1c8bd2fd2
    
def zscore_fit(raw, features):
    """
    Fits a z-scorer transformer. Returns scaler.
    """
    
    # Zscore feature columns
    scaler = StandardScaler()
    scaler.fit(raw[features])
    
    return(scaler)



def zscore_predict(raw, features, scaler):
    """
    Z-scores df using provided transformer.
    """
    
    # Zscore feature columns
    df_scaled = scaler.transform(raw[features])
    df_scaled = pd.DataFrame(df_scaled)
    df_scaled.columns = features
    
    return(df_scaled)



def umap_fit(df_scaled, hypers_umap):
    """
    Fits a umap transformer. Returns transformer.
    """
    
    # Instantiate and fit
    reducer = umap.UMAP(
        **hypers_umap
        ).fit(X=df_scaled)
    
    return(reducer)



def umap_predict(reducer, df_scaled):
    """
    Reduces df_scaled using provided umap transformer.
    """
    
    # Reduce
    embedding = pd.DataFrame(reducer.transform(df_scaled))
    
    return(embedding)



def hdbscan_fit(embedding, hypers_hdbscan):
    """
    Fit hdbscan onto dataframe.
    """
    
    clusterer = hdbscan.HDBSCAN(
        **hypers_hdbscan
    ).fit(embedding)
    
    return(clusterer)



def hdbscan_predict(embedding, df_scaled, clusterer, force_predict=True):
    if force_predict:
        mem_vec = pd.DataFrame(hdbscan.membership_vector(clusterer, embedding.values))
        test_labels = mem_vec.idxmax(axis=1).to_numpy()
        strengths = mem_vec.max(axis=1).to_numpy()
    else:
        test_labels, strengths = hdbscan.approximate_predict(clusterer, embedding)

    # Get probabilities
    scores = pd.DataFrame(strengths)
    scores.columns = ['score']

    # Get clusters
    labels = pd.DataFrame(test_labels)
    labels.columns = ['cluster']

    # Join
    scores = scores.join(labels).join(embedding).join(df_scaled)
    n_clusters = sum(scores['cluster'].unique()!=-1)
    scores['cluster'].value_counts()
    
    return(scores)



def reporting_metrics(scores, include_noise=False):
    """
    Calculate reporting metrics for clustering output. 
    """
    
    # Number of clusters
    n_clusters = sum(scores['cluster'].unique()!=-1)

    # Percent clustered   
    percent_clustered = sum(scores['cluster']!=-1)/len(scores)*100

    # Silhouette score: Only include non-noise points
    if include_noise:
        for_sil_score = scores.copy()
    else:
        for_sil_score = scores[scores['cluster']!=-1]
    sil_score = silhouette_score(X = for_sil_score[[0,1]], labels = for_sil_score['cluster'])

    # Species evenness
    if include_noise:
        for_evenness = scores.copy()
    else:
        for_evenness = scores[scores['cluster']!=-1]
    for_evenness = for_evenness.groupby(['cluster']).count()[[0]]
    for_evenness.columns = ['n']
    for_evenness['total'] = for_evenness['n'].sum()
    for_evenness['prop'] = for_evenness['n']/for_evenness['total']
    for_evenness['shannon'] = for_evenness['prop']*(np.log(for_evenness['prop']))
    for_evenness
    max_shannon = np.log(len(for_evenness))
    shannon = -for_evenness['shannon'].sum()
    pielou_evenness = (shannon)/max_shannon
    
    return(
        {
            'n_clusters': n_clusters,
            'percent_clustered': percent_clustered,
            'sil_score': sil_score,
            'pielou_evenness': pielou_evenness
        }
    )



<<<<<<< HEAD
def color_setter(n_clusters, scores, variable, variable_score, palette_name='deep'):
=======
def color_setter(n_clusters, scores):
>>>>>>> 756445fb173b0ddd19c53c5f04e12cf1c8bd2fd2
    """
    Sets color palette and color values per data point.
    """
    
<<<<<<< HEAD
    color_palette = sns.color_palette(palette_name, n_clusters)
=======
    color_palette = sns.color_palette('deep', n_clusters)
>>>>>>> 756445fb173b0ddd19c53c5f04e12cf1c8bd2fd2

    # Assign color based on index in color_palette (e.g., cluster 0 will be color_palette[0])
    # Else, assign as grey
    cluster_colors = [color_palette[x] if x >= 0
                      else (0.5, 0.5, 0.5) # Grey
<<<<<<< HEAD
                      for x in scores[variable]]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                             zip(cluster_colors, scores[variable_score])]
=======
                      for x in scores['cluster']]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                             zip(cluster_colors, scores['score'])]
>>>>>>> 756445fb173b0ddd19c53c5f04e12cf1c8bd2fd2

    return({
        "color_palette": color_palette,
        "cluster_colors": cluster_colors,
        "cluster_member_colors": cluster_member_colors
    })



<<<<<<< HEAD
def plot_clusters(embedding, plot_sample_size, cluster_member_colors, point_alpha, plot_title=None):
=======
def plot_clusters(embedding, plot_sample_size, cluster_member_colors, plot_title=None):
>>>>>>> 756445fb173b0ddd19c53c5f04e12cf1c8bd2fd2
    """
    Plots hdbscan clusters with corresponding colors.
    """
    
    # Sample
    for_plotting = embedding.sample(plot_sample_size)

    # Plot
    fig, ax = plt.subplots()
    fig.set_dpi(120)
    fig.set_size_inches(10,7)
    plt.title(plot_title)
<<<<<<< HEAD
    plot = plt.scatter(for_plotting[0], for_plotting[1], alpha=point_alpha, c=[cluster_member_colors[i] for i in for_plotting.index])



def summary_plotter(scores, features, variable, color_palette, cluster_number, aggregator='median', xlim_params=None, top_n=None):
    """
    Plot feature means for members in a cluster.
    """
    
    # Get select cluster
    for_plotting = scores[scores[variable]==cluster_number]
    for_plotting = for_plotting[features] # Get only feature columns
    if aggregator == 'median':
        for_plotting = pd.DataFrame(for_plotting.median()) # Get average z-score of each member
    else:
        for_plotting = pd.DataFrame(for_plotting.mean()) # Get average z-score of each member
    for_plotting.reset_index(inplace=True)
    for_plotting.columns = ['variable', 'value']

    # Get top n
    if top_n is not None:
        for_plotting.sort_values('value', ascending=False, inplace=True)
        for_plotting = for_plotting.head(top_n)
    
    # Plot
    fig, ax = plt.subplots()
    fig.set_dpi(120)
    fig.set_size_inches(10,7)
    
    if cluster_number == -1:
        plt.barh(for_plotting['variable'], for_plotting['value'], color=(0.5,0.5,0.5))
    else:
        plt.barh(for_plotting['variable'], for_plotting['value'], color=color_palette[int(cluster_number)])
    
    plt.axvline(0, ymin=0, ymax=1, color='grey', linewidth=2)
    if xlim_params is not None:
        plt.xlim(**xlim_params)

    ax.invert_yaxis()
    plt.title("Cluster {}".format(cluster_number))



def clustering_wrapper(df, features, hypers_umap, hypers_hdbscan, plot_sample_size, point_alpha, force_predict=True, include_noise=False, plot_title=None):
=======
    plot = plt.scatter(for_plotting[0], for_plotting[1], alpha=0.25, c=[cluster_member_colors[i] for i in for_plotting.index])




def clustering_wrapper(df, features, hypers_umap, hypers_hdbscan, plot_sample_size, force_predict=True, include_noise=False, plot_title=None):
>>>>>>> 756445fb173b0ddd19c53c5f04e12cf1c8bd2fd2
    """
    Does one round of umap + hdbscan.
    """

    # Time
    t0 = time.time()

    # Fit zscorer
    scaler = zscore_fit(df, features)

    # Transform
    df_scaled = zscore_predict(df, features, scaler)

    # Fit umap transformer
    reducer = umap_fit(df_scaled, hypers_umap)

    # Reduce
    embedding = umap_predict(reducer, df_scaled)

    # Initialize
    clusterer = hdbscan_fit(embedding, hypers_hdbscan)

    # Predict
<<<<<<< HEAD
    try:
        scores = hdbscan_predict(embedding, df_scaled, clusterer, force_predict)
        scores.index = df.index
        scores.reset_index(inplace=True)
    except KeyError: # Means no clusters detected
        print("No clusters detected")
        return(None)


    # Get reporting metrics
    metrics = reporting_metrics(scores, include_noise)

    # Define n_clusters
    n_clusters = metrics['n_clusters']

    # Get colors
    colors = color_setter(n_clusters, scores, 'cluster', 'score')

    # Plot
    plot_clusters(embedding, plot_sample_size, colors['cluster_member_colors'], point_alpha, plot_title)

    # Runtime
    runtime = time.time()-t0

    # Compile
    output = {
        'scaler': scaler, # zscore
        'reducer': reducer, # umap
        'clusterer': clusterer, # hdbscan
        'scores': scores,
        'colors': colors,
        'metrics': metrics,
        'runtime': runtime
    }

    return(output)


def top_n_extractor(x, top_n, only_positive=True):
    """
    Extract top n features (by value) for row x from a pivot table

    """
    top = x.sort_values(ascending=False)
    if only_positive:
        top = top[top>0]
    return top[0:top_n].index.tolist()


def summary_plotter_counts(scores, features, variable, color_palette, cluster_number, xlim_params=None, top_n=5, only_positive=True):
    """
    Count-based version of summary plotter. 
    
    """

    # Get select cluster
    for_plotting = scores[scores[variable]==cluster_number]
    for_plotting = for_plotting[features] # Get only feature columns
    
    # Get all top n
    out = for_plotting.apply(lambda x: top_n_extractor(x, top_n, only_positive), axis=1)
    
    # Get list of all top n
    all_top =[]
    for x in out:
        all_top.extend(x)

    # Get counts
    all_top = pd.DataFrame({'n_customers': pd.Series(all_top).value_counts()}).head(top_n)
    all_top['all_customers'] = for_plotting.shape[0]
    all_top['proportion'] = all_top['n_customers']/all_top['all_customers']
    all_top = all_top.sort_values('proportion')
    
    # Plot
    fig, ax = plt.subplots()
    fig.set_dpi(120)
    fig.set_size_inches(10,7)

    if cluster_number == -1:
        plt.barh(all_top.index, all_top['proportion'], color=(0.5,0.5,0.5))
    else:
        plt.barh(all_top.index, all_top['proportion'], color=color_palette[int(cluster_number)])

    plt.axvline(0, ymin=0, ymax=1, color='grey', linewidth=2)
    if xlim_params is not None:
        plt.xlim(**xlim_params)

    plt.title("Cluster {}".format(cluster_number))


def clustering_wrapper_predict(df, features, scaler, reducer, clusterer, plot_sample_size, point_alpha, force_predict=True, include_noise=False, plot_title=None):
    """
    Does one round of umap + hdbscan.
    """

    # Time
    t0 = time.time()

    # Transform
    df_scaled = zscore_predict(df, features, scaler)

    # Reduce
    embedding = umap_predict(reducer, df_scaled)

    # Predict
    try:
        scores = hdbscan_predict(embedding, df_scaled, clusterer, force_predict)
        scores.index = df.index
        scores.reset_index(inplace=True)
    except KeyError: # Means no clusters detected
        print("No clusters detected")
        return(None)
=======
    scores = hdbscan_predict(embedding, df_scaled, clusterer, force_predict)
    scores.index = df.index
    scores.reset_index(inplace=True)
>>>>>>> 756445fb173b0ddd19c53c5f04e12cf1c8bd2fd2

    # Get reporting metrics
    metrics = reporting_metrics(scores, include_noise)

    # Define n_clusters
    n_clusters = metrics['n_clusters']

    # Get colors
<<<<<<< HEAD
    colors = color_setter(n_clusters, scores, 'cluster', 'score')

    # Plot
    plot_clusters(embedding, plot_sample_size, colors['cluster_member_colors'], point_alpha, plot_title)
=======
    colors = color_setter(n_clusters, scores)

    # Plot
    plot_clusters(embedding, plot_sample_size, colors['cluster_member_colors'], plot_title)
>>>>>>> 756445fb173b0ddd19c53c5f04e12cf1c8bd2fd2

    # Runtime
    runtime = time.time()-t0

    # Compile
    output = {
        'scaler': scaler, # zscore
        'reducer': reducer, # umap
        'clusterer': clusterer, # hdbscan
        'scores': scores,
        'colors': colors,
        'metrics': metrics,
        'runtime': runtime
    }

    return(output)