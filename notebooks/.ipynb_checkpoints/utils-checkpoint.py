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
        plt.barh(for_plotting['variable'], for_plotting['value'], color=color_palette[cluster_number])
    
    plt.axvline(0, ymin=0, ymax=1, color='grey', linewidth=2)
    ax.invert_yaxis()
    plt.title("Cluster {}".format(cluster_number))
    
    
    
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



def umap_fit(df_scaled, RANDOM_SEED):
    """
    Fits a umap transformer. Returns transformer.
    """
    
    # Instantiate and fit
    reducer = umap.UMAP(random_state=RANDOM_SEED).fit(X=df_scaled)

    # Transform
    embedding = pd.DataFrame(reducer.transform(df_scaled))
    embedding.shape
    
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



def hdbscan_predict(embedding, df_scaled, clusterer):
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



def color_setter(n_clusters, scores):
    """
    Sets color palette and color values per data point.
    """
    
    color_palette = sns.color_palette('deep', n_clusters)

    # Assign color based on index in color_palette (e.g., cluster 0 will be color_palette[0])
    # Else, assign as grey
    cluster_colors = [color_palette[x] if x >= 0
                      else (0.5, 0.5, 0.5) # Grey
                      for x in scores['cluster']]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                             zip(cluster_colors, scores['score'])]

    return({
        "color_palette": color_palette,
        "cluster_colors": cluster_colors,
        "cluster_member_colors": cluster_member_colors
    })



def plot_clusters(embedding, plot_sample_size, cluster_member_colors):
    """
    Plots hdbscan clusters with corresponding colors.
    """
    
    # Sample
    for_plotting = embedding.sample(plot_sample_size)

    # Plot
    fig, ax = plt.subplots()
    fig.set_dpi(120)
    fig.set_size_inches(10,7)
    plot = plt.scatter(for_plotting[0], for_plotting[1], alpha=0.25, c=[cluster_member_colors[i] for i in for_plotting.index])