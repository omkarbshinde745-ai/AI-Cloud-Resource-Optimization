from sklearn.cluster import KMeans

def classify(df):
    features = df[['cpu_utilization', 'memory_utilization', 'disk_usage', 'network_usage']]

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(features)

    # Sort clusters by CPU average
    cluster_means = df.groupby('cluster')['cpu_utilization'].mean().sort_values()

    mapping = {
        cluster_means.index[0]: 'LOW',
        cluster_means.index[1]: 'OPTIMAL',
        cluster_means.index[2]: 'HIGH'
    }

    df['usage_type'] = df['cluster'].map(mapping)

    return df