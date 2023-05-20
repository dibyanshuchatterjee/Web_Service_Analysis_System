import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def confirm_k(feature_model):
    """
    Aids in confirming the K value for K-means
    :param feature_model: Extracted features
    :return: None
    """
    vals = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init="k-means++",
                        max_iter=300, n_init=10)
        kmeans.fit(feature_model)
        vals.append(kmeans.inertia_)
    plt.plot(range(1, 11), vals)
    plt.title("Knee Method")
    plt.xlabel("Number of clusters")
    plt.ylabel("Values")
    plt.show()


def compute_and_plot_NN(feature_model):
    """
    Aids in computing the nearest neighbours and their relative distances, which in turn helps evaluate
    hyper-parameters of DBSCAN
    :param feature_model: Extracted features
    :return: None
    """
    # Compute nearest neighbor distances for each point
    nn = NearestNeighbors(n_neighbors=10)
    nn.fit(feature_model)
    distances, indices = nn.kneighbors(feature_model)

    # Sort distances for each point
    sorted_distances = np.sort(distances, axis=0)

    # Plot sorted distances
    plt.plot(sorted_distances)
    plt.xlabel('Points sorted by distance to kth nearest neighbor')
    plt.ylabel('Distance to kth nearest neighbor')
    plt.title('Elbow Plot')
    plt.show()


def perform_clustering_and_evaluate(feature_model, feature_num, k=5, eps=0.0095, min_samples=5):
    """
    Aids in performing clustering and evaluating results
    :param feature_model: Extracted features
    :param feature_num: int representing which feature number is coming in
    :param k: The k value for K-means (int)
    :param eps: The eps parameter for DBSCAN (float)
    :param min_samples: The min_samples parameter for DBSCAN (int)
    :return: None
    """
    confirm_k(feature_model=feature_model)
    compute_and_plot_NN(feature_model=feature_model)
    # Perform clustering using k-means
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(feature_model)

    n_clusters_k_means = len(set(kmeans.labels_)) - (1 if -1 in kmeans.labels_ else 0)

    if n_clusters_k_means >= 2:
        # Evaluate the clustering result using Silhouette score
        silhouette_avg_k_means = silhouette_score(feature_model, kmeans.labels_)
        print(f"Silhouette score for k-means clustering using feature {feature_num} : ", silhouette_avg_k_means)

    else:
        silhouette_avg_k_means = None
        print("K-means clustering resulted in only one cluster. Cannot compute silhouette score.")

    # perform DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(feature_model)

    n_clusters_dbscan = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
    if n_clusters_dbscan >= 2:
        silhouette_avg_dbscan = silhouette_score(feature_model, dbscan.labels_)
        print(f"Silhouette score for DBSCAN clustering using feature {feature_num} : ", silhouette_avg_dbscan)
    else:
        silhouette_avg_dbscan = None
        print("DBSCAN clustering resulted in only one cluster. Cannot compute silhouette score.")

    if silhouette_avg_dbscan is not None and silhouette_avg_k_means is not None and silhouette_avg_dbscan >= silhouette_avg_k_means:
        n_clusters = n_clusters_dbscan
        print("Number of clusters: ", n_clusters)

    elif silhouette_avg_dbscan is not None and silhouette_avg_k_means is not None and silhouette_avg_dbscan <= silhouette_avg_k_means:
        n_clusters = n_clusters_k_means
        print("Number of clusters: ", n_clusters)

    else:
        print(f"Something was not right with feature extraction model {feature_num}")
