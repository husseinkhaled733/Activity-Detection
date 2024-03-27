import math
from collections import Counter


def compute_clusters(ground_truth, points_clusters, r):
    """
    :param ground_truth: aka labels, classes, or targets.
    :param points_clusters: points_clusters[point] -> the cluster where the point is inside.
    :param r: number of clusters.
    :return: clusters[Cluster Number] -> List contains the classes of the points in this cluster.
    """
    clusters = [[] for _ in range(r)]
    for point, point_cluster in enumerate(points_clusters):
        clusters[point_cluster].append(ground_truth[point])
    return clusters


def compute_cluster_purity(cluster):
    """
    :param cluster: list of the classes inside this cluster.
    :return: the purity of this cluster.
    """
    ni = len(cluster)  # number of classes in cluster i
    if ni == 0:  # check if cluster is empty
        return 0
    return max(Counter(cluster).values()) / ni


def compute_purity(clusters, n):
    """
    :param clusters: clusters[Cluster Number] -> List contains the classes of the points in this cluster.
    :param n: number of points.
    :return: the total purity.
    """
    r = len(clusters)  # number of clusters
    purity = 0  # total purity
    for i in range(r):
        ni = len(clusters[i])  # number of classes in cluster i
        purity += ni * compute_cluster_purity(clusters[i])
    return purity / n


def compute_cluster_recall(clusters, i):
    """
    :param clusters: clusters[Cluster Number] -> List contains the classes of the points in this cluster.
    :param i: the cluster number for which we want to calculate the recall.
    :return: the recall of this cluster.
    """
    if not clusters[i]:  # Check if the cluster is empty
        return 0  # Return 0 or any value that suits your logic

    target = max(clusters[i], key=clusters[i].count)  # The class with the maximum occurrences in the cluster
    target_cnt = 0  # Total number of occurrences of the target class across all clusters
    for cluster in clusters:
        for label in cluster:
            if label == target:
                target_cnt += 1
    return max(Counter(clusters[i]).values()) / target_cnt


def compute_recall(clusters, n):
    """
    :param clusters: clusters[Cluster Number] -> List contains the classes of the points in this cluster.
    :param n: number of points.
    :return: the total recall.
    """
    r = len(clusters)  # number of clusters
    recall = 0  # total recall
    for i in range(r):
        ni = len(clusters[i])  # number of classes in cluster i
        recall += ni * compute_cluster_recall(clusters, i)
    return recall / n


def compute_f1(clusters):
    """
    :param clusters: clusters[Cluster Number] -> List contains the classes of the points in this cluster.
    :return: the F1 score of the clustering.
    """
    r = len(clusters)  # number of clusters
    f1 = 0  # F1 score
    for i in range(r):
        purity_i = compute_cluster_purity(clusters[i])  # purity of cluster i
        recall_i = compute_cluster_recall(clusters, i)  # recall of cluster i
        if purity_i + recall_i != 0:  # Add this check to avoid division by zero
            f1 += 2 * purity_i * recall_i / (purity_i + recall_i)
    return f1 / r


def compute_cluster_entropy(cluster):
    """
    :param cluster: list of the classes inside this cluster.
    :return: h(T | Ci).
    """
    ni = len(cluster)  # number of classes in cluster i
    cluster_entropy = 0  # entropy of T w.r.t Ci
    for label in set(cluster):
        nij = cluster.count(label)  # label count in cluster
        cluster_entropy += (nij / ni) * math.log2(nij / ni)
    return -cluster_entropy


def compute_entropy(clusters, n):
    """
    :param clusters: clusters[Cluster Number] -> List contains the classes of the points in this cluster.
    :param n: number of points.
    :return: h(T | C).
    """
    r = len(clusters)  # number of clusters
    entropy = 0  # entropy of T w.r.t C
    for i in range(r):
        ni = len(clusters[i])  # number of classes in cluster i
        entropy += ni * compute_cluster_entropy(clusters[i])
    return entropy / n
