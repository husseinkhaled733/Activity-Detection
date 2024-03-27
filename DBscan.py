from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def readData():
    aps_eval = []
    for a in range(1, 20):
        p_eval = []
        for p in range(1, 9):
            s_eval = []
            for s in range(49, 61):
                temp = []
                path = "C:\\Users\\DELL\\Downloads\\daily+and+sports+activities\\data\\a"
                path += f'0{a}' if a < 10 else f'{a}'
                path += f'\\p{p}\\s'
                path += f'{s}.txt'
                file = open(path, "r")
                for l in range(125):
                    temp.append(np.array(file.readline().split(','), dtype=float))
                s_eval.append(np.array(temp))
            p_eval.append(np.array(np.array(s_eval)))
        aps_eval.append(np.array(p_eval))
    return np.array(aps_eval)


def getPointsByMeans(data):
    eval_points = []
    for a in range(19):
        for p in range(8):
            for s in range(12):
                eval_points.append(np.mean(data[a][p][s], axis=0))
    return np.array(eval_points)


def getPointsByFlattening(data):
    return PCA(n_components=0.9).fit_transform(data)


def dbscan(dataset, eps, min_pts):
    clusters = []
    visited = set()

    for point_index, point in enumerate(dataset):
        if point_index in visited:
            continue

        visited.add(point_index)
        neighbors = region_query(dataset, point_index, eps)

        if len(neighbors) >= min_pts:
            cluster = []
            expand_cluster(dataset, visited, neighbors, cluster, eps, min_pts)
            clusters.append(cluster)

    return clusters, extract_labels(clusters, len(dataset))


def expand_cluster(dataset, visited, neighbors, cluster, eps, min_pts):
    # cluster.append(neighbors)
    for index in neighbors:
        if index not in visited:
            cluster.append(index)
    queue = deque(neighbors)

    while queue:
        current_point_index = queue.popleft()
        current_point_neighbors = []
        if current_point_index not in visited:
            visited.add(current_point_index)
            current_point_neighbors = region_query(dataset, current_point_index, eps)
            if len(current_point_neighbors) >= min_pts:
                queue.extend(current_point_neighbors)
        for neighbor in current_point_neighbors:
            if neighbor not in cluster:
                cluster.append(neighbor)


def region_query(dataset, query_point_index, eps):
    neighbors = []
    for index, point in enumerate(dataset):
        if np.linalg.norm(point - dataset[query_point_index]) <= eps:
            neighbors.append(index)
    return neighbors


def extract_labels(clusters, n):
    labels = np.zeros(n, dtype=int) - 1

    for i, cluster in enumerate(clusters):
        for point_index in cluster:
            labels[point_index] = i

    return labels


data_set = getPointsByMeans(readData())

# eps_values = np.arange(1, 20, step=0.5)
# maxC = 0
# e = 0
# for eps in eps_values:
#     clusters, labels = dbscan(data_set, eps, 10)
#     print("Clusters at eps = ", eps, ": ", len(clusters))
#     print("Labels at eps = ", eps, ": ", labels)
#     if(len(clusters) > maxC):
#         maxC = len(clusters)
#         e = eps
# print("best epsilon = ",e)
# print("with number of cluser = ",maxC)
eps = 5
clusters, labels = dbscan(data_set, eps, 10)
print("Clusters at eps = ", eps, ": ", len(clusters))
print("Labels at eps = ", eps, ": ", labels)