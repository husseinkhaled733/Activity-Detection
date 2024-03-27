import os
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from Evaluation import compute_clusters, compute_purity, compute_recall, compute_f1, compute_entropy
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


def readData_means_flattened():
    training_data_means = []
    evaluation_data_means = []
    flattened_training_data = []
    flattened_evaluation_data = []
    training_labels = []
    evaluation_labels = []
    data_directory = "C:\\Users\\DELL\\Downloads\\daily+and+sports+activities\\data"

    # Step 1: Accessing the Data Directory
    activities = os.listdir(data_directory)

    # Step 2: Iterating Through Activity Folders
    for activity in activities:
        activity_path = os.path.join(data_directory, activity)
        activity_number = int(activity.split("a")[1])

        # Assigning labels
        training_labels.extend([activity_number] * 48 * 8)
        evaluation_labels.extend([activity_number] * 12 * 8)

        # Step 3: Iterating Through Subject Folders
        subjects = os.listdir(activity_path)
        for subject in subjects:
            subject_path = os.path.join(activity_path, subject)

            # Step 4: Reading Text Files (Segments)
            segments = os.listdir(subject_path)
            training_segments = segments[:48]  # First 48 segments for training
            evaluation_segments = segments[48:]  # Rest for evaluation

            for segment_file in training_segments:
                segment_file_path = os.path.join(subject_path, segment_file)
                with open(segment_file_path, 'r') as file:
                    segment_data = np.loadtxt(file, delimiter=',')
                    mean_data = np.mean(segment_data, axis=0)  # Taking mean along columns
                    training_data_means.append(mean_data)

                    flattened_data = segment_data.flatten()  # Flattening the segment
                    flattened_training_data.append(flattened_data)

            for segment_file in evaluation_segments:
                segment_file_path = os.path.join(subject_path, segment_file)
                with open(segment_file_path, 'r') as file:
                    segment_data = np.loadtxt(file, delimiter=',')
                    mean_data = np.mean(segment_data, axis=0)  # Taking mean along columns
                    evaluation_data_means.append(mean_data)

                    flattened_data = segment_data.flatten()  # Flattening the segment
                    flattened_evaluation_data.append(flattened_data)

    # Convert the data lists into numpy arrays
    training_data_means = np.array(training_data_means)
    evaluation_data_means = np.array(evaluation_data_means)
    flattened_training_data = np.array(flattened_training_data)
    flattened_evaluation_data = np.array(flattened_evaluation_data)

    # Apply PCA reduction
    PCA_reduction = PCA(n_components=0.9)
    PCA_training_data = PCA_reduction.fit_transform(flattened_training_data)
    PCA_evaluation_data = PCA_reduction.transform(flattened_evaluation_data)

    # Print dimensions
    print("Training Data Means Shape:", training_data_means.shape)
    print("Evaluation Data Means Shape:", evaluation_data_means.shape)
    print("PCA Training Data Shape:", PCA_training_data.shape)
    print("PCA Evaluation Data Shape:", PCA_evaluation_data.shape)
    print("Training Labels Shape:", len(training_labels))
    print("Evaluation Labels Shape:", len(evaluation_labels))

    return training_data_means, evaluation_data_means, PCA_training_data, PCA_evaluation_data, training_labels, evaluation_labels


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


(training_data_means, evaluation_data_means, PCA_training_data, PCA_evaluation_data, training_labels,
 evaluation_labels) = readData_means_flattened()

eps = 2.4

# First Approach using means
clusters, labels = dbscan(evaluation_data_means, eps, 13)
print("Method 1 Clusters at eps = ", eps, ": ", len(clusters))
print("Method 1 Labels at eps = ", eps, ": ", labels)
print("Method 1 Purity = ", compute_purity(compute_clusters(evaluation_labels, labels, len(np.unique(labels))),len(evaluation_data_means)))
print("Method 1 Recall = ", compute_recall(compute_clusters(evaluation_labels, labels, len(np.unique(labels))),len(evaluation_data_means)))
print("Method 1 F1 Score = ", compute_f1(compute_clusters(evaluation_labels, labels, len(np.unique(labels)))))
print("Method 1 Conditional Entropy = ", compute_entropy(compute_clusters(evaluation_labels, labels, len(np.unique(labels))),len(evaluation_data_means)))

# Second Approach using PCA
clusters, labels = dbscan(PCA_evaluation_data, eps, 13)
print("Method 2 Clusters at eps = ", eps, ": ", len(clusters))
print("Method 2 Labels at eps = ", eps, ": ", labels)

print("Method 2 Purity = ", compute_purity(compute_clusters(evaluation_labels, labels, len(np.unique(labels))),len(PCA_evaluation_data)))
print("Method 2 Recall = ", compute_recall(compute_clusters(evaluation_labels, labels, len(np.unique(labels))),len(PCA_evaluation_data)))
print("Method 2 F1 Score = ", compute_f1(compute_clusters(evaluation_labels, labels, len(np.unique(labels)))))
print("Method 2 Conditional Entropy = ", compute_entropy(compute_clusters(evaluation_labels, labels, len(np.unique(labels))),len(PCA_evaluation_data)))
