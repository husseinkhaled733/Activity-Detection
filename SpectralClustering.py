import numpy as np
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def readData():
    aps_eval = []
    for a in range(1, 20):
        p_eval = []
        for p in range(1, 9):
            s_eval = []
            for s in range(49, 61):
                temp = []
                path = "data\\a"
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


def getSimilarityMatrix(points, gamma):
    return rbf_kernel(points, points, gamma)


def getEigenAttributes(matrix, k):
    eigen_values, eigen_vectors = np.linalg.eig(matrix)
    idx = eigen_values.argsort()[::-1]
    eigen_vectors = np.flip(np.array(eigen_vectors.transpose()[idx, :]))
    return np.flip(np.array(eigen_vectors[:k]).transpose())


sim = np.array([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]])
sim_mat = getSimilarityMatrix(getPointsByMeans(readData()), 0.00001)
for i in range(len(sim)):
    x = np.sum(sim[i])
    sim[i][i] += -x
    sim[i] *= -1
    sim[i] /= x
    sim[i][sim[i] == -0] = 0
u_mat = np.flip(np.real(getEigenAttributes(sim, 2)))
print(u_mat)
# for i in range(len(u_mat)):
#     vec_sum = np.sqrt(np.sum(u_mat[i] ** 2))
#     u_mat[i] /= vec_sum
# kmeans = KMeans(n_clusters=2)
# kmeans.fit(u_mat)
# print(metrics.silhouette_score(u_mat, kmeans.labels_))
# labels_file = open("labels.txt", "w")
# for i in range(19):
#     for j in range(8):
#         for q in range(12):
#             labels_file.write(str(kmeans.labels_[i*96+j*12+q]))
#             labels_file.write(" ")
#         labels_file.write('\n')
#     labels_file.write('\n')
# labels_file.close()
