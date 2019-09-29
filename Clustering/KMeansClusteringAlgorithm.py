import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')


class KMeans:
    # k = number of clusters
    # tol: how much centroid is allowed to move
    # max_iter: max number of iterations
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}

        # first centroids will be the first k data points
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for j in range(self.k):
                self.classifications[j] = []

            for feature_set in data:
                # collect distances between each feature set and centroid
                distances = [np.linalg.norm(feature_set - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(feature_set)

            prev_centroids = dict(self.centroids)

            # redefine centroid for next iteration
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            # check if difference between previous and current centroids is within tolerance
            optimized = True
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                    optimized = False

            # if optimized, keep on running iterations, otherwise stop
            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])

# plt.scatter(X[:, 0], X[:, 1], s=150)
# plt.show()

colours = 10 * ['g', 'r', 'c', 'b', 'k']

clf = KMeans()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker='o', color='k', s=150, linewidths=5)

for classification in clf.classifications:
    colour = colours[classification]
    for feature_set in clf.classifications[classification]:
        plt.scatter(feature_set[0], feature_set[1], marker='x', color=colour, s=150, linewidths=5)

unknowns = np.array([[1, 3],
                     [8, 9],
                     [0, 3],
                     [5, 4],
                     [6, 4]])

for unknown in unknowns:
    classification = clf.predict(unknown)
    plt.scatter(unknown[0], unknown[1], marker='*', color=colours[classification], s=150, linewidths=5)

plt.show()
