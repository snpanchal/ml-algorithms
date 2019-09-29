import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')


class MeanShift:
    def __init__(self, radius=4):
        self.radius = radius

    def fit(self, data):
        centroids = {}

        # set all data points as centroids at first
        for i in range(len(data)):
            centroids[i] = data[i]

        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []  # will be populated with feature sets that are in radius
                centroid = centroids[i]
                for feature_set in data:
                    # if a feature set is in centroid's bandwidth, add to in_bandwidth
                    if np.linalg.norm(feature_set - centroid) < self.radius:
                        in_bandwidth.append(feature_set)

                new_centroid = np.average(in_bandwidth, axis=0)  # gives mean vector of all vectors in in_bandwidth
                new_centroids.append(tuple(new_centroid))  # can find unique tuples

            uniques = sorted(list(set(new_centroids)))  # get list of new sorted unique centroids

            prev_centroids = dict(centroids)
            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array((uniques[i]))

            optimized = True
            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                    break
            if optimized:
                break

        self.centroids = centroids

    def predict(self, data):
        pass


X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11],
              [8, 2],
              [10, 2],
              [9, 3]])

colours = 10 * ["g", "r", "c", "b", "k"]

clf = MeanShift()
clf.fit(X)

centroids = clf.centroids

plt.scatter(X[:, 0], X[:, 1], s=150)

for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)

plt.show()
