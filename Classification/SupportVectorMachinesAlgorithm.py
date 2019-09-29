import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')


class SupportVectorMachine:
    def __init__(self, visualization=True):
        self.visualization = visualization  # visualize data
        self.colours = {1: 'r', -1: 'b'}  # display colours for different classes
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    # train
    def fit(self, data):
        self.data = data
        # {||w||: [w, b]}
        opt_dict = {}
        # use transforms to check vector w in all directions
        transforms = [[1, 1], [-1, 1], [-1, -1], [1, -1]]
        all_data = []
        for yi in self.data:
            for feature_set in self.data[yi]:
                for feature in feature_set:
                    all_data.append(feature)
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        # support vectors yi(xi . w + b) = 1
        # step sizes for finding global minimum in convex problem
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # point of expense:
                      self.max_feature_value * 0.001]

        # extremely expensive
        b_range_multiple = 5
        # we don't need to take as small of steps
        # with b as we do w
        b_multiple = 5

        latest_optimum = self.max_feature_value * 10
        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])

            # can do this because of convex problem
            optimized = False
            while not optimized:
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                                   self.max_feature_value * b_range_multiple,
                                   step * b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True
                        # weakest link in SVM fundamentally
                        # SMO attempts to fix this a bit
                        # yi(xi . w + b) >= 1
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False
                                # print(xi, ':', yi * (np.dot(w_t, xi) + b))

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                if w[0] < 0:
                    optimized = True
                    print('Optimized a step')
                else:
                    # w = [5, 5]
                    # step = 1
                    # w - step = [4, 4]
                    w = w - step

            norms = sorted([n for n in opt_dict])  # sort norms collected in dictionary
            opt_choice = opt_dict[norms[0]]  # norms[0] = smallest norm
            # opt_dict -> ||w||, opt_choice -> [w, b]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step * 2

        for i in self.data:
            for xi in self.data[i]:
                yi = i
                print(xi, ':', yi * (np.dot(self.w, xi) + self.b))

    # prediction using SVM = sign(x . w + b)
    def predict(self, features):
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, c=self.colours[classification], marker='*')

        return classification

    # visualize SVM
    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colours[i]) for x in data_dict[i]] for i in data_dict]

        # hyperplane = x.w + b
        # v = x.w + b
        # + support vector = 1, - support vector = -1, decision boundary = 0
        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        data_range = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
        hyp_x_min = data_range[0]
        hyp_x_max = data_range[1]

        # + support vector hyperplane (w.x + b = 1)
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        # - support vector hyperplane (w.x + b = -1)
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        # decision boundary hyperplane (w.x + b = 0)
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()


data_dict = {
    -1: np.array([[1, 7],
                  [2, 8],
                  [3, 8]]),
    1: np.array([[5, 1],
                 [6, -1],
                 [7, 3]])
}

svm = SupportVectorMachine()
svm.fit(data=data_dict)

predict_data = [[0, 10], [1, 3], [3, 4], [3, 5], [5, 5], [5, 6], [6, -5], [5, 8]]
for p in predict_data:
    svm.predict(p)

svm.visualize()
