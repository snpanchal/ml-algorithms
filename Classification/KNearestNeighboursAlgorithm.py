import random
import warnings
from collections import Counter

import numpy as np
import pandas as pd


def k_nearest_neighbours(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to value less than total voting groups')

    distances = []
    for group in data:
        for features in data[group]:
            # euclidean_distance = np.sqrt(np.sum((np.array(features) - np.array(predict))**2))
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]] # find groups for k-nearest neighbours
    vote_result = Counter(votes).most_common(1)[0][0] # find most common group
    confidence = Counter(votes).most_common(1)[0][1] / k # number of most common group / k (number of neighbours)

    # print(vote_result, confidence)

    return vote_result, confidence


accuracies = []
for i in range(25):
    df = pd.read_csv('breast-cancer-wisconsin.data')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    full_data = df.astype(float).values.tolist()
    random.shuffle(full_data)

    test_size = 0.4
    train_set = {2: [], 4: []}
    test_set = {2: [], 4: []}
    train_data = full_data[:-int(test_size * len(full_data))]
    test_data = full_data[-int(test_size * len(full_data)):]

    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbours(train_set, data, k=5)
            if group == vote:
                correct += 1
            total += 1
    # print('Accuracy:', correct / total)
    accuracies.append(correct / total)
print('Average accuracy:', sum(accuracies) / len(accuracies))
