import numpy as np
from collections import Counter

data = [
    ["Peak", "Rainy", "High", "Yes"],
    ["Peak", "Clear", "High", "Yes"],
    ["OffPeak", "Clear", "Low", "No"],
    ["OffPeak", "Rainy", "Medium", "No"]
]

X = [row[:-1] for row in data]
y = [row[-1] for row in data]

def encode(data):
    mapping = {}
    encoded = []
    for col in zip(*data):
        unique = list(set(col))
        map_col = {val: i for i, val in enumerate(unique)}
        mapping[len(mapping)] = map_col
        encoded.append([map_col[val] for val in col])
    return np.array(encoded).T, mapping

X_encoded, mapping = encode(X)

def encode_sample(sample):
    return np.array([mapping[i][sample[i]] for i in range(len(sample))])

def knn_predict(X, y, sample, k=3):
    distances = []
    for i in range(len(X)):
        dist = np.linalg.norm(X[i] - sample)
        distances.append((dist, y[i]))

    distances.sort()
    neighbors = [label for _, label in distances[:k]]
    return Counter(neighbors).most_common(1)[0][0]

test = ["Peak", "Rainy", "High"]
sample_encoded = encode_sample(test)

print("Prediction:", knn_predict(X_encoded, y, sample_encoded))