import math
from collections import Counter

data = [
    ["Peak", "Rainy", "High", "Yes"],
    ["Peak", "Clear", "High", "Yes"],
    ["OffPeak", "Clear", "Low", "No"],
    ["OffPeak", "Rainy", "Medium", "No"]
]

features = ["TimeOfDay", "Weather", "VehicleCount"]

def entropy(data):
    labels = [row[-1] for row in data]
    total = len(labels)
    counts = Counter(labels)
    ent = 0
    for count in counts.values():
        p = count / total
        ent -= p * math.log2(p)
    return ent

def information_gain(data, index):
    total_entropy = entropy(data)
    values = set([row[index] for row in data])
    weighted_entropy = 0

    for value in values:
        subset = [row for row in data if row[index] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset)

    return total_entropy - weighted_entropy

def id3(data, features):
    labels = [row[-1] for row in data]

    if labels.count(labels[0]) == len(labels):
        return labels[0]

    if len(features) == 0:
        return Counter(labels).most_common(1)[0][0]

    gains = [information_gain(data, i) for i in range(len(features))]
    best_index = gains.index(max(gains))
    best_feature = features[best_index]

    tree = {best_feature: {}}
    values = set([row[best_index] for row in data])

    for value in values:
        subset = [row[:best_index] + row[best_index+1:] for row in data if row[best_index] == value]
        sub_features = features[:best_index] + features[best_index+1:]
        tree[best_feature][value] = id3(subset, sub_features)

    return tree

def predict(tree, features, sample):
    if not isinstance(tree, dict):
        return tree

    root = list(tree.keys())[0]
    index = features.index(root)
    value = sample[index]

    subtree = tree[root].get(value)
    if subtree is None:
        return "Unknown"

    return predict(subtree, features, sample)

tree = id3(data, features)
print("Decision Tree:", tree)

test = ["Peak", "Rainy", "High"]
print("Prediction:", predict(tree, features, test))