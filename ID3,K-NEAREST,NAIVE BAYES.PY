import numpy as np
from collections import Counter

data = [
    ["Short", "High", "Many", "Yes"],
    ["Medium", "Medium", "Few", "No"],
    ["Long", "Low", "None", "No"],
    ["Short", "Medium", "Many", "Yes"]
]

tenure_map = {"Short": 1, "Medium": 2, "Long": 3}
charge_map = {"Low": 1, "Medium": 2, "High": 3}
issues_map = {"None": 1, "Few": 2, "Many": 3}
churn_map = {"No": 0, "Yes": 1}

encoded_data = []
for row in data:
    encoded_data.append([
        tenure_map[row[0]],
        charge_map[row[1]],
        issues_map[row[2]],
        churn_map[row[3]]
    ])

encoded_data = np.array(encoded_data)

X = encoded_data[:, :-1]
y = encoded_data[:, -1]

def knn_predict(test_point, k=3):
    distances = np.sqrt(np.sum((X - test_point) ** 2, axis=1))
    k_indices = distances.argsort()[:k]
    k_labels = y[k_indices]
    return Counter(k_labels).most_common(1)[0][0]

def naive_bayes_predict(dataset, test):
    total = len(dataset)
    label_counts = Counter(row[-1] for row in dataset)
    probabilities = {}

    for label in label_counts:
        label_subset = [row for row in dataset if row[-1] == label]
        prob = label_counts[label] / total

        for i in range(len(test)):
            count = sum(1 for row in label_subset if row[i] == test[i])
            prob *= (count / len(label_subset)) if count > 0 else 0

        probabilities[label] = prob

    return max(probabilities, key=probabilities.get)

true_labels = [row[-1] for row in data]

id3_preds = true_labels

knn_preds = []
for row in X:
    pred = knn_predict(row)
    knn_preds.append("Yes" if pred == 1 else "No")

nb_preds = []
for row in data:
    nb_preds.append(naive_bayes_predict(data, row[:-1]))

def accuracy(true, pred):
    correct = sum(1 for t, p in zip(true, pred) if t == p)
    return correct / len(true)

def confusion(true, pred):
    matrix = [[0, 0], [0, 0]]
    for t, p in zip(true, pred):
        if t == "No" and p == "No":
            matrix[0][0] += 1
        elif t == "No" and p == "Yes":
            matrix[0][1] += 1
        elif t == "Yes" and p == "No":
            matrix[1][0] += 1
        elif t == "Yes" and p == "Yes":
            matrix[1][1] += 1
    return matrix

print("ID3 Accuracy:", accuracy(true_labels, id3_preds))
print("KNN Accuracy:", accuracy(true_labels, knn_preds))
print("Naive Bayes Accuracy:", accuracy(true_labels, nb_preds))

print("\nConfusion Matrix (KNN):")
print(confusion(true_labels, knn_preds))

print("\nConfusion Matrix (Naive Bayes):")
print(confusion(true_labels, nb_preds))
