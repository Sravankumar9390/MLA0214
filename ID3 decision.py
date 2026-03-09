import numpy as np
from collections import Counter

# Encoding categorical values
mapping = {
    "Short": 1, "Medium": 2, "Long": 3,
    "Low": 1, "Medium": 2, "High": 3,
    "None": 1, "Few": 2, "Many": 3,
    "No": 0, "Yes": 1
}

encoded_data = []
for row in data:
    encoded_data.append([mapping[val] for val in row])

encoded_data = np.array(encoded_data)

X = encoded_data[:, :-1]
y = encoded_data[:, -1]

def knn_predict(test_point, k=3):
    distances = np.sqrt(np.sum((X - test_point)**2, axis=1))
    k_indices = distances.argsort()[:k]
    k_labels = y[k_indices]
    return Counter(k_labels).most_common(1)[0][0]

# Example Test Record: Short, High, Many
test = np.array([1, 3, 3])
prediction = knn_predict(test)
print("KNN Prediction:", "Yes" if prediction == 1 else "No")
