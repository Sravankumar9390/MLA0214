import numpy as np
from sklearn.linear_model import LogisticRegression

data = [
    ["Peak", "Rainy", "High", "Yes"],
    ["Peak", "Clear", "High", "Yes"],
    ["OffPeak", "Clear", "Low", "No"],
    ["OffPeak", "Rainy", "Medium", "No"]
]

X = [row[:-1] for row in data]
y = [1 if row[-1] == "Yes" else 0 for row in data]

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

model = LogisticRegression()
model.fit(X_encoded, y)

test = ["Peak", "Rainy", "High"]
sample_encoded = encode_sample(test).reshape(1, -1)

prob = model.predict_proba(sample_encoded)[0][1]
pred = model.predict(sample_encoded)[0]

print("Probability:", prob)
print("Prediction:", "Yes" if pred == 1 else "No")