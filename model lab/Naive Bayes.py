from collections import Counter

data = [
    ["Peak", "Rainy", "High", "Yes"],
    ["Peak", "Clear", "High", "Yes"],
    ["OffPeak", "Clear", "Low", "No"],
    ["OffPeak", "Rainy", "Medium", "No"]
]

def naive_bayes(data, sample):
    labels = [row[-1] for row in data]
    total = len(data)
    label_counts = Counter(labels)

    probs = {}

    for label in label_counts:
        prior = label_counts[label] / total
        likelihood = 1

        for i in range(len(sample)):
            count = sum(1 for row in data if row[i] == sample[i] and row[-1] == label)
            likelihood *= (count + 1) / (label_counts[label] + len(set([row[i] for row in data])))

        probs[label] = prior * likelihood

    print("Probabilities:", probs)
    return max(probs, key=probs.get)

test = ["Peak", "Rainy", "High"]
print("Prediction:", naive_bayes(data, test))