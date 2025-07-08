import csv
import random

from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


model = Perceptron()
# model = GaussianNB()
# model = KNeighborsClassifier(n_neighbors=3)

# Read data in from file
with open("assets/banknote.csv") as f:
    reader = csv.reader(f)
    next(reader)

    data = []
    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": "Authentic" if row[4] == "0" else "Counterfeit" 
        })

# Seperate data into training and testing groups
evidence = [row["evidence"] for row in data]
labels = [row["label"] for row in data]

X_training, X_testing, y_training, y_testing = train_test_split(
    evidence, labels, test_size = 0.5
)

# Fit model
model.fit(X_training, y_training)

# Make predictions on the testing set
predictions = model.predict(X_testing)

# Compute how well we performed
correct = (y_testing == predictions).sum()
incorrect = (y_testing != predictions).sum()
total = len(predictions)
accuracy = correct / total * 100

# Print results
print(f"Total predictions: {total}")
print(f"Correct predictions: {correct}")
print(f"Incorrect predictions: {incorrect}")
print(f"Accuracy: {accuracy:.2f}%")