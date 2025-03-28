""" 
Shreeya Sampat
Lab 3
OMSBA 5067 - Machine Learning
"""
import numpy as np

def calculate_distance(instance1, instance2, distance):
    if distance == 1:
        return np.sum(np.abs(instance1 - instance2))
    elif distance == 2:
        return np.sqrt(np.sum(np.square(instance1 - instance2)))
    elif distance == 3:
        return np.max(np.abs(instance1 - instance2))

def myKNN(trainX, trainY, testX, distance, K):
    predictions = []
    for test_instance in testX:
        distances = []
        for train_instance in trainX:
            dist = calculate_distance(train_instance, test_instance, distance)
            distances.append(dist)
        sorted_indices = np.argsort(distances)
        k_nearest_neighbors = sorted_indices[:K]
        k_nearest_labels = trainY[k_nearest_neighbors]
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        predicted_label = unique_labels[np.argmax(counts)]
        predictions.append(predicted_label)
    return predictions

# Toy dataset
trainX = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [1, 0, 1, 0], [1, 1, 1, 1], [0, 0, 0, 1],
                   [0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1], [0, 1, 0, 0], [1, 0, 0, 0],
                   [1, 0, 1, 1], [1, 0, 1, 0], [1, 1, 0, 1], [0, 1, 1, 0], [0, 0, 0, 1],
                   [1, 1, 1, 1], [0, 1, 1, 1], [1, 0, 1, 1], [0, 1, 0, 1]])
trainY = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0])
testX = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 1]])
testY = np.array([1, 3, 2, 3, 3])  # True classes for test data

distances = [1, 2, 3]
K_values = [1, 3]

print("Test Data point\tTrue Class\tL1 and K=1\tL1 and K=3\tL2 and K=1\tL2 and K=3\tL∞ and K=1\tL∞ and K=3")
for i in range(len(testX)):
    true_class = testY[i]
    predictions = []
    for distance in distances:
        for K in K_values:
            pred = myKNN(trainX, trainY, testX[i:i+1], distance, K)[0]
            predictions.append(pred)
    print(f"{i+20}\t\t{true_class}\t\t{predictions[0]}\t\t{predictions[1]}\t\t{predictions[2]}\t\t{predictions[3]}\t\t{predictions[4]}\t\t{predictions[5]}")
    
    