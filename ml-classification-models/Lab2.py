from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Load the dataset
data = genfromtxt('transfusion.csv', delimiter=',', skip_header=1)

# Split features (first four columns) and target (last column)
X = data[:, :-1]
y = data[:, -1]

# Split the data into training and testing sets (80% train, 20% test)
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(trainX, trainY)

# Test and error measurement
estimatedY = gnb.predict(testX)
misclassification_rate = (testY != estimatedY).mean()

print("Misclassification Rate:", misclassification_rate)


