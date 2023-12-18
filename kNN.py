import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the data from cirrhosis.csv
data = pd.read_csv('newData.csv', sep=',')

# Split the data into features (X) and target variable (y)
X = data.drop('Status', axis=1)
y = data['Status']

# Perform the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create a kNN classifier
clf = KNeighborsClassifier(n_neighbors=3)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)