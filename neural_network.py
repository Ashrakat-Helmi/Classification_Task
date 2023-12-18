import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

# Load the data from cirrhosis.csv
data = pd.read_csv('newData.csv', sep=',')

# Split the data into features (X) and target variable (y)
X = data.drop('Status', axis=1)
y = data['Status']

# Perform the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create a Sequential model
model = Sequential()

# Add the input layer and hidden layer
model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))

# Add the output layer
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the training data
model.fit(X_train, y_train, epochs=50, verbose=0)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Round the predictions to the nearest integer (0 or 1)
y_pred = [round(p[0]) for p in y_pred]

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)