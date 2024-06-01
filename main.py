#open csv file 
import csv
import pandas as pd
# invalid start byte
df = pd.read_csv("test2.csv")

print(df.head())

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Sample data: Replace this with your own time series data.
# X_train: Training data
# y_train: Training labels
# X_test: Testing data
# y_test: Testing labels

# Define the model
model = keras.Sequential([
    layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_shape, num_features)),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy * 100}%')
