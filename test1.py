import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Load the dataset
file_path = 'electric_motor_temp_predict/fp_all_normalized.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Splitting the data into features and target
X = df.drop('SV_Motor_Temperature', axis=1)
y = df['SV_Motor_Temperature']

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=False)

# Reshape the data for Conv1D input
# Number of time steps (parametr do eksperymentow)
time_steps = 20
num_features = X.shape[1]

# Function to create sequences
def create_sequences(data, time_steps):
    sequences = []
    for i in range(len(data) - time_steps):
        sequence = data[i:(i + time_steps)]
        sequences.append(sequence)
    return np.array(sequences)

# Creating sequences for train and test data
X_train_seq = create_sequences(X_train.values, time_steps)
X_test_seq = create_sequences(X_test.values, time_steps)

# Adjusting labels for the sequence data
y_train_seq = y_train[time_steps:].values
y_test_seq = y_test[time_steps:].values

# Defining the Conv1D model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(time_steps, num_features)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1)  # Regression problem
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, validation_data=(X_test_seq, y_test_seq))

# Evaluate the model
test_loss = model.evaluate(X_test_seq, y_test_seq)
print(f'Test Loss: {test_loss}')

# Make predictions
y_pred = model.predict(X_test_seq)

# Plot the results
import matplotlib.pyplot as plt

plt.plot(y_test_seq, label='True')
plt.plot(y_pred, label='Predicted')
plt.title('Prediction')
plt.legend()
plt.show()


#accuracy
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

print('Mean squared error: %.2f'
        % mean_squared_error(y_test_seq, y_pred))
print('Mean absolute error: %.2f'
        % mean_absolute_error(y_test_seq, y_pred))
print('R2 score: %.2f'
        % r2_score(y_test_seq, y_pred))
print('Explained variance score: %.2f'
        % explained_variance_score(y_test_seq, y_pred))

