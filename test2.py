import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import keras_tuner as kt



# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import GridSearchCV


# Load the dataset
file_path = 'fp_all_normalized.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Splitting the data into features and target
X = df.drop('SV_Motor_Temperature', axis=1)
y = df['SV_Motor_Temperature']


# print(X)
# print(y)

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, shuffle=False)

# Reshape the data for Conv1D input
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
# y_train_seq = create_sequences(y_train.values, time_steps)
X_test_seq = create_sequences(X_test.values, time_steps)
# y_test_seq = create_sequences(y_test.values, time_steps)
y_train_seq = y_train[time_steps:].values
y_test_seq = y_test[time_steps:].values


import keras_tuner as kt
# Search for the best hyperparameters
def model_builder(hp):
    model = Sequential()
    model.add(Conv1D(filters=hp.Int('filters', 32, 256, step=32), kernel_size=hp.Int('kernel_size', 2, 5), activation='relu', input_shape=(time_steps, num_features)))
    model.add(MaxPooling1D(pool_size=hp.Int('pool_size', 2, 3)))
    model.add(Flatten())
    model.add(Dense(units=hp.Int('units', 50, 150, step=50), activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

tuner = kt.Hyperband(model_builder,
                        objective='val_loss',
                        max_epochs=20,
                        factor=3,
                        directory='my_dir',
                        project_name='intro_to_kt')

tuner.search(X_train_seq, y_train_seq, epochs=20, validation_data=(X_test_seq, y_test_seq))

best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

print(best_hyperparameters.values)
print(best_model.summary())

y_pred = best_model.predict(X_test_seq)

# Plot the results
import matplotlib.pyplot as plt

plt.plot(y_test_seq, label='True')
plt.plot(y_pred, label='Predicted')
plt.title('Prediction')
plt.legend()
plt.show()


