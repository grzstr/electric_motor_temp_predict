import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from kerastuner.tuners import RandomSearch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

# Load the dataset
file_path = 'electric_motor_temp_predict/fp_all_normalized.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Splitting the data into features and target
X = df.drop('SV_Motor_Temperature', axis=1)
y = df['SV_Motor_Temperature']

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=False)

# Reshape the data for Conv1D input
time_steps = 20  # Number of time steps
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

# Hyperparameter tuning function
def build_model(hp):
    model = Sequential()
    model.add(Conv1D(
        filters=hp.Int('filters', min_value=32, max_value=128, step=32),
        kernel_size=hp.Choice('kernel_size', values=[3, 5]),
        activation='relu',
        input_shape=(time_steps, num_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=hp.Int('units', min_value=50, max_value=200, step=50), activation='relu'))
    model.add(Dense(1))  # Regression problem
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
        loss='mean_squared_error')
    return model

# Initialize the tuner
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,  # Set more trials for a more thorough search
    executions_per_trial=3,
    directory='gsn',  # Path where the tuner data is stored
    project_name='electric_motor_temp_predict')

# Perform hyperparameter search
tuner.search(X_train_seq, y_train_seq, epochs=10, validation_data=(X_test_seq, y_test_seq))

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the best hyperparameters and train it
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train_seq, y_train_seq, epochs=50, validation_data=(X_test_seq, y_test_seq))  # Use more epochs

# Evaluate the model
test_loss = model.evaluate(X_test_seq, y_test_seq)
print(f'Test Loss: {test_loss}')

# Make predictions
y_pred = model.predict(X_test_seq)

# Plot the results
plt.plot(y_test_seq, label='True')
plt.plot(y_pred, label='Predicted')
plt.title('Prediction')
plt.legend()
plt.show()

# Accuracy metrics
print('Mean squared error: %.2f' % mean_squared_error(y_test_seq, y_pred))
print('Mean absolute error: %.2f' % mean_absolute_error(y_test_seq, y_pred))
print('R2 score: %.2f' % r2_score(y_test_seq, y_pred))
print('Explained variance score: %.2f' % explained_variance_score(y_test_seq, y_pred))
