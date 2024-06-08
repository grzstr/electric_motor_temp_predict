import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import keras_tuner as kt

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense


#Wczytanie wszystkich plików z danymi
fp_all_tab = []
for i in range(1,5):
    fp_all_tab.append(np.array(pd.read_csv('fp1/FP1_' + str(i) + '.csv', sep=',',low_memory=False)))
for i in range(1,7):
    fp_all_tab.append(np.array(pd.read_csv('fp2/FP2_' + str(i) + '.csv', sep=',',low_memory=False)))
for i in range(1,7):
    fp_all_tab.append(np.array(pd.read_csv('r/RACE_' + str(i) + '.csv', sep=',',low_memory=False)))

df = pd.read_csv('r/RACE_1.csv', sep=',',low_memory=False)
          
#Złączenie tabel
fp_all = fp_all_tab[0]
for i in range(1, len(fp_all_tab)):
    fp_all = np.concatenate((fp_all, fp_all_tab[i]))

#Dodanie podpisów kolumn
fp_all_df = pd.DataFrame(fp_all)
fp_all_df.columns = df.columns

# Splitting the data into features and target
X = fp_all_df.drop('SV_Motor_Temperature', axis=1)
y = fp_all_df['SV_Motor_Temperature']

# Normalizacja danych
scaler_x = StandardScaler()
X_norm = scaler_x.fit_transform(X)
X_norm_df = pd.DataFrame(X_norm, columns=X.columns)

scaler_y = StandardScaler()
y_norm = scaler_y.fit_transform(y.values.reshape(-1, 1))
y_norm_df = pd.DataFrame(y_norm)

X_norm_rest = X_norm_df.iloc[:5991+4616, :]
X_VALIDATE = X_norm_df.iloc[5991+4613:, :]

y_norm_rest = y_norm_df.iloc[:5991+4616, :]
y_VALIDATE = y_norm_df.iloc[5991+4613:, :]

# Resetuj indeksy dla X_norm_rest i X_VALIDATE
X_norm_rest = X_norm_rest.reset_index(drop=True)
X_VALIDATE = X_VALIDATE.reset_index(drop=True)

# Resetuj indeksy dla y_norm_rest i y_VALIDATE
y_norm_rest = y_norm_rest.reset_index(drop=True)
y_VALIDATE = y_VALIDATE.reset_index(drop=True)

y_VALIDATE_inv = scaler_y.inverse_transform(y_VALIDATE)

#plot x, y
plt.plot( y,color='black')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_norm_rest, y_norm_rest, test_size=0.30, shuffle=False)

# Reshape the data for Conv1D input
# Number of time steps (parametr do eksperymentow)
time_steps = 10
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

# # Defining the Conv1D model
# model = Sequential([
#     Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(time_steps, num_features)),
#     MaxPooling1D(pool_size=2),
#     Flatten(),
#     Dense(50, activation='relu'),
#     Dense(1)  # Regression problem
# ])

#search for the best hyperparameters
import keras_tuner as kt

def model_builder(hp):
    model = Sequential()
    model.add(Conv1D(filters=hp.Int('filters', 32, 256, step=32), kernel_size=hp.Int('kernel_size', 2, 5), activation='relu', input_shape=(time_steps, num_features)))
    model.add(MaxPooling1D(pool_size=hp.Int('pool_size', 2, 3)))
    model.add(Flatten())
    model.add(Dense(units=hp.Int('units', 50, 150, step=50), activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

tuner = kt.Hyperband(model_builder, objective='val_loss', max_epochs=50, factor=3, directory='my_dir', project_name='intro_to_kt') 
tuner.search(X_train_seq, y_train_seq, epochs=50, validation_data=(X_test_seq, y_test_seq))

#print the best hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The hyperparameter search is complete. The optimal number of filters is {best_hps.get('filters')}, kernel size is {best_hps.get('kernel_size')}, pool size is {best_hps.get('pool_size')}, and number of units is {best_hps.get('units')}.
""")
print(best_hps.values)

#build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train_seq, y_train_seq, epochs=50, validation_data=(X_test_seq, y_test_seq))

# Evaluate the model
test_loss = model.evaluate(X_test_seq, y_test_seq)
print(f'Test Loss: {test_loss}')

# Make predictions
y_pred = model.predict(X_test_seq)

# # Plot the results
import matplotlib.pyplot as plt

plt.plot(y_test_seq, label='True')
plt.plot(y_pred, label='Predicted')
plt.title('Prediction')
plt.legend()
plt.show()


# #accuracy
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




#Valitation of the model on different data
# Load the dataset
# file_path = 'race22_normalized.csv'  # Replace with your file path

# df3 = pd.read_csv(file_path3)
# Splitting the data into train and test sets


# # Creating sequences for train and test data
X_val_seq = create_sequences(X_VALIDATE.values, time_steps)
# 

# # Adjusting labels for the sequence data

y_val_seq = y_VALIDATE[time_steps:].values



# #make predictions

y_pred_FP2 = model.predict(X_val_seq)
#acuracy
print('Mean squared error: %.2f'
        % mean_squared_error(y_VALIDATE[time_steps:], y_pred_FP2))
print('Mean absolute error: %.2f'
        % mean_absolute_error(y_VALIDATE[time_steps:], y_pred_FP2))
print('R2 score: %.2f'
        % r2_score(y_VALIDATE[time_steps:], y_pred_FP2))
print('Explained variance score: %.2f'
        % explained_variance_score(y_VALIDATE[time_steps:], y_pred_FP2))

# # Plot the results
import matplotlib.pyplot as plt

plt.plot(y_VALIDATE[time_steps:], label='True')
plt.plot(y_pred_FP2, label='Predicted')
plt.title('Prediction')
plt.legend()
plt.show()


#save fatures and target do dataframe
df4=pd.DataFrame(X_val_seq)
# df4['SV_Motor_Temperature']=y_VALIDATE[time_steps:]
df4['SV_Motor_Temperature']=y_pred_FP2
# df4.to_csv("fp2_123_predicted.csv", index=False)

y_pred_FP2_inv = scaler_y.inverse_transform(y_pred_FP2)


plt.plot(y_VALIDATE_inv[time_steps:], label='True')
plt.plot(y_pred_FP2_inv, label='Predicted')
plt.title('Prediction')
plt.legend()
plt.show()







