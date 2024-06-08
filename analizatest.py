import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
# file_path = 'electric_motor_temp_predict/r\RACE_3.csv'
file_path = '400Hz_03-R1-LEM-01.csv'
data = pd.read_csv(file_path)

# # Display the first few rows and basic information about the data
print(data.head())
print(data.info())
# print(data.describe())

# # Check for missing values
missing_values = data.isnull().sum()
print(missing_values[missing_values > 0])

# # Remove columns where all values are zero
data_cleaned = data.loc[:, (data != 0).any(axis=0)]

# # Check which columns were removed
removed_columns = set(data.columns) - set(data_cleaned.columns)
print(removed_columns)
print(data_cleaned.head())

# # Filter peaks in voltage and current data
data_cleaned['SV_Capacitor_Voltage'] = data_cleaned['Pack_Inst_Voltage42'].apply(lambda x: 0 if x > 120 else x)
data_cleaned['SV_Battery_Current'] = data_cleaned['SV_Battery_Current'].apply(lambda x: 0 if x > 500 else x)
data_cleaned['Power2'] = data_cleaned['Pack_Inst_Voltage42'] * data_cleaned['SV_Battery_Current']

# Calculate power based on torque and RPM
data_cleaned['Power3'] = data_cleaned['SV_Torque_value'] * data_cleaned['SV_RPM'] / 9.5488

# Plot the power calculations
plt.figure(figsize=(12, 8))
plt.plot(data_cleaned['Power2'], marker='o', linestyle='-', markersize=2)
plt.plot(data_cleaned['Power3'], marker='o', linestyle='-', markersize=2)
plt.title('Power')
plt.xlabel('Time')
plt.ylabel('Power')
plt.legend(['PACK_VOLTAGE*PACK_CURRENT', 'SV_TORQUE*SV_RPM/9.5488'])
# plt.show()

# Create a new DataFrame to store energy calculations
a = pd.DataFrame()

# Calculate energy consumption for Power2 and Power3 in kWh
time_interval = 0.0025  # time interval is 0.16 seconds
a['Energy_Power2_kWh'] = (data_cleaned['Power2'] * time_interval) / 3600  # converting to kWh
a['Energy_Power3_kWh'] = (data_cleaned['Power3'] * time_interval) / 3600  # converting to kWh

# Sum the energy for the entire race duration
total_energy_power2_kWh = a['Energy_Power2_kWh'].sum()
total_energy_power3_kWh = a['Energy_Power3_kWh'].sum()

print("Total Energy Power2 (kWh):", total_energy_power2_kWh/1000)
print("Total Energy Power3 (kWh):", total_energy_power3_kWh/1000)

# Plot energy calculations
plt.figure(figsize=(12, 8))
plt.plot(a['Energy_Power2_kWh'], marker='o', linestyle='-', markersize=2)
plt.plot(a['Energy_Power3_kWh'], marker='o', linestyle='-', markersize=2)
plt.title('Energy')
plt.xlabel('Time')
plt.ylabel('Energy (kWh)')
plt.legend(['PACK_VOLTAGE*PACK_CURRENT', 'SV_TORQUE*SV_RPM/9.5488'])
# plt.show()



#calculate efficiency based on energy
data_cleaned['Efficiency_Energy'] = a['Energy_Power3_kWh'] / a['Energy_Power2_kWh']*100

#filter efficiency
data_cleaned['Efficiency_Energy'] = data_cleaned['Efficiency_Energy'].apply(lambda x: 0 if x>100 else x)

plt.figure(figsize=(12, 8))

plt.plot(data_cleaned['Efficiency_Energy'], marker='o', linestyle='-', markersize=2)
plt.plot(data_cleaned['Power2']/1000, marker='o', linestyle='-', markersize=2)
plt.plot(data_cleaned['Power3']/1000, marker='o', linestyle='-', markersize=2)
plt.title('Efficiency')
plt.xlabel('Time')
plt.legend(['Efficiency', 'Power_BAT', 'Power_MOT'])
plt.ylabel('Efficiency/Power')    
plt.show()


# #print current and voltage
# plt.figure(figsize=(12, 8))
# plt.plot(data['Pack_Current41'], marker='o', linestyle='-', markersize=2)
# # plt.plot(data['SV_Capacitor_Voltage'], marker='o', linestyle='-', markersize=2)
# # plt.plot(data['Pack_Inst_Voltage42'], marker='o', linestyle='-', markersize=2)
# plt.title('Current and Voltage')
# plt.xlabel('Time')
# plt.ylabel('Current and Voltage')
# plt.legend(['Current', 'Voltage'])
# plt.show()
