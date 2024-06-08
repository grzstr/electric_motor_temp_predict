import pandas as pd

# Wczytanie danych z pliku CSV
# file_path = 'electric_motor_temp_predict/fp_all.csv'
file_path = 'electric_motor_temp_predict/r\RACE_3.csv'
data = pd.read_csv(file_path)

# Wyświetlenie pierwszych kilku wierszy danych oraz podstawowych informacji o danych
data.head(), data.info(), data.describe()


# Sprawdzenie brakujących wartości
missing_values = data.isnull().sum()

# Wyświetlenie kolumn z brakującymi wartościami
missing_values[missing_values > 0]

# Usunięcie kolumn, w których wszystkie wartości są wyzerowane
data_cleaned = data.loc[:, (data != 0).any(axis=0)]

# Sprawdzenie, które kolumny zostały usunięte
removed_columns = set(data.columns) - set(data_cleaned.columns)

removed_columns, data_cleaned.head()

data = data_cleaned


import matplotlib.pyplot as plt
import seaborn as sns

# # Ustawienia wykresów
# sns.set(style="whitegrid")

# # Wykres rozkładu prędkośc
# plt.figure(figsize=(10, 6))
# sns.histplot(data['V_Sat'], kde=True)
# plt.title('Rozkład prędkości')
# plt.xlabel('Prędkość (km/h)')
# plt.ylabel('Częstość')
# plt.show()

# # Korelacja między zmiennymi
# correlation_matrix = data_cleaned.corr()

# # Wykres macierzy korelacji
# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation_matrix, cmap='plasma', annot=False, fmt='.2f')
# plt.title('Macierz korelacji')
# plt.show()




# # Podstawowe statystyki dla zmiennej przechyłów
# banking_stats = data_cleaned['Banking_GPS'].describe()

# # Wykres rozkładu przechyłów
# plt.figure(figsize=(10, 6))
# sns.histplot(data_cleaned['Banking_GPS'], kde=True)
# plt.title('Rozkład przechyłów motocykla')
# plt.xlabel('Przechył (stopnie)')
# plt.ylabel('Częstość')
# plt.show()

# banking_stats


# # Przykładowe pary zmiennych do wizualizacji
# pairs_to_visualize_banking = [
#     ('Banking_GPS', 'V_Sat'),  
#     ('Banking_GPS', 'Dist'),
#     ('Banking_GPS', 'Lat_dez'),
#     ('Banking_GPS', 'Lon_dez')
# ]

# # Wizualizacja wykresów punktowych
# for x, y in pairs_to_visualize_banking:
#     plt.figure(figsize=(10, 6))
#     sns.scatterplot(data=data_cleaned, x=x, y=y)
#     plt.title(f'Korelacja między {x} a {y}')
#     plt.xlabel(x)
#     plt.ylabel(y)
#     plt.show()


# plt.figure(figsize=(12, 8))
# plt.plot(data_cleaned['Lon_dez'], data_cleaned['Lat_dez'], marker='o', linestyle='-', markersize=2)
# plt.title('Trasa motocykla')
# plt.xlabel('Długość geograficzna')
# plt.ylabel('Szerokość geograficzna')
# plt.show()



#calculate power based on sv_voltage and SV_current

#filter peaks   
# data_cleaned['SV_Capacitor_Voltage'] = data_cleaned['SV_Capacitor_Voltage'].apply(lambda x: 0 if x>120 else x)
# data_cleaned['SV_Battery_Current'] = data_cleaned['SV_Battery_Current'].apply(lambda x: 0 if x>600 else x)
data_cleaned['Pack_Inst_Voltage42'] = data_cleaned['Pack_Inst_Voltage42'].apply(lambda x: 0 if x>120 else x)
data_cleaned['Pack_Current41'] = data_cleaned['Pack_Current41'].apply(lambda x: 0 if x>500 else x)
data_cleaned['Power2'] = data_cleaned['Pack_Inst_Voltage42'] * data_cleaned['Pack_Current41']

#calculate power based on torque and and angular velocity of motor (RPM) result in kW (1 kW = 9.5488 Nm/s)

#if rpm >8000 then then rpm =0
# data_cleaned['SV_RPM'] = data_cleaned['SV_RPM'].apply(lambda x: 0 if x>8000 else x)

data_cleaned['Power3'] = data_cleaned['SV_Torque_value'] * data_cleaned['SV_RPM'] / 9.5488

#plot results

plt.figure(figsize=(12, 8))
plt.plot(data_cleaned['Power2'], marker='o', linestyle='-', markersize=2)
plt.plot(data_cleaned['Power3'], marker='o', linestyle='-', markersize=2)
plt.title('Power')
plt.xlabel('Time')
plt.ylabel('Power')
plt.legend(['PACK_VOLTAGE*PACK_CURRENT', 'SV_TORQUE*SV_RPM/9.5488'])
plt.show()





# #calculate efficiency based on power2 and power3
# data_cleaned['Efficiency'] = data_cleaned['Power3'] / data_cleaned['Power2']*100

# #filter efficiency
# data_cleaned['Efficiency'] = data_cleaned['Efficiency'].apply(lambda x: 0 if x>100 else x)

# plt.figure(figsize=(12, 8))
# plt.plot(data_cleaned['Efficiency'], marker='o', linestyle='-', markersize=2)
# plt.title('Efficiency')
# plt.xlabel('Time')
# plt.ylabel('Efficiency')
# plt.show()

#create new dataframe a
a = pd.DataFrame()


# Calculate energy consumption for Power2 and Power3 in kWh
time_interval = 0.16  # time interval is 0.16 seconds
a['Ene_Power2_kWh'] = (data_cleaned['Power2'] * time_interval)    # converting to kWh
a['Energy_Power3_kWh'] = (data_cleaned['Power3'] * time_interval)   # converting to kWh

# # Sum the energy for the entire race duration
total_energy_power2_kWh = a['Energy_Power2_kWh'].sum()
total_energy_power3_kWh = a['Energy_Power3_kWh'].sum()

print(total_energy_power2_kWh, total_energy_power3_kWh)


# #plot energy power
plt.figure(figsize=(12, 8))
plt.plot(data_cleaned['Energy_Power2_kWh'], marker='o', linestyle='-', markersize=2)
plt.plot(data_cleaned['Energy_Power3_kWh'], marker='o', linestyle='-', markersize=2)
plt.title('Energy')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend(['PACK_VOLTAGE*PACK_CURRENT', 'SV_TORQUE*SV_RPM/9.5488'])
plt.show()



#calculate efficiency based on energy
# a['Efficiency_Energy'] = a['Energy_Power3_kWh'] / a['Energy_Power2_kWh']*100

# #filter efficiency
# a['Efficiency_Energy'] = a['Efficiency_Energy'].apply(lambda x: 0 if x>100 else x)

# plt.figure(figsize=(12, 8))

# plt.plot(a['Efficiency_Energy'], marker='o', linestyle='-', markersize=2)
# plt.title('Efficiency')
# plt.xlabel('Time')
# plt.ylabel('Efficiency')    
# plt.show()

#calculate average efficiency
# average_efficiency = a['Efficiency_Energy'].mean()
# print(average_efficiency)