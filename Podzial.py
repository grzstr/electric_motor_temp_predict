#Podział datasetu na poszczegole okrążenia

import pandas as pd
import numpy as np
import csv


# Wczytanie danych
fp1 = pd.read_csv('electric_motor_temp_predict/03-FreePractice1-LEM-01.csv', sep=',',low_memory=False)
fp2 = pd.read_csv('electric_motor_temp_predict/04-FreePractice2-LEM-01.csv', sep=',',low_memory=False)
race = pd.read_csv('electric_motor_temp_predict\wyscig.csv',sep=',',low_memory=False)

print(fp1.shape)
print(fp2.shape)
print(race.shape)

# Podział na okrążenia
# FP1

sekund_od_startu1 = [360, 527, 709, 904, 1098]
#1 okrążenie od 6 minuty do 8 minuty 47 sekundy
fp1_1 = fp1[fp1['Time'] >= sekund_od_startu1[0]]
fp1_1 = fp1_1[fp1_1['Time'] <= sekund_od_startu1[1]]

#2 okrążenie od 8 minuty 47 sekund do 11 minuty 49 sekund
fp1_2 = fp1[fp1['Time'] >= sekund_od_startu1[1]]
fp1_2 = fp1_2[fp1_2['Time'] <= sekund_od_startu1[2]]

#3 okrążenie od 11 minuty 49 sekund do 14 minuty 24 sekund
fp1_3 = fp1[fp1['Time'] >= sekund_od_startu1[2]]
fp1_3 = fp1_3[fp1_3['Time'] <= sekund_od_startu1[3]]

#4 okrążenie od 15 minuty 4 sekund do 18 minuty 18 sekund
fp1_4 = fp1[fp1['Time'] >= sekund_od_startu1[3]]
fp1_4 = fp1_4[fp1_4['Time'] <= sekund_od_startu1[4]]

#poworwanie predkosci dla kazdego okrazenia na wspolnym wykresie
import matplotlib.pyplot as plt

# plt.plot(fp1_1['Time'],fp1_1['V_Sat'],label='1 okrążenie')
# plt.plot(fp1_2['Time'],fp1_2['V_Sat'],label='2 okrążenie')
# plt.plot(fp1_3['Time'],fp1_3['V_Sat'],label='3 okrążenie')
# plt.plot(fp1_4['Time'],fp1_4['V_Sat'],label='4 okrążenie')
# plt.legend()
# plt.xlabel('Czas [s]')
# plt.ylabel('Prędkość [km/h]')
# plt.title('Prędkość pojazdu w czasie')
# plt.show()

#podział na pliki
fp1_1.to_csv('electric_motor_temp_predict/fp1/FP1_1.csv',index=False)
fp1_2.to_csv('electric_motor_temp_predict/fp1/FP1_2.csv',index=False)
fp1_3.to_csv('electric_motor_temp_predict/fp1/FP1_3.csv',index=False)
fp1_4.to_csv('electric_motor_temp_predict/fp1/FP1_4.csv',index=False)

# FP2
sekund_od_startu2 = [400, 554, 712, 872, 1034, 1195, 1358, 1520]

#1 okrążenie od 6 minuty 40 sekund do 9 minuty 14 sekundy
fp2_1 = fp2[fp2['Time'] >= sekund_od_startu2[0]]
fp2_1 = fp2_1[fp2_1['Time'] <= sekund_od_startu2[1]]

#2 okrążenie od 9 minuty 14 sekund do 11 minuty 52 sekund
fp2_2 = fp2[fp2['Time'] >= sekund_od_startu2[1]]
fp2_2 = fp2_2[fp2_2['Time'] <= sekund_od_startu2[2]]

#3 okrążenie od 11 minuty 52 sekund do 14 minuty 32 sekund
fp2_3 = fp2[fp2['Time'] >= sekund_od_startu2[2]]
fp2_3 = fp2_3[fp2_3['Time'] <= sekund_od_startu2[3]]

#4 okrążenie od 14 minuty 32 sekund do 17 minuty 14 sekund
fp2_4 = fp2[fp2['Time'] >= sekund_od_startu2[3]]
fp2_4 = fp2_4[fp2_4['Time'] <= sekund_od_startu2[4]]

#5 okrążenie od 17 minuty 14 sekund do 19 minuty 55 sekund
fp2_5 = fp2[fp2['Time'] >= sekund_od_startu2[4]]
fp2_5 = fp2_5[fp2_5['Time'] <= sekund_od_startu2[5]]

#6 okrążenie od 19 minuty 55 sekund do 22 minuty 38 sekund
fp2_6 = fp2[fp2['Time'] >= sekund_od_startu2[5]]
fp2_6 = fp2_6[fp2_6['Time'] <= sekund_od_startu2[6]]

#7 okrążenie od 22 minuty 38 sekund do 25 minuty 20 sekund
fp2_7 = fp2[fp2['Time'] >= sekund_od_startu2[6]]
fp2_7 = fp2_7[fp2_7['Time'] <= sekund_od_startu2[7]]

#poworwanie predkosci dla kazdego okrazenia na wspolnym wykresie
# plt.plot(fp2_1['Time'],fp2_1['V_Sat'],label='1 okrążenie')
# plt.plot(fp2_2['Time'],fp2_2['V_Sat'],label='2 okrążenie')
# plt.plot(fp2_3['Time'],fp2_3['V_Sat'],label='3 okrążenie')
# plt.plot(fp2_4['Time'],fp2_4['V_Sat'],label='4 okrążenie')
# plt.plot(fp2_5['Time'],fp2_5['V_Sat'],label='5 okrążenie')
# plt.plot(fp2_6['Time'],fp2_6['V_Sat'],label='6 okrążenie')
# plt.plot(fp2_7['Time'],fp2_7['V_Sat'],label='7 okrążenie')
# plt.legend()
# plt.xlabel('Czas [s]')
# plt.ylabel('Prędkość [km/h]')
# plt.title('Prędkość pojazdu w czasie')
# plt.show()

#podział na pliki
fp2_1.to_csv('electric_motor_temp_predict/fp2/FP2_1.csv',index=False)
fp2_2.to_csv('electric_motor_temp_predict/fp2/FP2_2.csv',index=False)
fp2_3.to_csv('electric_motor_temp_predict/fp2/FP2_3.csv',index=False)
fp2_4.to_csv('electric_motor_temp_predict/fp2/FP2_4.csv',index=False)
fp2_5.to_csv('electric_motor_temp_predict/fp2/FP2_5.csv',index=False)
fp2_6.to_csv('electric_motor_temp_predict/fp2/FP2_6.csv',index=False)
fp2_7.to_csv('electric_motor_temp_predict/fp2/FP2_7.csv',index=False)

# Wyscig
czas_od_startu_r = [429, 582, 730, 884, 1039, 1200, 1357]

#1 okrążenie od 7 minuty 9 sekund do 9 minuty 42 sekundy
race_1 = race[race['Time'] >= czas_od_startu_r[0]]
race_1 = race_1[race_1['Time'] <= czas_od_startu_r[1]]

#2 okrążenie od 9 minuty 42 sekund do 12 minuty 10 sekund
race_2 = race[race['Time'] >= czas_od_startu_r[1]]
race_2 = race_2[race_2['Time'] <= czas_od_startu_r[2]]

#3 okrążenie od 12 minuty 10 sekund do 14 minuty 44 sekund
race_3 = race[race['Time'] >= czas_od_startu_r[2]]
race_3 = race_3[race_3['Time'] <= czas_od_startu_r[3]]

#4 okrążenie od 14 minuty 44 sekund do 17 minuty 19 sekund
race_4 = race[race['Time'] >= czas_od_startu_r[3]]
race_4 = race_4[race_4['Time'] <= czas_od_startu_r[4]]

#5 okrążenie od 17 minuty 19 sekund do 19 minuty 59 sekund
race_5 = race[race['Time'] >= czas_od_startu_r[4]]
race_5 = race_5[race_5['Time'] <= czas_od_startu_r[5]]

#6 okrążenie od 19 minuty 59 sekund do 22 minuty 39 sekund
race_6 = race[race['Time'] >= czas_od_startu_r[5]]
race_6 = race_6[race_6['Time'] <= czas_od_startu_r[6]]

#poworwanie predkosci dla kazdego okrazenia na wspolnym wykresie
plt.plot(race_1['Time'],race_1['V_Sat'],label='1 okrążenie')
plt.plot(race_2['Time'],race_2['V_Sat'],label='2 okrążenie')
plt.plot(race_3['Time'],race_3['V_Sat'],label='3 okrążenie')
plt.plot(race_4['Time'],race_4['V_Sat'],label='4 okrążenie')
plt.plot(race_5['Time'],race_5['V_Sat'],label='5 okrążenie')
plt.plot(race_6['Time'],race_6['V_Sat'],label='6 okrążenie')
plt.legend()
plt.xlabel('Czas [s]')
plt.ylabel('Prędkość [km/h]')
plt.title('Prędkość pojazdu w czasie')
plt.show()

#zapis do plikow
race_1.to_csv('electric_motor_temp_predict/r/RACE_1.csv',index=False)
race_2.to_csv('electric_motor_temp_predict/r/RACE_2.csv',index=False)
race_3.to_csv('electric_motor_temp_predict/r/RACE_3.csv',index=False)
race_4.to_csv('electric_motor_temp_predict/r/RACE_4.csv',index=False)
race_5.to_csv('electric_motor_temp_predict/r/RACE_5.csv',index=False)
race_6.to_csv('electric_motor_temp_predict/r/RACE_6.csv',index=False)














