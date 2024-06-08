#Import bibliotek
import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler

#Wczytanie wszystkich plików z danymi
fp_all_tab = []

for i in range(1,5):
    fp_all_tab.append(np.array(pd.read_csv('fp1/FP1_' + str(i) + '.csv', sep=',',low_memory=False)))
for i in range(1,7):
    fp_all_tab.append(np.array(pd.read_csv('fp2/FP2_' + str(i) + '.csv', sep=',',low_memory=False)))
for i in range(1,7):
    fp_all_tab.append(np.array(pd.read_csv('r/RACE_' + str(i) + '.csv', sep=',',low_memory=False)))

df = pd.read_csv('r/RACE_1.csv', sep=',',low_memory=False)

"""
#Wyszukanie tabeli z największą ilością wierszy
the_longest_len = 0
for fp in fp_all_tab:
    len_fp = len(fp)
    if the_longest_len < len_fp:
        the_longest_len = len_fp

width = len(fp_all_tab[0][0])


#Dodanie wierszy wypełnionych zerami, by wszystkie tabele miały taki sam wymiar
for i in range(len(fp_all_tab)):
    if len(fp_all_tab[i]) < the_longest_len:
        for j in range(the_longest_len-len(fp_all_tab[i])):
            zero = np.zeros((the_longest_len-len(fp_all_tab[i]),width))
            fp_all_tab[i] = np.concatenate((fp_all_tab[i], zero))
"""
            

#Złączenie tabel
fp_all = fp_all_tab[0]
for i in range(1, len(fp_all_tab)):
    fp_all = np.concatenate((fp_all, fp_all_tab[i]))

#Dodanie podpisów kolumn
fp_all_df = pd.DataFrame(fp_all)
#fp_all_df.drop(0, inplace = True)
fp_all_df.columns = df.columns

#Zapis do pliku CSV
#fp_all_df.to_csv("race_without_zeros.csv", index = False)


#*******************************************************
#*** NORMALIZACJA DANYCH (WSZYSTKIE ZŁĄCZONE TABELE) ***
#*******************************************************

scaler = StandardScaler()
normalized_data = scaler.fit_transform(fp_all_df)
normalized_df = pd.DataFrame(normalized_data, columns=df.columns)

#Usunięcie kolumny index
normalized_df = normalized_df.reset_index(drop=True)

#*******************************************************

# Zapisz znormalizowane dane do pliku CSV


normalized_df.to_csv("fp_all_race_normalized.csv", index=False)

pierwsza_czesc = normalized_df.iloc[:4617, :]
druga_czesc = normalized_df.iloc[4617:5991+4616:]
trzecia_czesc = normalized_df.iloc[5991+4613:, :]

pierwsza_czesc.to_csv("fp1_123_normalized.csv", index=False)
druga_czesc.to_csv("fp2_123_normalized.csv", index=False)
trzecia_czesc.to_csv("race_123_normalized.csv", index=False)

# Odwrócenie transformacji




#*******************************************************
# ODWRACANIE NORMALIZACJI
#*******************************************************


file_name = "fp_all_race_normalized"
loaded_normalized_data = pd.read_csv(file_name + ".csv")
original_data = scaler.inverse_transform(loaded_normalized_data)
original_df = pd.DataFrame(original_data, columns=loaded_normalized_data.columns)
original_df.to_csv(file_name + "_original.csv", index=False)