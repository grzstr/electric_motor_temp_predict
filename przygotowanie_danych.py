#Import bibliotek
import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler

#Wczytanie wszystkich plików z danymi
fp_all_tab = []

for i in range(1,5):
    fp_all_tab.append(np.array(pd.read_csv('fp1/FP1_' + str(i) + '.csv', sep=',',low_memory=False)))

for i in range(1,8):
    fp_all_tab.append(np.array(pd.read_csv('fp2/FP2_' + str(i) + '.csv', sep=',',low_memory=False)))

df = pd.read_csv('fp1/FP1_1.csv', sep=',',low_memory=False)

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

#Złączenie tabel
fp_all = fp_all_tab[0]
for i in range(1, len(fp_all_tab)):
    fp_all = np.concatenate((fp_all, fp_all_tab[i]))

#Dodanie podpisów kolumn
fp_all_df = pd.DataFrame(fp_all)
#fp_all_df.drop(0, inplace = True)
fp_all_df.columns = df.columns

#Zapis do pliku CSV
fp_all_df.to_csv("fp_all.csv", index = False)

#****************************************************
#*** NORMALIZACJA DANYCH (PRZED ZŁĄCZENIEM TABEL) ***
# #****************************************************
# """
# scaler = StandardScaler()

# for i in range(len(fp_all_tab)):
#     fp_df = pd.DataFrame(fp_all_tab[i])
#     # Zastosuj normalizację do danych
#     normalized_data = scaler.fit_transform(fp_df)
#     # Stwórz nowy DataFrame z znormalizowanymi danymi
#     normalized_df_one = pd.DataFrame(normalized_data, columns=df.columns)
#     if i == 0:
#         normalized_df = normalized_df_one
#     normalized_df = pd.concat([normalized_df, normalized_df_one], axis=0)

# #****************************************************
# """


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
normalized_df.to_csv("fp_all_normalized.csv", index=False)



# Load fp1 data
fp1_tab = []
for i in range(1, 5):
    fp1_tab.append(np.array(pd.read_csv('fp1/FP1_' + str(i) + '.csv', sep=',', low_memory=False)))

# Find the longest length in fp1 data
the_longest_len_fp1 = max(len(fp) for fp in fp1_tab)

# Add zero rows to make all tables the same dimension
for i in range(len(fp1_tab)):
    if len(fp1_tab[i]) < the_longest_len_fp1:
        zero = np.zeros((the_longest_len_fp1 - len(fp1_tab[i]), width))
        fp1_tab[i] = np.concatenate((fp1_tab[i], zero))

# Concatenate fp1 data
fp1_all = np.concatenate(fp1_tab, axis=0)

# Create DataFrame and add column names
fp1_all_df = pd.DataFrame(fp1_all, columns=df.columns)

# Normalize fp1 data
normalized_fp1_data = scaler.transform(fp1_all_df)
normalized_fp1_df = pd.DataFrame(normalized_fp1_data, columns=df.columns)

# Save normalized fp1 data to CSV
normalized_fp1_df.to_csv("fp1_normalized.csv", index=False)


# Load fp2 data
fp2_tab = []
for i in range(1, 8):
    fp2_tab.append(np.array(pd.read_csv('fp2/FP2_' + str(i) + '.csv', sep=',', low_memory=False)))

# Find the longest length in fp2 data
the_longest_len_fp2 = max(len(fp) for fp in fp2_tab)

# Add zero rows to make all tables the same dimension
for i in range(len(fp2_tab)):
    if len(fp2_tab[i]) < the_longest_len_fp2:
        zero = np.zeros((the_longest_len_fp2 - len(fp2_tab[i]), width))
        fp2_tab[i] = np.concatenate((fp2_tab[i], zero))

# Concatenate fp2 data
fp2_all = np.concatenate(fp2_tab, axis=0)

# Create DataFrame and add column names
fp2_all_df = pd.DataFrame(fp2_all, columns=df.columns)

# Normalize fp2 data
normalized_fp2_data = scaler.transform(fp2_all_df)
normalized_fp2_df = pd.DataFrame(normalized_fp2_data, columns=df.columns)

# Save normalized fp2 data to CSV
normalized_fp2_df.to_csv("fp2_normalized.csv", index=False)
