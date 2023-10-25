# Predykcja temperatury silnika motocykla elektycznego

## Skład grupy: Piotr Wyszyński 252921 | Grzegorz Strama 
### 1. Cel projektu

Wykorzystanie 1-wymiarowej neuronowej sieci konwolucyjnej do predykcji temperatury silnika synchroniczniego z magnesami trwałymi (PMSM), na podstawie danych z pozostałych czujników motocykla elektrycznego.

### 2. Opis danych

Dane zostały zebrane podczas testów prototypu elektrycznego motocykla wyścigowego. Silnik jest chłodzony z wykorzystaniem płaszcza wodnego, a woda jest chłodzona z wykorzystaniem chłodnicy z czujnikami temperatury na wejściu i wyjściu chłodnic.
Najważniejsze parametry w kontekście temperatury silnika:

* Temperatura otoczenia **[°C]**
* Temperatura wejściowa chłodnicy **[°C]**
* Temperatira wyjściowa chłodnicy **[°C]**
* u_d - Składowa napięcia w osi d **[V]**
* u_q - Składowa napięcia w osi q **[V]**
* Prędkość silnika **[RPM]**
* Moment obrotowy wywołany prądem **[Nm]**
* i_d - Składowa prądu w osi d **[A]**
* i_q - Składowa prądu w osi q **[A]**

### 3. Kamienie milowe:

26.10	Wybór tematu, przedstawienie dokumentu początkowego

16.11	Przygotowanie danych: Ekstrakcja danych

30.11	Przygotowanie danych: Kształtowanie danych czasowych

14.12	Budowa modelu i tuning hiperparametrów

11.01	Opracowanie wyników. Przygotowanie dokumentacji oraz prezentacji.

18.01/01.02	Oddanie raportu końcowego

