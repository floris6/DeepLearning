import pandas as pd
import numpy as np

# Load the weather dataset (result.txt)
weather_cols = ["STN", "YYYYMMDD", "HH", "DD", "FH", "T", "TD", "SQ", "Q", "P", "N", 'ID']
weather = pd.read_csv("result (1).txt", sep=",", names=weather_cols, skiprows=1)
weather["HH"] = weather["HH"].astype(int)
weather["HH"] = weather["HH"] - 1
weather["ID"] = 0

for i in range(len(weather)):
    weather.loc[i, "ID"] = i

#print(weather)

energy = pd.read_csv("Belpex.txt", sep=";", names=["DATETIME", "EUR"], skiprows=1, encoding="latin1")

energy = energy.sort_values(by="DATETIME", ascending=True)

energy.columns = energy.columns.str.strip()
energy["EUR"] = energy["EUR"].str.replace("  ", "", regex=False)
energy["EUR"] = energy["EUR"].str.replace(",", ".").astype(float)
energy["EUR"] = np.round(energy["EUR"] / 10) * 10

def map_EUR(value):
    if -200 < value <= -100:
        return 0
    elif -100 < value <= -50:
        return 2
    elif -50 < value <= -25:
        return 3
    elif -25 < value <= 0:
        return 4
    elif 0 < value <= 25:
        return 5
    elif 25 < value <= 50:
        return 6
    elif 50 < value <= 75:
        return 7
    elif 75 < value <= 100:
        return 8
    elif 100 < value <= 125:
        return 9
    elif 125 < value <= 150:
        return 10
    elif 150 < value <= 200:
        return 11
    elif 200 < value <= 375:
        return 12
    else:
        return 13

energy["EUR"] = energy["EUR"].apply(map_EUR)
energy["EUR"] = energy["EUR"].astype(int)
# -30 until 240


energy["ID"] = 0
for i in range(len(energy) - 1, -1, -1):  # Start from the bottom and go up
    energy.loc[i, "ID"] = len(energy) - 1 - i  # Assign the ID from 0 to len(energy) - 1


#print(energy)

merged = pd.merge(weather, energy[['ID', 'EUR']], on='ID', how="left")

weather_cols = ["YYYYMMDD", "HH", "DD", "FH", "T", "TD", "SQ", "Q", "P", "N", "EURdataset2.py"]
merged = merged[weather_cols]
merged['YYYYMMDD'] = (merged['YYYYMMDD'] % 10000) / 1000
merged["HH"] = (merged["HH"] / 12) -1
merged["DD"] = (merged["DD"] / 180) -1
merged["FH"] = merged["FH"] / 100 - 0.4
merged["T"] = (merged["T"] / 100) -1.5
merged["TD"] = (merged["TD"] / 100) -1
merged["SQ"] = (merged["SQ"] / 5) -0.5
merged["Q"] = (merged["Q"] / 100) -1
merged["P"] = (merged["P"] / 10000) -1
merged["N"] = pd.to_numeric(merged["N"], errors='coerce')
merged["N"] = (merged["N"] / 10) -1

print(merged)


merged.to_csv("merged_weather_energy.csv", index=False)


