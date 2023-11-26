import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from amazing import functions as f

currentDir = os.getcwd()
audioData = "audio.csv"
audioPath = os.path.join(currentDir, audioData)

audioTable = f.constructTable(audioPath)

avgHeartRates = []
for hrRange in audioTable['heartrate'].values:
    bounds = hrRange.split("-")
    
    avgHeartRates.append((int(bounds[0]) + int(bounds[1]))/2)


# correlation between sleep duration & heart rate
x = avgHeartRates
y = audioTable['sleepDuration'].values
corr, pVal = stats.pearsonr(x, y)
plt.scatter(x, y)
title = "corr = " + str(corr)
plt.title(title)

plt.show()


# for i in range(len(audioTable['id'].values)):
