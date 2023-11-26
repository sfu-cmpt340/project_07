import os
import pandas as pd
import matplotlib as plt
from amazing import functions as f

currentDir = os.getcwd()
audioData = "audio.csv"
audioPath = os.path.join(currentDir, audioData)

audioTable = f.constructTable(audioPath)

print(audioTable)