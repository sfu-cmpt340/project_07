import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

columnNames = ['id','date','subjectID','heartrate','state','activity','BMI','age','caffeineLevel','sleepDuration']

def constructTable(dataFilePath):
    table = pd.read_csv(dataFilePath)
    try:
        table.columns = columnNames
        table = table.astype({
            'id': int,
            'date': 'datetime64',
            'subjectID': int,
            'heartrate': str,
            'state': str,
            'activity': str,
            'BMI': float,
            'age': int,
            'caffeineLevel': str,
            'sleepDuration': float
        })
    
    except Exception as error:
        print('error:', error)   
        print("Error creating table - make sure columns in csv are same as above.")
 
    return table

def correlate(x,y):
    corr, pVal = stats.pearsonr(x, y)
    plt.scatter(x, y)
    title = "corr = " + str(corr)
    plt.title(title)
    plt.xlabel('Average Heart Rate (bpm)')
    plt.ylabel('Sleep Duration (hours)')

    if(pVal < 0.05):
        print("Statistically Significant.")
    else:
        print(pVal)

    plt.show()

