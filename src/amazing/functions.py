import pandas as pd
import numpy as np

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
    
    except:
        print("Error creating table - make sure columns in csv are same as above.")

 
    return table
