import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

columnNames = ['id','date','heartrate','state','activity','BMI','age','caffeineLevel','sleepDuration']

def constructTable(dataFilePath):
    table = pd.read_csv(dataFilePath)
    try:
        table.columns = columnNames
        table = table.astype({
            'id': int,
            'date': 'datetime64[ns]',
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

def getVideoAvgBrightnesses(videoPath):
    vid_source = cv2.VideoCapture(videoPath)
    if vid_source.get(cv2.CAP_PROP_FPS) > 40:
        iterator=2
        numFrames=500
    else:
        iterator=1
        numFrames=250
    
    avg_brightnesses = []
    # Get average brightness value of each frame
    success, img = vid_source.read()
    count = 0
    while (count < numFrames and success):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.resize(gray_img,(500,500))
        avg_brightnesses.append(np.average(gray_img))
        success, img = vid_source.read() 
        count+=iterator
    return avg_brightnesses, vid_source.get(cv2.CAP_PROP_FPS)

# Order: bandwidth of the frequency range that the filter passes
def getBandpassFilter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def getVideoLengthSeconds(videoPath):
    vid_source = cv2.VideoCapture(videoPath)
    fps = vid_source.get(cv2.CAP_PROP_FPS)
    frame_count = int(vid_source.get(cv2.CAP_PROP_FRAME_COUNT))
    return frame_count / fps