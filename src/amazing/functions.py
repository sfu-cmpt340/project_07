import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier

columnNames = ['id','date','heartrate','state','activity','BMI','age','caffeineLevel','sleepDuration', 'gender', 'fitnessLevel']

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
            'sleepDuration': float,
            'gender': str,
            'fitnessLevel': int
        })
    
    except Exception as error:
        print('error:', error)   
        print("Error creating table - make sure columns in csv are same as the one in functions.py.")
 
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

def getHrRange(hr):
    hr = int(hr)
    if 60 <= hr < 70:
        return "60-70"
    elif 70 <= hr < 80:
        return "70-80"
    elif 80 <= hr < 90:
        return "80-90"
    elif 90 <= hr < 100:
        return "90-100"
    elif 100 <= hr < 110:
        return "100-110"
    elif 110 <= hr < 120:
        return "110-120"
    elif 120 <= hr < 130:
        return "120-130"
    elif 130 <= hr < 140:
        return "130-140"
    elif 140 <= hr < 150:
        return "140-150"
    elif 150 <= hr <= 160:
        return "150-160"
import PySimpleGUI as sg



def getMainSelectionPage(classifier):
    main_menu_layout = [
        [sg.Text("How would you like to predict your heart rate?")],
        [sg.Button("Predict Heart Rate from Your States")], 
        [sg.Button("Predict Heart rate from a video")],
        [sg.Button("Exit")]
    ]
    window = sg.Window('Main Selection', main_menu_layout)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED or event == 'Exit':
            break

        if event == 'Predict Heart Rate from Your States':
            window.close()
            getHrPredictionPage(classifier)

        if event == 'Predict Heart rate from a video':
            window.close()
            getVideoPredictionPage(classifier)

    window.close()

def getHrPredictionPage(classifier):
    hr_prediction_layout = [
        [sg.Text("Input your information")],
        [sg.Text('State'),sg.Combo(['Sitting', 'Standing', 'Lying Down'], key='state')],
        [sg.Text('Activity'),sg.Combo(['Resting', 'Light Exercise', 'Moderate Exercise', 'Intense Exercise'], key='activity')],
        [sg.Text('Age (Eg. 22)'), sg.InputText(key='age', size=(10, 5), enable_events=True)],
        [sg.Text('BMI (Eg. 25.0)'), sg.InputText(key='bmi', size=(5, 1), enable_events=True)],
        [sg.Text('Approximate Caffeine Intake'), sg.Combo(['None', 'Low (1-2 cup of tea)', 'Moderate (1-2 cup of coffee)', 'High (3+ cup of coffee)'], key='caffeine')],
        [sg.Text('Sleep Duration (Eg. 8.0)'), sg.InputText(key='sleep_duration', size=(5, 1), enable_events=True)],
        [sg.Text('Biological Gender'),sg.Combo(['Female', 'Male'], key='gender')],
        [sg.Text('Self Assessed Fitness Level'),sg.Combo(['Not Active', 'Occasionally Active', 'Average', 'Very Active', 'Extremely Active'], key='fitness')],
        [sg.Button('Predict My Current Heart Rate Range!'), sg.Button('Return')]
    ]

    window = sg.Window('Heart Rate Prediction', hr_prediction_layout)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break

        if event == 'Return':
            window.close()
            getMainSelectionPage(classifier)

        if event == 'Predict My Current Heart Rate Range!':
            # Validate input data types
            state = values['state']
            activity = values['activity']
            caffeine = values['caffeine']
            age = values['age']
            bmi = values['bmi']
            sleep_duration = values['sleep_duration']
            gender = values['gender']
            fitness = values['fitness']

            if state == 'Sitting':
                state = 1
            elif state == 'Standing':
                state = 2
            else:
                state = 0

            if activity == 'Resting':
                activity = 0
            elif activity == 'Light Exercise':
                activity = 1
            elif activity == 'Moderate Excercise':
                activity = 2
            else:
                activity = 3    

            if caffeine == 'None':
                caffeine = 0
            elif caffeine == 'Low (1-2 cup of tea)':
                caffeine = 1
            elif caffeine == 'Moderate (1-2 cup of coffee)':
                caffeine = 2
            else:
                caffeine = 3
            if gender == 'Female':
                gender = 0
            elif gender == 'Male':
                gender = 1

            if fitness == 'Not Active':
                fitness = 1
            elif fitness == 'Occasionally Active':
                fitness = 2
            elif fitness == 'Average':
                fitness = 3
            elif fitness == 'Very Active':
                fitness = 4
            else:
                fitness = 5  

            try:
                age = int(age)
                bmi = float(bmi)
                sleep_duration = float(sleep_duration)
            except:
                sg.popup("Incorrect Input!")

            print (state, activity, bmi, age, caffeine, sleep_duration, gender)
            window.close()
            getHrPredictionResultPage(classifier, state, activity, bmi, age, caffeine, sleep_duration, gender, fitness)

    window.close()

def getHrPredictionResultPage(classifier, state, activity, bmi, age, caffeine, sleep_duration, gender, fitness):
    # STATE, ACTIVITY, BMI, AGE, CAFFEINE INTAKE, SLEEP DURATION
    data = [state, activity, bmi, age, caffeine, sleep_duration, gender, fitness]
    labels = ['Very Low (0-60)', 'Low (60 - 70)', 'Medium Low (70-80)', 'Medium (80-90)', 'Medium High (90-100)', 'High (100-120)', 'Very High (120-140)', 'Extremely High (140-160)']
    colour = ['darkblue', 'blue', 'lightblue', 'green', 'yellow', 'orange', 'red', 'darkred']

    prediction = classifier.predict([data])[0]
    result_layout = [
        [sg.Text("Your predicted Heart rate range is: "), sg.Text(labels[prediction], text_color=colour[prediction])],
        [sg.Button("Redo"), sg.Button("Main Menu"), sg.Button("Exit")]
    ]
    window = sg.Window('BPM Prediction Result', result_layout)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED or event == 'Exit':
            break

        if event == 'Redo':
            window.close()
            getHrPredictionPage(classifier)
        elif event == "Main Menu":
            window.close()
            getMainSelectionPage(classifier)

    window.close()

def getVideoPredictionPage (classifier):
    video_layout = [
        [sg.Text('Select a MP4 file:')],
        [sg.InputText(key='file_path'), sg.FileBrowse()],
        [sg.Button("Predict My Heart Rate Range!"), sg.Button("Return")]
    ]
    window = sg.Window('Video Prediction', video_layout)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break

        if event == 'Return':
            window.close()
            getMainSelectionPage(classifier)
        elif event == 'Predict My Heart Rate Range!':
            # TODO: placeholder result right now, it holds path rn
            result = values['file_path']
            window.close()
            getVideoPredictionResultPage(classifier, result)

    window.close()

def getVideoPredictionResultPage (classifier, result):
    video_layout = [
        [sg.Text("Your predicted Heart rate is: "),sg.Text(result)],
        [sg.Button("Redo"), sg.Button("Main Menu"), sg.Button("Exit")]
    ]
    window = sg.Window('Video Prediction Result', video_layout)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED or event == "Exit":
            break

        if event == 'Redo':
            window.close()
            getVideoPredictionPage(classifier)
        elif event == "Main Menu":
            window.close()
            getMainSelectionPage(classifier)

    window.close()
