import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from scipy.signal import find_peaks

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

def getHrZoneIndex(hr):
    hr = int(hr)
    if 60 <= hr < 90:
        return 0
    elif 90 <= hr < 110:
        return 1
    elif 110 <= hr < 130:
        return 2
    elif 130 <= hr < 160:
        return 3
    # elif 150 <= hr < 170:
    #     return 4

import PySimpleGUI as sg



def getMainSelectionPage(classifier, videoClf):
    main_menu_layout = [
        [sg.Text("How would you like to predict your heart rate?")],
        [sg.Button("Predict Heart Rate from Your States")], 
        [sg.Button("Predict Heart rate from a video (With DL)")],
        [sg.Button("Predict Heart rate from a video (Without DL)")],
        [sg.Button("Exit")]
    ]
    window = sg.Window('Main Selection', main_menu_layout)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED or event == 'Exit':
            break

        if event == 'Predict Heart Rate from Your States':
            window.close()
            getHrPredictionPage(classifier, videoClf)

        if event == 'Predict Heart rate from a video (With DL)': # method 1
            window.close()
            getVideoPage(classifier, videoClf, 1)

        if event == 'Predict Heart rate from a video (Without DL)': # method 2
            window.close()
            getVideoPage(classifier, videoClf, 2)

    window.close()

def getHrPredictionPage(classifier, videoClf):
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
            getMainSelectionPage(classifier, videoClf)

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
            getHrPredictionResultPage(classifier, videoClf, state, activity, bmi, age, caffeine, sleep_duration, gender, fitness)

    window.close()

def getHrPredictionResultPage(classifier, videoClf, state, activity, bmi, age, caffeine, sleep_duration, gender, fitness):
    # STATE, ACTIVITY, BMI, AGE, CAFFEINE INTAKE, SLEEP DURATION
    data = [state, activity, bmi, age, caffeine, sleep_duration, gender, fitness]
    labels = ['Very Low (0-60)', 'Low (60 - 70)', 'Medium Low (70-80)', 'Medium (80-90)', 'Medium High (90-100)', 'High (100-120)', 'Very High (120-140)', 'Extremely High (140-160)', 'Are You Ok (160+)']
    colour = ['darkblue', 'blue', 'lightblue', 'green', 'yellow', 'orange', 'red', 'darkred','purple4']

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
            getHrPredictionPage(classifier, videoClf)
        elif event == "Main Menu":
            window.close()
            getMainSelectionPage(classifier, videoClf)

    window.close()

def getVideoPage (classifier, videoClf, method):
    video_layout = [
        [sg.Text('Select a MP4 file:')],
        [sg.InputText(key='file_path'), sg.FileBrowse()],
        [sg.Button("Predict My Heart Rate Range!"), sg.Button("Return")]
    ]
    window_text = 'Video Prediction Method ' + str(method)
    window = sg.Window(window_text, video_layout)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break

        if event == 'Return':
            window.close()
            getMainSelectionPage(classifier, videoClf)
        elif event == 'Predict My Heart Rate Range!':
            path = values['file_path']
            if method == 1: # ML method
                result = getWithMLVideoPrediction(videoClf, path)
                window.close()
                getVideoPredictionResultPage(classifier, videoClf, result, method)
            else: # No ML method
                result = getNoMLVideoPrediction(path)
                window.close()
                getVideoPredictionResultPage(classifier, videoClf, result, method)

    window.close()

def getVideoPredictionResultPage (classifier, videoClf, result, method):
    if method == 1:
        video_layout = [
            [sg.Text("Your predicted heart beat range is: "), sg.Text(result)],
            [sg.Button("Redo"), sg.Button("Main Menu"), sg.Button("Exit")]
        ]
    else:
        video_layout = [
            [sg.Text("Your detected heart rate throughout the video is: "), sg.Text(result)],
            [sg.Text("Your average heart rate throughout the video is: "), sg.Text(np.average(result))],
            [sg.Button("Redo"), sg.Button("Main Menu"), sg.Button("Exit")]
        ]
    window = sg.Window('Video Prediction Result', video_layout)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED or event == "Exit":
            break

        if event == 'Redo':
            window.close()
            getVideoPage(classifier, videoClf, method)
        elif event == "Main Menu":
            window.close()
            getMainSelectionPage(classifier, videoClf)

    window.close()

def getNoMLVideoPrediction (path):
    # Help: http://www.ignaciomellado.es/blog/Measuring-heart-rate-with-a-smartphone-camera
    avg_brightness, fps = getVideoAvgBrightnesses(path)
    lowcut = 0.5
    highcut = 2.5

    # Apply band-pass filter to average brightness values
    # it makes the resulting heart rate signal smoother
    filtered_brightness = getBandpassFilter(avg_brightness, lowcut, highcut, fps)
    print("Plotting the detected signal from video...")
    # Finding peaks
    peaks, _ = find_peaks(filtered_brightness, height=0)
    print(peaks)
    plt.figure(figsize=(10, 6))
    plt.plot(avg_brightness, label='Original Signal')
    plt.plot(filtered_brightness, label='Filtered Signal')
    plt.title('Average Brightness with Band-pass Filtering [Close to continue]')
    plt.xlabel('Frame')
    plt.ylabel('Average Brightness')
    plt.legend()
    plt.plot(peaks, filtered_brightness[peaks], "x")
    plt.show()

    video_length = getVideoLengthSeconds(path)
    if (fps > 40):
        fps /= 2
    peak_times = peaks / fps
    # print("Heartbeat peak times: ", peak_times)
    # avg_bpm = (len(peaks) / video_length) * 60
    # print("Overall average BPM without sliding window: ", avg_bpm)

    # define a sliding window and step size (seconds)
    WINDOW_SIZE = 5
    STEP_SIZE = 0.5

    num_windows = int((peak_times[-1] - peak_times[0]) / STEP_SIZE)
    window_starts = np.zeros(num_windows)
    avg_bpm_in_windows = np.zeros(num_windows)
    window_peaks = []

    for i in range(num_windows):
        window_start = peak_times[0] + i * STEP_SIZE
        window_end = window_start + WINDOW_SIZE

        if window_end > peak_times[-1]:
            break

        # Find indices of peaks within the current window
        peaks_in_window = np.where((peak_times >= window_start) & (peak_times < window_end))[0]
        window_peaks.append(peak_times[peaks_in_window])
        window_starts[i] = window_start

    window_bpms = []
    for window in window_peaks:
        # number of heartbeats / window time = bps
        window_bpms.append((len(window) / WINDOW_SIZE) * 60) 

    return window_bpms

def getWithMLVideoPrediction (videoClf, path):
    # load test data
    x = []
    brightnesses, fps = getVideoAvgBrightnesses(path)
    brightnesses = getBandpassFilter(brightnesses, 0.5, 2.5, fps)
    x.append(brightnesses)

    # make prediction    
    predict = videoClf.predict(x)
    CATEGORIES = ["Zone 1 - Resting/Very light - 60-90 bpm", "Zone 2 - Light - 90-110 bpm", "Zone 3 - Moderate - 110-130 bpm", "Zone 4 - Hard - 130-160 bpm"]
    return CATEGORIES[predict[0]]
