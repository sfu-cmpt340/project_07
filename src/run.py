import os
import pandas as pd
from amazing import functions as f
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, ifft
from sklearn import svm
import pickle

print("Reading data csv...")
video_data_path = os.path.join(os.getcwd(), "video.csv")
data_table = f.constructTable(video_data_path) # overall table from csv

x = data_table[['id','heartrate']] # one table for id and heartrate
y = data_table.drop('heartrate', axis=1) # another excluding heartrate

#printing tables
print("-----X-----\n",x.head())
print("-----Y-----\n",y.head())

## Grabbing BPM from video
# Help: http://www.ignaciomellado.es/blog/Measuring-heart-rate-with-a-smartphone-camera
# Set your personal data path here:
VIDEO_PATH = os.path.join(os.getcwd() + "/Video/") #EX. os.getcwd() + "\\src\\TrainingData\\Video\\"
AUDIO_PATH = os.path.join(os.getcwd() + "/Audio/") #EX. os.getcwd() + "\\src\\TrainingData\\Audio\\"
for i in range(1,50):
    print("__________Taking BPM from video ", i, "_______________")
    avg_brightness, fps = f.getVideoAvgBrightnesses(VIDEO_PATH + str(i) + ".mp4")
    lowcut = 0.5
    highcut = 2.5

    # Apply band-pass filter to average brightness values
    # it makes the resulting heart rate signal smoother
    filtered_brightness = f.getBandpassFilter(avg_brightness, lowcut, highcut, fps)
    print("Plotting the detected signal from video...")
    # Finding peaks
    peaks, _ = find_peaks(filtered_brightness, height=0)
    print(peaks)
    # Plotting for easier debugging
    # plt.figure(figsize=(10, 6))
    # plt.plot(avg_brightness, label='Original Signal')
    # plt.plot(filtered_brightness, label='Filtered Signal')
    # plt.title('Average Brightness with Band-pass Filtering')
    # plt.xlabel('Frame')
    # plt.ylabel('Average Brightness')
    # plt.legend()
    # plt.plot(peaks, filtered_brightness[peaks], "x")
    # plt.show()

    video_length = f.getVideoLengthSeconds(VIDEO_PATH + str(i) + ".mp4")
    if (fps > 40):
        fps /= 2
    peak_times = peaks / fps
    print("Heartbeat peak times: ", peak_times)
    avg_bpm = (len(peaks) / video_length) * 60
    print("Overall average BPM without sliding window: ", avg_bpm)

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

    print(f"Windowed bpms, {WINDOW_SIZE}s window size: ", window_bpms)
    print("Windowed bpms average: ", np.average(window_bpms))

# DL Model training code here
print("DL Model Training...")
CATEGORIES = ["60-70","70-80","80-90","90-100","100-110","110-120","120-130","130-140","140-150","150-160"]
heartrates = data_table['heartrate']
Classifications = [f.getHrRange(x) for x in heartrates]

# Define training data
X = []
Y = []

# Process Video
for i in range(0,50):
     print(f"Processing sample {i+1}")
     avg_brightnesses, fps = f.getVideoAvgBrightnesses(VIDEO_PATH + str(i+1) + ".mp4")
     # Apply band pass filter to average brightnesses to make it easier to distinguish peaks
     filtered_brightness = f.getBandpassFilter(avg_brightnesses, 0.5, 2.5, fps)
     # Construct training data
     X.append(filtered_brightness)
     Y.append(CATEGORIES.index(Classifications[i]))

clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(X[:-1], Y[:-1])

with open('predict_heart_rate_from_video.pkl', 'wb') as fid:
    pickle.dump(clf, fid)  

# try predicting
print("DL Model Prediction Testing...")

# load model
with open('predict_heart_rate_from_video.pkl', 'rb') as fid:
    clf_loaded = pickle.load(fid)

# load test data
test_indices = range(66,71)
x_test = []
for i in test_indices:
    test_brightnesses, test_fps = f.getVideoAvgBrightnesses(VIDEO_PATH + str(i) + ".mov")
    test_brightnesses = f.getBandpassFilter(test_brightnesses, 0.5, 2.5, test_fps)
    x_test.append(test_brightnesses)

# make prediction    
predictions = clf_loaded.predict(x_test)

for i in range(len(predictions)):
    print(f"Heartrate prediction for video {test_indices[i]} is: {CATEGORIES[predictions[i]]}")