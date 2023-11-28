import os
import pandas as pd
import audiofile
from amazing import functions as f
import tensorflow as tf
import sounddevice as sd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, ifft

print("Reading data csv...")
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

f.correlate(x,y)
# for i in range(len(audioTable['id'].values)):

## Grabbing BPM from video
# Help: https://stackoverflow.com/questions/34516323/heart-rate-monitor-using-opencv
print("Taking BPM from video...")
avg_brightness, fps = f.getVideoAvgBrightnesses("C:/Users/carme/OneDrive/Desktop/CMPT340/Final Project/video/Video/1.mp4")
lowcut = 0.5
highcut = 2.5

# Apply band-pass filter to average brightness values
# it makes the resulting heart rate signal smoother
filtered_brightness = f.getBandpassFilter(avg_brightness, lowcut, highcut, fps)
print("Plotting the detected signal from video...")
# Finding peaks
peaks, _ = find_peaks(filtered_brightness, height=0)
print(peaks)

plt.figure(figsize=(10, 6))
plt.plot(avg_brightness, label='Original Signal')
plt.plot(filtered_brightness, label='Filtered Signal')
plt.title('Average Brightness with Band-pass Filtering')
plt.xlabel('Frame')
plt.ylabel('Average Brightness')
plt.legend()
plt.plot(peaks, filtered_brightness[peaks], "x")
plt.show()

video_length = f.getVideoLengthSeconds("C:/Users/carme/OneDrive/Desktop/CMPT340/Final Project/video/Video/1.mp4")
peak_times = peaks / fps
print("Heartbeat peak times: ", peak_times)

avg_bpm = (len(peaks) / video_length) * 60
print("Overall average BPM without sliding window: ", avg_bpm)

# define a sliding window and step size (seconds)
WINDOW_SIZE = 6
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

# ## DL Model training code here
# print("DL Model Training...")
# training_data = []

# # Set your personal data path here:
# VIDEO_PATH = "" #EX. os.getcwd() + "\\src\\TrainingData\\Video\\"
# AUDIO_PATH = "" #EX. os.getcwd() + "\\src\\TrainingData\\Audio\\"

# # Process Video
# for i in range(1,50):
#     avg_brightnesses = f.getVideoAvgBrightnesses(VIDEO_PATH + str(i) + ".mp4")
#     # get audio signal 
#     audio_signal, sampling_rate = audiofile.read(AUDIO_PATH + str(i) + ".wav")
#     # add pair to training data
#     training_data.append([avg_brightnesses,audio_signal[0][:300000]])

# X = []
# Y = []

# for video,audio in training_data:
#     X.append(video)
#     Y.append(audio)

# X = np.array(X)
# Y = np.array(Y)

# training_size=45

# x_train = X[:training_size]
# y_train = Y[:training_size]
# x_test = X[training_size:]
# y_test = Y[training_size:]

# model = tf.keras.models.Sequential()

# model.add(tf.keras.layers.Flatten(input_shape=(250,)))
# model.add(tf.keras.layers.Dense(768, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(768, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(300000))

# model.compile(optimizer = 'adam', loss = 'mse') 

# model.fit(x_train, y_train, epochs = 5, validation_data=(x_test, y_test))
# model.save('heart_rate.model')

# # try predicting
# print("DL Model Prediction Testing...")
# new_model = tf.keras.models.load_model('heart_rate.model')
# x_test = f.getVideoAvgBrightnesses(VIDEO_PATH + str(48) + ".mp4")
# predictions = new_model.predict([x_test])
# print(predictions[0])

# # Write output signal to file
# with open("output.txt", "w") as txt_file:
#     for line in predictions[0]:
#         txt_file.write(str(line)+'\n')

# # Play sound - WARNING may be very loud
# sd.play(predictions[0],44100)
# sd.wait()