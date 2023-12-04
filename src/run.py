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

## DL Model training code here
print("DL Model Training...")
training_data = []

# Process Video
for i in range(1,50):
    avg_brightnesses, _ = f.getVideoAvgBrightnesses(VIDEO_PATH + str(i) + ".mp4")
    # get audio signal 
    #audio_signal, sampling_rate = audiofile.read(AUDIO_PATH + str(i) + ".wav")
    # add pair to training data
    #training_data.append([avg_brightnesses,CATEGORIES.index(Classifications[i])])
    X.append(avg_brightnesses)
    Y.append(CATEGORIES.index(Classifications[i]))



# for video,audio in training_data:
#     X.append(video)
#     Y.append(audio)

X = np.array(X)
Y = np.array(Y)
# X = tf.keras.utils.normalize(X,axis=1)
training_size=25

x_train = X[:training_size]
y_train = Y[:training_size]
print(f"x_train is {x_train}")
print(f"y_train is {y_train}")
x_test = X[training_size:]
y_test = Y[training_size:]

#model = tf.keras.models.Sequential()

# model.add(tf.keras.layers.Dense(128,input_shape=(250,),activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(len(CATEGORIES), activation=tf.nn.softmax))

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])


model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy']) 

model.fit(x_train, y_train,epochs = 5, validation_data=(x_test, y_test))
model.save('heart_rate.model')

# try predicting
print("DL Model Prediction Testing...")
new_model = tf.keras.models.load_model('heart_rate.model')
x_test, _ = f.getVideoAvgBrightnesses(VIDEO_PATH + str(48) + ".mp4")
predictions = new_model.predict([x_test])
print(predictions)
print(CATEGORIES[np.argmax(predictions[0])])

# Write output signal to file
# with open("output.txt", "w") as txt_file:
#     for line in predictions[0]:
#         txt_file.write(str(line)+'\n')

# Play sound - WARNING may be very loud
# sd.play(predictions[0],44100)
# sd.wait()