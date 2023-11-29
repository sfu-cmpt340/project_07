import os
import pandas as pd
import audiofile
from amazing import functions as f
import tensorflow as tf
import sounddevice as sd
import numpy as np

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

## DL Model training code here
print("DL Model Training...")
training_data = []

# Set your personal data path here:
VIDEO_PATH = "" #EX. os.getcwd() + "\\src\\TrainingData\\Video\\"
AUDIO_PATH = "" #EX. os.getcwd() + "\\src\\TrainingData\\Audio\\"

# Process Video
for i in range(1,50):
    avg_brightnesses = f.getVideoAvgBrightnesses(VIDEO_PATH + str(i) + ".mp4")
    # get audio signal 
    audio_signal, sampling_rate = audiofile.read(AUDIO_PATH + str(i) + ".wav")
    # add pair to training data
    training_data.append([avg_brightnesses,audio_signal[0][:300000]])

X = []
Y = []

for video,audio in training_data:
    X.append(video)
    Y.append(audio)

X = np.array(X)
Y = np.array(Y)

training_size=45

x_train = X[:training_size]
y_train = Y[:training_size]
x_test = X[training_size:]
y_test = Y[training_size:]

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(250,)))
model.add(tf.keras.layers.Dense(768, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(768, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(300000))

model.compile(optimizer = 'adam', loss = 'mse') 

model.fit(x_train, y_train, epochs = 5, validation_data=(x_test, y_test))
model.save('heart_rate.model')

# try predicting
print("DL Model Prediction Testing...")
new_model = tf.keras.models.load_model('heart_rate.model')
x_test = f.getVideoAvgBrightnesses(VIDEO_PATH + str(48) + ".mp4")
predictions = new_model.predict([x_test])
print(predictions[0])

# Write output signal to file
with open("output.txt", "w") as txt_file:
    for line in predictions[0]:
        txt_file.write(str(line)+'\n')

# Play sound - WARNING may be very loud
sd.play(predictions[0],44100)
sd.wait()

bpms = []

for filename in os.listdir(AUDIO_PATH):
    if filename.endswith(".wav"):
        file_path = os.path.join(AUDIO_PATH, filename)
        bpm = f.getAudioBPM2(file_path)
        bpms.append((filename, bpm))
