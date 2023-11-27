import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import cv2
import audiofile
import sounddevice as sd


def getVideoAvgBrightnesses(videoPath, numFrames):
    vid_source = cv2.VideoCapture(videoPath)
    avg_brightnesses = []
    # Get average brightness value of each frame
    success, img = vid_source.read()
    count = 0
    while (count < numFrames):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.resize(gray_img,(100,100))
        avg_brightnesses.append(np.average(gray_img))
        success, img = vid_source.read() 
        count+=1
    return avg_brightnesses

training_data = []

# Set your personal data path here:
VIDEO_PATH = ""
AUDIO_PATH = ""

# Process Video
for i in range(1,5):
    avg_brightnesses = getVideoAvgBrightnesses(VIDEO_PATH + str(i) + ".mp4",250)
    # get audio signal 
    audio_signal, sampling_rate = audiofile.read(AUDIO_PATH + str(i) + ".wav")
    # add pair to training data
    training_data.append([avg_brightnesses,audio_signal[0][:420000]])

X = []
Y = []

for video,audio in training_data:
    X.append(video)
    Y.append(audio)

X = np.array(X)
Y = np.array(Y)

training_size=3


x_train = X[:training_size]
y_train = Y[:training_size]
x_test = X[training_size:]
y_test = Y[training_size:]

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(250,)))
model.add(tf.keras.layers.Dense(768, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(768, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(420000))

model.compile(optimizer = 'adam', loss = 'mse') 

model.fit(x_train, y_train, epochs = 5, validation_data=(x_test, y_test))
model.save('cool_model.model')
# try predicting
new_model = tf.keras.models.load_model('cool_model.model')

# try_predict_this_vid = np.array(try_predict_this_vid);
# print(try_predict_this_vid.shape)
predictions = new_model.predict([x_test])
print(len(predictions[0]))
print(predictions[0])

# sd.play(predictions[0],44100)
# sd.wait()