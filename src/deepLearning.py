import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import cv2
import audiofile
import sounddevice as sd


def getVideoAvgBrightnesses(videoPath):
    vid_source = cv2.VideoCapture(videoPath)
    if vid_source.get(cv2.CAP_PROP_FPS) > 40:
        iterator=2;
        numFrames=500;
    else:
        iterator=1;
        numFrames=250;
    
    avg_brightnesses = []
    # Get average brightness value of each frame
    success, img = vid_source.read()
    count = 0
    while (count < numFrames and success):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.resize(gray_img,(100,100))
        avg_brightnesses.append(np.average(gray_img));
        success, img = vid_source.read() 
        count+=iterator
    return avg_brightnesses

training_data = []

# Set your personal data path here:
VIDEO_PATH = "" #EX. os.getcwd() + "\\src\\TrainingData\\Video\\"
AUDIO_PATH = "" #EX. os.getcwd() + "\\src\\TrainingData\\Audio\\"

# Process Video
for i in range(1,50):
    avg_brightnesses = getVideoAvgBrightnesses(VIDEO_PATH + str(i) + ".mp4")
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
# new_model = tf.keras.models.load_model('heart_rate.model')
# x_test = getVideoAvgBrightnesses(VIDEOPATH + str(48) + ".mp4")
# predictions = new_model.predict([x_test])
# print(predictions[0])

# Write output signal to file
# with open("output.txt", "w") as txt_file:
#     for line in predictions[0]:
#         txt_file.write(str(line)+'\n')

# Play sound - WARNING may be very loud
# sd.play(predictions[0],44100)
# sd.wait()