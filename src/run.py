import os
import pandas as pd
import audiofile
from amazing import functions as f
import tensorflow as tf
import sounddevice as sd
import numpy as np
from scipy import signal
from scipy.io import wavfile

# print("Reading data csv...")
# currentDir = os.getcwd()
# audioData = "audio.csv"
# audioPath = os.path.join(currentDir, audioData)

# audioTable = f.constructTable(audioPath)

# avgHeartRates = []
# for hrRange in audioTable['heartrate'].values:
#     bounds = hrRange.split("-")
    
#     avgHeartRates.append((int(bounds[0]) + int(bounds[1]))/2)


# # correlation between sleep duration & heart rate
# x = avgHeartRates
# y = audioTable['sleepDuration'].values

# f.correlate(x,y)


# # for i in range(len(audioTable['id'].values)):

## DL Model training code here
print("DL Model Training...")
training_data = []

# Set your personal data path here:
VIDEO_PATH = os.getcwd() + "\\src\\TrainingData\\Video\\"
AUDIO_PATH = os.getcwd() + "\\src\\TrainingData\\Audio\\"

CATEGORIES = ["60-70","70-80","80-90","90-100","100-110","110-120","120-130","130-140","140-150","150-160"]
Classifications = ["80-90", "80-90", "120-130", "80-90", "80-90", "100-110", "140-150", "90-100","90-100","90-100","80-90","80-90","130-140","80-90","100-110","110-120","120-130","90-100","130-140","110-120","70-80","60-70","60-70","110-120","90-100","80-90","80-90","110-120","70-80","110-120","70-80","100-110","150-160","100-110","60-70","100-110","70-80","130-140","100-110","100-110","100-110","130-140","70-80","90-100","70-80","70-80","70-80","150-160","110-120"]
print(len(Classifications))
X = []
Y = []

# Process Video
for i in range(0,49):
    print(f"Processing sample {i+1}")
    avg_brightnesses = f.getVideoAvgBrightnesses(VIDEO_PATH + str(i+1) + ".mp4")
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
x_test = f.getVideoAvgBrightnesses(VIDEO_PATH + str(9) + ".mp4")
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