import os
import pandas as pd
from amazing import functions as f
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, ifft
from sklearn import svm
import pickle
import seaborn as sn
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import PySimpleGUI as sg

print("Reading data csv...")
video_data_path = os.path.join(os.getcwd(), "video.csv")
data_table = f.constructTable(video_data_path) # overall table from csv

x = data_table[['id','heartrate']] # one table for id and heartrate
y = data_table.drop('heartrate', axis=1) # another excluding heartrate

# printing tables
print("-----X-----\n",x.head())
print("-----Y-----\n",y.head())

## Grabbing BPM from video -----------------------------------------------------------------------------------
# Help: http://www.ignaciomellado.es/blog/Measuring-heart-rate-with-a-smartphone-camera
# Set your personal data path here:
VIDEO_PATH = os.path.join(os.getcwd() + "\\src\\TrainingData\\Video\\") #EX. os.getcwd() + "\\src\\TrainingData\\Video\\"
AUDIO_PATH = os.path.join(os.getcwd() + "\\src\\TrainingData\\Audio\\") #EX. os.getcwd() + "\\src\\TrainingData\\Audio\\"
print (VIDEO_PATH)
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
    ## Plotting for easier debugging
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
clf.fit(X, Y)

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
## Predicting BPM range from Ys (Classification) -----------------------------------------------------------------------------------
# Cite: https://www.kaggle.com/code/durgancegaur/a-guide-to-any-classification-problem
df = pd.read_csv(os.getcwd()+"\\video.csv")
df = df.drop(['ID', 'DATE'], axis=1)
df = df.set_axis(['RATE', 'STATE', 'ACTIVITY', 'BMI', 'AGE', 'CAFFEINE INTAKE', 'SLEEP DURATION', 'GENDER', 'FITNESS LEVEL'], axis=1)

# Correlation matrix
print("__________Correlation Matrix_______________")
print(df.corr())

# Correlation matrix in Heatmap
sn.heatmap(df.corr())
plt.title("Correlation Matrix (Heatmap)")
plt.show()

# Defining ranges for classification
bins = [0, 60, 70, 80, 90, 100, 120, 140, 160]
labels = ['Very Low (0-60)', 'Low (60 - 70)', 'Medium Low (70-80)', 'Medium (80-90)', 'Medium High (90-100)', 'High (100-120)', 'Very High (120-140)', 'Extremely High (140-160)']
df['RATE CATEGORY'] = pd.cut(df['RATE'], bins=bins, labels=labels, right=False)
# df.head(100)

# Change categorical variables into 0/1
target="RATE CATEGORY"
df_tree = df.apply(LabelEncoder().fit_transform)
feature_col_tree=df_tree.columns.to_list()
feature_col_tree.remove(target)
feature_col_tree.remove("RATE")

# feature_col_tree.head()
categorical_columns = ['STATE', 'ACTIVITY', 'CAFFEINE INTAKE', 'GENDER', 'FITNESS LEVEL']
df_nontree=pd.get_dummies(df,columns=categorical_columns,drop_first=False)
y=df_nontree[target].values
df_nontree.drop("RATE",axis=1,inplace=True)
df_nontree=pd.concat([df_nontree,df[target]],axis=1)
df_nontree.head()

# Classifier method: Random forest
acc_RandF=[]
kf=model_selection.StratifiedKFold(n_splits=2) # test with different n_splits when we finish updating the dataset

for fold , (trn_,val_) in enumerate(kf.split(X=df_tree,y=y)):
    
    X_train=df_tree.loc[trn_,feature_col_tree]
    y_train=df_tree.loc[trn_,target]
    
    X_valid=df_tree.loc[val_,feature_col_tree]
    y_valid=df_tree.loc[val_,target]
    
    clf=RandomForestClassifier(n_estimators=200,criterion="entropy")
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_valid)

print(f"Fold: {fold}")

# Checking Feature importance 
plt.figure(figsize=(10,6))
importance = clf.feature_importances_
idxs = np.argsort(importance)
plt.title("Feature Importance")
plt.barh(range(len(idxs)),importance[idxs],align="center")
plt.yticks(range(len(idxs)),[feature_col_tree[i] for i in idxs])
plt.xlabel("Random Forest Feature Importance")
plt.show()

# Display GUI
f.getMainSelectionPage(clf)

## DL Video -> Audio Prediction Model training code here -----------------------------------------------------------------------------------
# print("DL Model Training...")
# training_data = []

# # Process Video
# for i in range(1,50):
#     avg_brightnesses, _ = f.getVideoAvgBrightnesses(VIDEO_PATH + str(i) + ".mp4")
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
# x_test, _ = f.getVideoAvgBrightnesses(VIDEO_PATH + str(48) + ".mp4")
# predictions = new_model.predict([x_test])
# print(predictions[0])

# # Write output signal to file
# with open("output.txt", "w") as txt_file:
#     for line in predictions[0]:
#         txt_file.write(str(line)+'\n')

# # Play sound - WARNING may be very loud
# sd.play(predictions[0],44100)
# sd.wait()
