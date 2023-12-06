import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import functions as f
import os


print("Reading data csv...")
video_data_path = os.path.join(os.getcwd() + "\\src\\TrainingData\\video.csv")
data_table = f.constructTable(video_data_path) # overall table from csv

# full training data not included in final submission (sample dataset provided in folder "sample_training_data")
VIDEO_PATH = os.path.join(os.getcwd() + "\\src\\TrainingData\\Video\\") #EX. os.getcwd() + "\\src\\TrainingData\\Video\\"

# DL Model training code here
print("DL Model Training...")
# HR zones slightly modified from : https://www.polar.com/blog/running-heart-rate-zones-basics/
CATEGORIES = ["Zone 1 - Resting/Very light - 60-90 bpm", "Zone 2 - Light - 90-110 bpm", "Zone 3 - Moderate - 110-130 bpm", "Zone 4 - Hard - 130-160 bpm"]
heartrates = data_table['heartrate']
Classifications = [f.getHrZoneIndex(x) for x in heartrates]

# Define training data
X = []
Y = []

# Selected videos that have the most easily seen peaks
# good videos (not equal amount for each HR zone)
# training_videos = [1,3,5,6,7,10,12,17,21,23,29,34,35,37,39,43,45,46,54,64]
# good + decent videos (not equal amount for each HR zone)
# training_videos1 = [1,3,4,5,6,7,9,10,12,14,15,17,21,22,23,25,27,29,31,32,34,35,36,37,39,41,43,45,46,47,49,64,54]

# 3 videos for each of the 4 HR zones
best_training_videos = [1,5,12,4,6,9,3,17,49,7,33,63]

# Process Video into average brightnesses per frame
for i in best_training_videos:
     print(f"Processing sample {i}")
     avg_brightnesses, fps = f.getVideoAvgBrightnesses(VIDEO_PATH + str(i) + ".mp4")
     # Apply band pass filter to average brightnesses to make it easier to distinguish peaks
     filtered_brightness = f.getBandpassFilter(avg_brightnesses, 0.5, 2.5, fps)
     # Construct training data
     X.append(filtered_brightness)
     Y.append(Classifications[i-1])

# test samples to get score (not needed for training)
# x_test = []
# y_test = []
# for i in [37,29,35,23,10,45,46,54,64]:
#     avg_brightnesses, fps = f.getVideoAvgBrightnesses(VIDEO_PATH + str(i) + ".mp4")
#     # Apply band pass filter to average brightnesses to make it easier to distinguish peaks
#     filtered_brightness = f.getBandpassFilter(avg_brightnesses, 0.5, 2.5, fps)
#     x_test.append(filtered_brightness)
#     y_test.append(Classifications[i-1])

# fine tune parameters
clf = KNeighborsClassifier()
param_grid = [{
    "n_neighbors" : [1,3,5], "weights": ["uniform","distance"], "algorithm" : ["auto", "ball_tree", "kd_tree", "brute"]
}]

grid_search = GridSearchCV(clf, param_grid, cv=2, scoring="accuracy", return_train_score=True, verbose=10)
grid_search.fit(X, Y)

video_clf = grid_search.best_estimator_

print("Saving model...")
with open('src/pretrained_models/predict_heart_rate_from_video.pkl', 'wb') as fid:
    pickle.dump(video_clf, fid)  