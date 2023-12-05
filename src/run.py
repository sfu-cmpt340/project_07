import os
from amazing import functions as f
import pickle

print("Reading data csv...")
video_data_path = os.path.join(os.getcwd(), "video.csv")
data_table = f.constructTable(video_data_path) # overall table from csv

x = data_table[['id','heartrate']] # one table for id and heartrate
y = data_table.drop('heartrate', axis=1) # another excluding heartrate

# printing tables
print("-----X-----\n",x.head())
print("-----Y-----\n",y.head())

# load models
with open('predict_heart_rate_from_video.pkl', 'rb') as fid:
    clf_video = pickle.load(fid)

with open('predict_heart_rate_from_feature.pkl', 'rb') as fid:
    clf_feature = pickle.load(fid)

# Display GUI
f.getMainSelectionPage(clf_feature, clf_video)

