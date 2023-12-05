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

## MODEL TRAINING CODE: Video DL Model training code here -----------------------------------------------------------------------------------
# NOTE: uncomment this to train the model
# TRAINING_VIDEO_PATH = os.path.join(os.getcwd() + "\\src\\TrainingData\\Video\\")
# print("Video DL Model Training...")
# CATEGORIES = ["60-70","70-80","80-90","90-100","100-110","110-120","120-130","130-140","140-150","150-160"]
# heartrates = data_table['heartrate']
# Classifications = [f.getHrRange(x) for x in heartrates]

# # Define training data
# X = []
# Y = []

# # Process Video
# NOTE: Change the range based on the number of videos
# for i in range(0,49):
#      print(f"Processing sample {i+1}")
#      avg_brightnesses, fps = f.getVideoAvgBrightnesses(TRAINING_VIDEO_PATH + str(i+1) + ".mp4")
#      # Apply band pass filter to average brightnesses to make it easier to distinguish peaks
#      filtered_brightness = f.getBandpassFilter(avg_brightnesses, 0.5, 2.5, fps)
#      # Construct training data
#      X.append(filtered_brightness)
#      Y.append(CATEGORIES.index(Classifications[i]))

# clf = svm.SVC(gamma=0.001, C=100.)
# clf.fit(X, Y)

# with open('predict_heart_rate_from_video.pkl', 'wb') as fid:
#     pickle.dump(clf, fid)  

# load model
with open('predict_heart_rate_from_video.pkl', 'rb') as fid:
    clf_loaded = pickle.load(fid)

## MODEL TRAINING CODE: Predicting BPM range from Ys (Classification) -----------------------------------------------------------------------------------
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

# Checking Feature importance 
plt.figure(figsize=(10,6))
importance = clf.feature_importances_
idxs = np.argsort(importance)
plt.title("Feature Importance")
plt.barh(range(len(idxs)),importance[idxs],align="center")
plt.yticks(range(len(idxs)),[feature_col_tree[i] for i in idxs])
plt.xlabel("Random Forest Feature Importance")
plt.show()

with open('predict_heart_rate_from_feature.pkl', 'wb') as fid:
    pickle.dump(clf, fid)

# load model
with open('predict_heart_rate_from_feature.pkl', 'rb') as fid:
    clf = pickle.load(fid)

## MAIN GUI CALLER -----------------------------------------------------------------------------------
f.getMainSelectionPage(clf, clf_loaded)
