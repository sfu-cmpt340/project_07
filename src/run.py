from amazing import functions as f
import pickle

# load models
with open('src/pretrained_models/predict_heart_rate_from_video.pkl', 'rb') as fid:
    clf_video = pickle.load(fid)

with open('src/pretrained_models/predict_heart_rate_from_feature.pkl', 'rb') as fid:
    clf_feature = pickle.load(fid)

# Display GUI
f.getMainSelectionPage(clf_feature, clf_video)

