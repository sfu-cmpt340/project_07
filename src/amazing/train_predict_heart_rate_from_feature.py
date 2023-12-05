import os
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np

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
# *Close figure to continue

# Defining ranges for classification
bins = [0, 60, 70, 80, 90, 100, 120, 140, 160, 200]
labels = ['Very Low (0-60)', 'Low (60 - 70)', 'Medium Low (70-80)', 'Medium (80-90)', 'Medium High (90-100)', 'High (100-120)', 'Very High (120-140)', 'Extremely High (140-160)', 'Are You Ok (160+)']
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
# *Close figure to continue

print("Saving model...")
with open('predict_heart_rate_from_feature.pkl', 'wb') as fid:
    pickle.dump(clf, fid)