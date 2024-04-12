import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import joblib

df=pd.read_csv('parkinsons.data')

# *******Preprocessing*******

print(df)
print(df.shape)
print(df.isnull().sum())
print(df.info())

# Check for values similar to null values or question marks
# for i in df.columns:
#     print('*****'*10,i,'******'*10)
#     print(set(df[i]))
#     print()

print(df['status'].value_counts())

X=df.drop(['status','name'],axis=1)
y=df['status']
print(X.columns)
ros=RandomOverSampler()
x_ros,y_ros=ros.fit_resample(X,y)
print(x_ros)

scaler=MinMaxScaler()
x=scaler.fit_transform(x_ros)
y=y_ros

print(x)
print(y)

# ********Preprocessing Done*********

# ********Feature Extraction using PCA*********
pca=PCA(n_components=0.95)
x_pca=pca.fit_transform(x)

print(x.shape)
print(x_pca.shape)

# Split the data into train & test
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.1, random_state=42)


# Model Building

# Model1-LogisticRegression
classifier1=LogisticRegression(random_state=42)

param_grid1 = {
    'C': np.arange(0.1,0.5,0.1),
    'solver': ['liblinear', 'saga']
}

grid_search_lr=GridSearchCV(estimator=classifier1, param_grid=param_grid1, cv=5)
grid_search_lr.fit(X_train,y_train)

# Train model with best set of parameters
lr=LogisticRegression(**grid_search_lr.best_params_, random_state=42)
lr.fit(X_train,y_train)

# Prediction
y_pred1=lr.predict(X_test)
# Accuracy
accuracy_LR=accuracy_score(y_test,y_pred1)


# Model2-DecisionTree
classifier2=DecisionTreeClassifier(random_state=42)
dt=classifier2.fit(X_train,y_train)
# Prediction
y_pred2=classifier2.predict(X_test)
# Accuracy
accuracy_DT=accuracy_score(y_test,y_pred2)


# Model3-RandomForest_gini
classifier3=RandomForestClassifier(criterion='gini',random_state=42)
rfg=classifier3.fit(X_train,y_train)
# Prediction
y_pred3=classifier3.predict(X_test)
# Accuracy
accuracy_RFG=accuracy_score(y_test,y_pred3)


# Model4-RandomForest_entropy
classifier4=RandomForestClassifier(criterion='entropy',random_state=42)
rfe=classifier4.fit(X_train,y_train)
# Prediction
y_pred4=classifier4.predict(X_test)
# Accuracy
accuracy_RFE=accuracy_score(y_test,y_pred4)


# Model5-SVM
model_svm=SVC()
svm=model_svm.fit(X_train,y_train)
# Prediction
y_pred5=model_svm.predict(X_test)
# Accuracy
accuracy_SVM=accuracy_score(y_test,y_pred5)


# Model6-KNearestNeighbors
model_knn=KNeighborsClassifier()
knn=model_knn.fit(X_train,y_train)
# Prediction
y_pred6=knn.predict(X_test)
# Accuracy
accuracy_KNN=accuracy_score(y_test,y_pred6)


# Model7-GaussianNaiveBayes
model_gnb=GaussianNB()
gnb=model_gnb.fit(X_train,y_train)
# Prediction
y_pred7=gnb.predict(X_test)
# Accuracy
accuracy_GNB=accuracy_score(y_test,y_pred7)


# Model8-BernoulliNaiveBayes
model=BernoulliNB()
bnb=model.fit(X_train,y_train)
# Prediction
y_pred8=bnb.predict(X_test)
# Accuracy
accuracy_BNB=accuracy_score(y_test,y_pred8)

# Combine all models by using votingclassifier model
evc=VotingClassifier(estimators=[('lr',lr),('DT',dt),('RFI',rfg),('RFE',rfe),
                                 ('SVC',svm),('KNN',knn),('GNB',gnb),('BNB',bnb)],
                                 voting='hard',flatten_transform=True)

model_evc=evc.fit(X_train,y_train)
# Prediction
pred_evc=evc.predict(X_test)
# Accuracy
accuracy_EVC=accuracy_score(y_test,pred_evc)

list1=['Logistic Regression', 'Decision Tree', 'Random Forest Gini', 'Random Forest Entropy', 
       'Suppot Vector', 'K Nearest Neighbors', 'GaussianNB', 'BernoulliNB', 'Voting Classifier']

list2=[accuracy_LR,accuracy_DT,accuracy_RFG,accuracy_RFE,accuracy_SVM,accuracy_KNN,
       accuracy_GNB,accuracy_BNB,accuracy_EVC]

list3=[classifier1,classifier2,classifier3,classifier4,model_svm,model_knn,model_gnb,
       model,model_evc]

df_accuracy=pd.DataFrame({'Method Used':list1,'Accuracy':list2})
print(df_accuracy)

# Other evaluation method
# RandomFroest entropy
y_pred4_train=classifier4.predict(X_train)
y_pred4_test=classifier4.predict(X_test)


# KNN
pred_knn_train=model_knn.predict(X_train)
pred_knn_test=model_knn.predict(X_test)

print('*********RandomForest*********')
print(confusion_matrix(y_train,y_pred4_train))
print('******'*6)
print(confusion_matrix(y_test,y_pred4_test))

print('*********KNN*********')
print(confusion_matrix(y_train,pred_knn_train))
print('******'*6)
print(confusion_matrix(y_test,pred_knn_test))


print('*********RandomForest*********')
print(classification_report(y_train,y_pred4_train))
print('******'*6)
print(classification_report(y_test,y_pred4_test))

print('*********KNN*********')
print(classification_report(y_train,pred_knn_train))
print('******'*6)
print(classification_report(y_test,pred_knn_test))


# Saving th model using joblib
joblib.dump(model_evc,'model_EVC.pkl')