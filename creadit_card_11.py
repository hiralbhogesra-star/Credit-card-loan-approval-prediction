# ============================================================
# Project Name : Energy Consumption Prediction using Support Vector Machine (SVM)
# Tool         : PyCharm
# Purpose      :  To predict whether a credit card application will be Approved / Rejected using
# Random Forest (Best Algorithm) and Logistic Regression, then compare both models
# using Cross-Validation, K-Fold Cross Validation, and Hyperparameter Tuning.
# ============================================================


#----------------------- IMPORT LIBRARIES -------------------#
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

#-------------------- STEP 1: LOAD DATASET -------------------#
data=pd.read_csv("credit_card.csv")
print("\n Original Dataset:\n",data)

#-------------------- STEP 2 :  CHECK AND HANDLE MISSING(NULL) VALUE ------------------#

print("\nMissing Values:\n", data.isnull().sum())

#-------------------- STEP 3 : REMOVE DUPLICATE VALUES ---------------------#

data = data.drop_duplicates()
print("Total Records:", len(data))

#-------------------- STEP 4 : SELECT INPUT FEATURE AND OUTPUT LABEL(APPROVED/REJECTED) ---------------------#

X = data[['Age', 'Income', 'CreditScore', 'Debt', 'YearsEmployed', 'PriorDefault', 'Employed']]
y = data['Approved']

#--------------------- STEP 5 :  SPLIT DATASET INTO TRAIN AND TEST(TRAIN-TEST SPLIT) --------------------#

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

#-------------- ENCODE LABEL ---------------#
le = LabelEncoder()
y = le.fit_transform(y)

print("Classes:", le.classes_)

#----------------------- STEP 6 : TRAIN LOGISTIC REGRESSION MODEL -----------------------#

lr=LogisticRegression(max_iter=1000)
lr.fit(X_train,y_train)

#-------------------- STEP 7 : PREDICT APPROVAL STATUS USING LOGISTIC REGRESSION -------------------#

lr_pred=lr.predict(X_test)

#--------------------- STEP 8 :  EVALUATE LOGISTIC REGRESSION USING-ACCURACY,PREDICTION,RECALL,F1-SCORE,CONFUSION MATRIX -----------------------#

print("Logistic regression Accuracy:",accuracy_score(y_test,lr_pred))
print("confusion matrix:\n",confusion_matrix(y_test,lr_pred))
print("classfication Report:\n",classification_report(y_test,lr_pred,target_names=lr.classes_))

#----------------------- STEP 9 : TRAIN RANDOM FOREST CLASSIFIER(BASE MODEL) --------------------#

rf=RandomForestClassifier(random_state=42)
rf.fit(X_train,y_train)

#---------------------- STEP 10 : PREDICT APPROVAL STATUS USING RANDOM FOREST ---------------------#

rf_pred=rf.predict(X_test)

#----------------------- STEP 11 : EVALUATE RANDOM FOREST USING-ACCURACY,PREDICTION,RECALL,F1-SCORE,CONFUSION MATRIX --------------------3

print("Random forest Accuracy:",accuracy_score(y_test,rf_pred))
print("confusion matrix:\n",confusion_matrix(y_test,rf_pred))
print("classification Report:\n",classification_report(y_test,rf_pred,target_names=rf.classes_))

#----------------------- STEP 12 : APPLY CROSS-VALIDATION(CV=5) FOR BOTH MODELS AND CALCULATE AVERAGE ACCURACY --------------------#

lr_cv=cross_val_score(lr,X,y,cv=5)
rf_cv=cross_val_score(rf,X,y,cv=5)

print("Logistic regression cv Accuracy:",lr_cv.mean())
print("Random forest cv Accuracy",rf_cv.mean())

#-------------------- STEP 13 : APPLY K-FOLD CROSS VALIDATION(5 SPLITS) FOR BOTH MODELS AND CALCULATE AVERAGE ACCURACY ----------------#

kf = KFold(n_splits=5, shuffle=True, random_state=42)
lr_kf=cross_val_score(lr,X,y,cv=kf)
rf_kf=cross_val_score(rf,X,y,cv=kf)

print("LR K-Fold Accuracy:", lr_kf.mean())
print("RF K-Fold Accuracy:", rf_kf.mean())

#----------------------- STE 14 : ERFORM HYPERPARAMETER TUNING(GridSearchCV) FOR RANDOM FOREST TO FIND BEST PARAMETER ----------------------#

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42),
                    param_grid, cv=5, scoring='accuracy')

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

#--------------------- STEP 15 : TRAIN THE BEST TUNED RANDOM FOREST MODEL ---------------------#

best_rf = grid.best_estimator_
best_rf.fit(X_train, y_train)

#------------------- STEP 16 : PREDICTION CHART(ACTUAL VS PREDICTED) -----------------------#

plt.figure(figsize=(6,4))
plt.plot(y_test.values, label="Actual", marker='o')
plt.plot(best_rf.predict(X_test), label="Predicted", marker='X')
plt.legend()
plt.title("Actual vs Predicted Approval")
plt.show()

#------------------ STEP 17 : PREDICT APPROVAL RESULT FOR A NEW APPLICANT --------------------#

new_applicant = pd.DataFrame([[35, 50000, 720, 2000, 5, 0, 1]],
    columns=['Age', 'Income', 'CreditScore', 'Debt', 'YearsEmployed', 'PriorDefault', 'Employed'])

prediction = best_rf.predict(new_applicant)

if prediction[0] == 1:
    print("\nNew Applicant Result: APPROVED")
else:
    print("\nNew Applicant Result: REJECTED")