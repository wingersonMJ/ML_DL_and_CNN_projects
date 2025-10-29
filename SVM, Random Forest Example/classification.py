import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# pre-processing stuff
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# needed for rf and accuracy evaluation 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# needed to get roc curve
from sklearn.metrics import roc_curve, roc_auc_score

# svc 
from sklearn.svm import SVC

# log regression
from sklearn.linear_model import LogisticRegression

###############
# Load in data 
df = pd.read_csv('./SVM, Random Forest Example/Data/brain_stroke.csv')
df.columns

# check missing data - there is none
print(df.isna().any())

# check categorical levels for each var
print(df['gender'].unique()) 
print(df['heart_disease'].unique()) 
print(df['ever_married'].unique()) 
print(df['work_type'].unique()) 
print(df['Residence_type'].unique()) 
print(df['smoking_status'].unique()) 
print(df['stroke'].unique()) 
print(df['heart_disease'].unique()) 

# One-hot encode categorical vars that are not already binary
work = pd.get_dummies(df['work_type'], drop_first=True)
smoking = pd.get_dummies(df['smoking_status'], drop_first=True)

df.drop(['work_type', 'smoking_status'], axis = 1, inplace = True)

df = pd.concat([df, work, smoking], axis = 1)

# Use lable encoder to get categoricals to numerics
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['ever_married'] = le.fit_transform(df['ever_married'])
df['Residence_type'] = le.fit_transform(df['Residence_type'])

# Standardize numerics
scaler = StandardScaler()
columns_to_standardize = ['age', 'hypertension', 'avg_glucose_level', 'bmi']

df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
df.head()



##########################
## Random Forests
# separate features and target
X = df.drop('stroke', axis=1) 
y = df['stroke'] 

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1989)

# initialize and fit
rf = RandomForestClassifier(random_state=1989)
rf.fit(X_train, y_train)

# predict on the train data
y_train_pred = rf.predict(X_train)

# evaluate 
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("\nClassification Report (Training Data):\n", classification_report(y_train, y_train_pred))

# predict on the test data
y_test_pred = rf.predict(X_test)

# evaluate
print("\nTest Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report (Test Data):\n", classification_report(y_test, y_test_pred))

# ROC curve
y_test_prob = rf.predict_proba(X_test)[:, 1]

# ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
auc_score = roc_auc_score(y_test, y_test_prob)

# plot roc and auc
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Random Forest on Test Data')
plt.legend(loc='lower right')
plt.savefig("./SVM, Random Forest Example/figs/random_forest_roc_auc.png")
plt.show()

# plot probabilities
plt.figure(figsize=(10, 8))
sns.kdeplot(y_test_prob[y_test == 0], label='Class 0 (No Stroke)', fill=True, alpha=0.5, linewidth=2)
sns.kdeplot(y_test_prob[y_test == 1], label='Class 1 (Stroke)', fill=True, alpha=0.5, linewidth=2)
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.title('Predicted Probabilities for Random Forests on Test Data')
plt.legend(loc='upper center')
plt.savefig("./SVM, Random Forest Example/figs/random_forest_kde.png")
plt.show()



#####################
## Support vector machine!
# define and fit
svc = SVC(C=1.0, kernel='rbf', gamma='scale', class_weight='balanced', probability=True)
svc.fit(X_train, y_train)

# training data
y_train_pred = svc.predict(X_train)
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("\nClassification Report (Training Data):\n", classification_report(y_train, y_train_pred))

# test data
y_test_pred = svc.predict(X_test)
print("\nTest Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report (Test Data):\n", classification_report(y_test, y_test_pred))

# probabilities for the ROC curve
y_test_prob = svc.predict_proba(X_test)[:, 1]

# ROC and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
auc_score = roc_auc_score(y_test, y_test_prob)

# plot roc
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for SVM on Test Data')
plt.legend(loc='lower right')
plt.savefig("./SVM, Random Forest Example/figs/svm_roc_auc.png")
plt.show()

# kde plot
plt.figure(figsize=(10, 8))
sns.kdeplot(y_test_prob[y_test == 0], label='Class 0 (No Stroke)', fill=True, alpha=0.5, linewidth=2)
sns.kdeplot(y_test_prob[y_test == 1], label='Class 1 (Stroke)', fill=True, alpha=0.5, linewidth=2)
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.title('Predicted Probabilities for SVM on Test Data')
plt.legend(loc='upper center')
plt.figure("./SVM, Random Forest Example/figs/svm_kde.png")
plt.show()


##################
## Logistic regression
# define and fit
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=1989)
log_reg.fit(X_train, y_train)

# training dat
y_train_pred = log_reg.predict(X_train)
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("\nClassification Report (Training Data):\n", classification_report(y_train, y_train_pred))

# testing dat 
y_test_pred = log_reg.predict(X_test)
print("\nTest Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report (Test Data):\n", classification_report(y_test, y_test_pred))

y_test_prob = log_reg.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
auc_score = roc_auc_score(y_test, y_test_prob)

# roc curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Logistic Regression on Test Data')
plt.legend(loc='lower right')
plt.savefig("./SVM, Random Forest Example/figs/lreg_roc_auc.png")
plt.show()

# kde
plt.figure(figsize=(8, 6))
sns.kdeplot(y_test_prob[y_test == 0], label='Class 0 (No Stroke)', fill=True, alpha=0.5, linewidth=2)
sns.kdeplot(y_test_prob[y_test == 1], label='Class 1 (Stroke)', fill=True, alpha=0.5, linewidth=2)
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.title('Predicted Probabilities for Logistic Regression on Test Data')
plt.legend(loc='upper center')
plt.savefig("./SVM, Random Forest Example/figs/lreg_kde.png")
plt.show()