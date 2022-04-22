# This is the analysis as well
# Import the required libraries here
import pandas as pd
import numpy as np

# Import the graphical packages
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msn
import shap

# Import the machine learning libraries here 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# Import model assessment libraries here 
from sklearn.metrics import classification_report, confusion_matrix

# Model improvement and parameter optimisation
from sklearn.model_selection import GridSearchCV

# Import the data
#df = pd.read_csv("ilrData.csv")
df = pd.read_csv("final_data.csv")

#############################  Exp;loratory data analysis ##########################
# Explore the data here 
df.head(2)
#df.tail()
#df.shape
#df.info()
#df.describe()
#df.isna().sum()
#df.shape

# Visualise all the missing data 
# A bar graph of the missing data
#msn.heatmap(df);
msn.matrix(df)
#msn.dendrogram(df)
# Save the figure to the figure folder
#plt.savefig("Figure/missingData.png")
plt.savefig("Figure/enginearedFeature.jpeg")

# Perform some basic visualisation 
sns.countplot(x="Gender", data=df)
sns.despine()
plt.savefig("Figure/GenderCount.jpeg")

plt.Figure(figsize=(10,5))
sns.boxplot(y = "Age",data=df);
sns.despine()
plt.savefig("Figure/AgeBoxplot.jpeg")

# A histogram showing the distribution of the age here 
sns.histplot(data=df, x='Age', bins=50, kde=True);
sns.despine()
plt.savefig("Figure/AgeDistribution.jpeg")

# Create a new feature that looks at only syncope or others as the concern before 
# Symptoms presented before the 
lst = []
for i in df["Indication"]:
    if "syncope" in i.lower() or "collapse" in i.lower():
        lst.append("Syncope")
    else:
        lst.append("Other")

# Underlying cause of the Syncope 
lst1 = []
lst2 = "SVT AF NSVT flutter tachycardia bradycardia AT VT pause SR(SVEs) AV-block".split()
for item in df["Solus-Diagnosis"]:
    if item in lst2:
        lst1.append("arrhythmia")
    else:
        lst1.append("others")
        
# Feature enginearing of the Daignosis (Success of the ILR)
diagnosis = []
for j in df["Solus_Diagnosis"]:
    if j.lower() == "no abnormal findings" or j.lower() == "no notes":
        diagnosis.append('No')
    elif j.lower() == "undersensing" or j.lower() == "signal issue":
        diagnosis.append("Failure")
    else:
        diagnosis.append("Yes")   
        
ageGroup = []
for age in df["Age"]:
    if age <= 30:
        ageGroup.append("≤ 30")
    elif age > 30 and age <= 64:
        ageGroup.append("31 - 64")
    else:
        ageGroup.append("≥ 65")

# Add teh engineared features to the dataframe here
df["Cause"] = lst1  
df["Concern"] = lst
df["diagnosis"] = diagnosis
df["AgeGroup"] = ageGroup

# Create the dummy variables here for categorical variables for further analysis in Scipy and sklearn
df["Gender_male_1"] = pd.get_dummies(df['Gender'], drop_first=True)
df["Symptoms_yes_1"] = pd.get_dummies(df['Symptoms'], drop_first=True)
df["Presentation"] = pd.get_dummies(df["Concern"], drop_first=True)

# Export the processed file here for analysis in r
#df.to_csv("ilr_processed.csv")

Diagnosis = []
af = "AT, AF PAF".split()
svt = "NSVT, NSVT , SVT,".split()
for case in df["Solus_Diagnosis"]:
    if case.strip().upper() in af:
        Diagnosis.append("AF")
    elif case.upper() in svt:
        Diagnosis.append("SVT")
    elif case.lower() == "flutter":
        Diagnosis.append("Atrial Flutter")
    elif case == 'High grade AV block and intermittent sinus arrest':
        Diagnosis.append("AV-block")
    else:
        Diagnosis.append(case)
        
arrhythmia = []
for cause in df["Cause"]:
    if cause == "arrhythmia":
        arrhythmia.append(1)
    else:
        arrhythmia.append(0)

df['Diagnosis'] = Diagnosis
df['arrhythmia'] = arrhythmia
df.head()
# Export the processed file here for analysis in r
#df.to_csv("ilr_processed.csv")

# Prepare the data for machine learning here 
x = df[['Age','Gender_male_1','Presentation']]
X = df[['Age','Gender_male_1','Presentation']].values
y = df['arrhythmia'].values

# Build the model here 
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=101)

# Initialise the classifiers here
svm = SVC() 
rf = RandomForestClassifier()
lgr = LogisticRegression()

# Build models here 
svm.fit(X_train, y_train)
rf.fit(X_train, y_train)
lgr.fit(X_train, y_train)

# Predict with the models built here
y_predict_svm = svm.predict(X_test)
y_predict_rf = rf.predict(X_test)
y_predict_lgr = lgr.predict(X_test)

# Implement a grid search to optimse the SVM predictors 
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)

# May take awhile!
grid.fit(X_train,y_train)

# print the best paramters here
print (grid.best_params_)

# Rerun the svm with the best parameters here 
grid_predictions = grid.predict(X_test)

plt.figure(figsize=(7,7))
sns.heatmap(confusion_matrix(y_test,grid_predictions), annot=True, cmap="coolwarm");
sns.heatmap(confusion_matrix(y_predict_rf, y_test),annot=True, cmap="coolwarm");
sns.heatmap(confusion_matrix(y_predict_lgr, y_test), annot=True,cmap="coolwarm");

feat_importances = pd.DataFrame(rf.feature_importances_,index=x.columns,columns=["Importance"])
feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
feat_importances.plot(kind='bar', figsize=(8,6));

print (classification_report(y_test, grid_predictions))
print (classification_report(y_test, y_predict_rf))
print (classification_report(y_test, y_predict_lgr))



















































