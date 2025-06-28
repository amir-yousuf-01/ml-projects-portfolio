

import jj
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from pandas.io.formats.style_render import jinja2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
import warnings
warnings.filterwarnings("ignore")
import jinja2 as jj


df = pd.read_csv("synthetic_health_lifestyle_dataset.csv")
df.info()
print(df.shape)
print(df.sample())


# Data Cleaning & Handling
print(df.head())
df["Exercise_Freq"] = df["Exercise_Freq"].fillna(df["Exercise_Freq"].mode()[0])
df["Alcohol_Consumption"] = df["Alcohol_Consumption"].fillna(df["Alcohol_Consumption"].mode()[0])
exercise_map = {
    "Daily":2,
    "1-2 times/week ":1,
    "3-5 times/week ":0
}
df["Exercise_Freq"]= df["Exercise_Freq"].map(exercise_map)
df.drop(columns=["ID"], inplace=True)
print(df.isna().sum())
print(df.duplicated().sum())



# EDA & Visualization



# Average BMI Classification By Gender
classfi_bmi = df.groupby ("Gender")["BMI"].mean().reset_index()
print(classfi_bmi)

plt.figure(figsize=(12,6))
sns.set_style('whitegrid')
sns.scatterplot(data=classfi_bmi, x= "Gender",y="BMI",color = "red",edgecolor="black")
plt.title("Average BMI Classification By Gender",size = 13)
plt.show()




# Smoker & Non-sMOLER Classification Based on Gender
classifi_sm=df.groupby(["Gender","Smoker"]).size().reset_index(name="Count")
classifi_sm.style.background_gradient(cmap="Reds")
print(classifi_sm)

plt.figure(figsize=(12,6))
sns.set_style('whitegrid')
sns.barplot(data=classifi_sm,x="Smoker",y="Count",hue="Gender",palette="Accent",edgecolor="black")
plt.title("Smoker & Non Smoker Classification Based On Gender ",size=13)
plt.show()





# Stress Level Classification by Gender
classifi_str = df.groupby("Gender")["Stress_Level"].sum().reset_index()
classifi_str.style.background_gradient(cmap="Oranges")
print(classifi_str)

plt.figure(figsize=(12,6))
sns.set_style("whitegrid")
sns.barplot(data=classifi_str,x="Gender",y="Stress_Level",palette="Set2",edgecolor="black")
plt.title("Stress Level Classification By Gender ",size=13)
plt.show()




# Diet Quality & Alcohol Consumption By Gender & Age

classifi_di = df.groupby(["Gender","Age","Diet_Quality","Alcohol_Consumption"]).size().reset_index(name="Count")
print(classifi_di)

plt.figure(figsize=(12,6))
sns.set_style("whitegrid")
sns.barplot(data= classifi_di,x="Diet_Quality",y="Count",hue= "Gender",ci=None,palette="Dark2",edgecolor="black")
plt.title("Diet Quality & Alcohol Consumption By Gender & Age",size=13)
plt.show()

plt.figure(figsize=(12,6))
sns.set_style("whitegrid")
sns.barplot(data=classifi_di,x ="Diet_Quality",y="Count",hue="Gender",ci=None,palette="YlOrBr",edgecolor="black")
plt.title("Diet Quality & Alcohol Consumption By Gender & Age",size=13)
plt.show()





# Average Sleep By Gender
classifi_sl= df.groupby("Gender")["Sleep_Hours"].mean().reset_index()
classifi_sl.style.background_gradient(cmap="Purples")
print(classifi_sl)

plt.figure(figsize=(12,6))
sns.set_style("whitegrid")
sns.barplot(data=classifi_sl,x= "Gender",y="Sleep_Hours",edgecolor="black",palette="Set1")
plt.title("Average Sleep Hours By Gender",size=13)
plt.show()


print(df.head())
le = LabelEncoder()
cols= df[["Gender","Diet_Quality","Alcohol_Consumption","Chronic_Disease","Smoker","Exercise_Freq"]]
for col in cols:
    df[col]=le.fit_transform(df[col])
print(df.head())


x= df.drop(columns="Chronic_Disease")
y = df["Chronic_Disease"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)



# Logistic Regression Model

model=LogisticRegression(class_weight="balanced")
model.fit(x_train_scaled,y_train)

y_pred= model.predict(x_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Classification Report:\n",classification_report(y_test,y_pred))


# Random Forest Classfier Model

modelr= RandomForestClassifier(n_estimators=100,max_depth=None,random_state=42,class_weight= "balanced")
modelr.fit(x_train_scaled,y_train)

print(df["Chronic_Disease"].value_counts())

y_pred=modelr.predict(x_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))




# XG Boost Model
from xgboost import XGBClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

modelx = XGBClassifier(use_label_encoder = False,eval_metric='logloss',scale_pos_weight=4.0,random_state=42)
modelx.fit(x_train_scaled,y_train)

y_proba = modelx.predict_proba(x_test_scaled)[:,1]
y_proba = (y_proba>= 0.4).astype(int)
print("Accuracy:", accuracy_score(y_test,y_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_proba))
print("Classification Report:\n", classification_report(y_test, y_proba))

cor=df.corr()
plt.figure(figsize=(12,8))
sns.heatmap(data=cor,annot=True,fmt=".2f")
plt.title("Correlation Heatmap",size=13)
plt.show()
