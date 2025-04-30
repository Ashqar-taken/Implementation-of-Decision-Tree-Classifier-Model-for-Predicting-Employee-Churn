# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries
2. Find the null values and count them and count number of left values.
3. Assign the train dataset and test dataset.
4. Train a Decision Tree model using training data, use criteria as entropy.
5. Find the accuracy of our model and predict the require values.

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Ashqar Ahamed S.T
RegisterNumber: 212224240018
```
```
import pandas as pd
data = pd.read_csv(r"C:\College\SEM 2\Machine Learning\Exp8\Employee.csv")
print(data.head())
print("\nData Info:\n")
print(data.info())

data.isnull().sum()

data["left"].value_counts

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
print("\nData after label encoder:\n")
print('\n',data.head())

x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

print("\nX Values:\n",x.head())
y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_pred)
print("\nAccuracy:",accuracy)

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```


## Data:
![data](https://github.com/user-attachments/assets/6ff5867b-8f14-4d13-b5ab-c79bc4d8d9c7)
## Data Info:
![image](https://github.com/user-attachments/assets/845a8b30-5093-4f67-890b-b9a498306391)
## Data after labelencoder:
![image](https://github.com/user-attachments/assets/c98670e8-e900-4f73-8bad-1f1f356b406d)
## X Values:
![image](https://github.com/user-attachments/assets/22b9496a-fd38-477a-bcba-9b21dcf6035c)
## Accuracy:
![image](https://github.com/user-attachments/assets/c5cb90f7-9e61-4b0f-81c3-881d8e23219b)
## Prediction for new value:
![image](https://github.com/user-attachments/assets/2a56436b-ab65-46df-9a1b-2f6567436fb3)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
