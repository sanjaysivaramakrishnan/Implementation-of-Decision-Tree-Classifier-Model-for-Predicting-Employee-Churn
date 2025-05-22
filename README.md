# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import pandas module and import the required data set.

2. Find the null values and count them.

3. Count number of left values.

4. From sklearn import LabelEncoder to convert string values to numerical values.

5. From sklearn.model_selection import train_test_split.

6. Assign the train dataset and test dataset.

7. From sklearn.tree import DecisionTreeClassifier.

8. Use criteria as entropy.

9. From sklearn import metrics.

10. Find the accuracy of our model and predict the require values.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Sanjay Sivamakrishnan M
RegisterNumber: 212223240151

import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv(r'C:\Users\admin\Desktop\Python_jupyter\ML LEARN\intro_machine_learning\data_sets\Employee.csv')
df.head()
print(df.isnull().sum())
df.info()
le = LabelEncoder()
df['salary'] = le.fit_transform(df['salary'])
df.info()
df.columns
X = df.drop(columns=['left','Departments '])
y = df['left']
X.head()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print('The models accuracy is : ',accuracy)
model.predict([[0.8,	0.86,	5,	262,	6,	0,	0,	2]])

*/
```

## Output:

![image](https://github.com/user-attachments/assets/78411f2f-2817-49a1-a557-f1222a567f81)
![image](https://github.com/user-attachments/assets/e7bf4363-d0ba-408a-bfc4-d427c0135e0d)
![image](https://github.com/user-attachments/assets/f6f06e65-2450-4ee0-8e8f-198e34ebabab)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
