## Kaggle Titanic  Final

# Importing Important Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math

### Importing Data Sets
train = pd.read_csv("C:\\Users\\Reema\\Desktop\\Rishabh\\Kaggle\\Kaggle Titanic\\train.csv")
test = pd.read_csv("C:\\Users\\Reema\\Desktop\\Rishabh\\Kaggle\\Kaggle Titanic\\test.csv")

### Data Wrangling
train_working = train[:]
train_describe = train_working.describe(include='all')
train_working.drop(['PassengerId','Ticket'], axis = 1, inplace = True)

# Deriving New Variables 
train_working['Cabin_Code'] = 0
for i in range(len(train_working)):
    train_working['Cabin_Code'][i] = str(train_working.Cabin[i])[0].upper()
train_working.drop('Cabin', axis = 1, inplace = True)

train_working['Name_Code'] = 0
for i in range(len(train_working)):
    dot_pos = train_working.Name[i].split(",")[1].strip().find(".")
    train_working['Name_Code'][i] = train_working.Name[i].split(",")[1].strip()[:dot_pos]
train_working.drop('Name',axis = 1,inplace = True)

# Replacing Nulls in Age by the avg. age of the Designation in Name
age_null = train_working.loc[train_working.Age.isnull(),['Name_Code','Age']]
desigation_avg = train_working.groupby('Name_Code').Age.mean()
for i in age_null.index:
    Designation = age_null['Name_Code'][i]
    train_working['Age'][i] = desigation_avg[Designation]

train_working.dropna(axis = 0, inplace = True)

# Detecting Outliers
cont_var = ['Age','Fare']
from sklearn.preprocessing import scale
for i in cont_var:
    train_working[(i+'_Scaled')] = scale(train_working[i])
    
# Removing Outliers
train_cleaned = train_working[(train_working.Age_Scaled > -1.95) & (train_working.Age_Scaled < 1.97)]
train_cleaned = train_cleaned[(train_working.Fare_Scaled > -1.95) & (train_working.Fare_Scaled < 1.97)]
train_cleaned.drop(['Age','Fare'],axis=1,inplace=True)

# Creating Dummies
train_final = pd.get_dummies(train_cleaned)
x = list(train_final.columns)
x.remove('Survived')
y='Survived'

## Train-Test Split
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(train_final[x], train_final[y], random_state = 1)

## Modelling : Logistic Regression
from sklearn.linear_model import LogisticRegression
test_model = LogisticRegression(random_state=1)
test_model.fit(train_x,train_y)
test_predict = test_model.predict(test_x)

## Model Evaluation
from sklearn.metrics import accuracy_score
model_accuracy = accuracy_score(test_y,test_predict)

def model_fn(test):
    ### Data Wrangling
    test_working = test[:]
    test_working.drop(['PassengerId','Ticket'], axis = 1, inplace = True)
    
    # Deriving New Variables 
    test_working['Cabin_Code'] = 0
    for i in range(len(test_working)):
        test_working['Cabin_Code'][i] = str(test_working.Cabin[i])[0].upper()
    test_working.drop('Cabin', axis = 1, inplace = True)
    
    test_working['Name_Code'] = 0
    for i in range(len(test_working)):
        dot_pos = test_working.Name[i].split(",")[1].strip().find(".")
        test_working['Name_Code'][i] = test_working.Name[i].split(",")[1].strip()[:dot_pos]
    test_working.drop('Name',axis = 1,inplace = True)

    # Replacing Nulls in Age by the avg. age of the Designation in Name
    age_null = test_working.loc[test_working.Age.isnull(),['Name_Code','Age']]
    desigation_avg = test_working.groupby('Name_Code').Age.mean()
    for i in age_null.index:
        Designation = age_null['Name_Code'][i]
        test_working['Age'][i] = desigation_avg[Designation]
    test_working.fillna(test_working.mean(), inplace = True)
    
    # Detecting Outliers
    cont_var = ['Age','Fare']
    from sklearn.preprocessing import scale
    for i in cont_var:
        test_working[(i+'_Scaled')] = scale(test_working[i])
    
    # Creating Dummies
    test_final = pd.get_dummies(test_working)
    x = list(set(train_final.columns) & set(test_final.columns))
    y='Survived'
    
    ## Final Model
    from sklearn.linear_model import LogisticRegression
    final_model = LogisticRegression(random_state=1)
    final_model.fit(train_final[x],train_final[y])
    final_predict = final_model.predict(test_final[x])
    
    ## Final Results
    Ultimate_Final = pd.concat([test['PassengerId'],pd.DataFrame(final_predict)], axis = 1 )
    Ultimate_Final.columns = ['PassengerId','Survived']
    Ultimate_Final.to_csv("C:\\Users\\Reema\\Desktop\\Rishabh\\Kaggle\\Kaggle Titanic\\Ultimate_Final.csv",index = False)

    return()
    
model_fn(test)
    
    