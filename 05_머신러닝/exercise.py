from sklearn.base import BaseEstimator
import numpy as np

class MyDummyClassifier(BaseEstimator):
    def fit(self,x,y=None):
        pass
    def predict(self,x):
        pred=np.zeros((x.shape[0],1)) # 0으로 채워진 배열
        for i in range(x.shape[0]):
            if x['Sex'].iloc[i] ==1:
                pred[i] = 0
            else:
                pred[i] = 1
        return pred


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def fillna(df):
    df['Age'].fillna(df['Age'].mean(),inplace=True)
    df['Cabin'].fillna('N',inplace = True)
    df['Embarked'].fillna('N',inplace = True)
    df['Fare'].fillna(0,inplace = True)
    return df

def drop_features(df):
    df.drop(columns=['PassengerId','Name','Ticket'],inplace = True)
    return df

def format_features(df):
    from sklearn.preprocessing import LabelEncoder
    df['Cabin']=df.Cabin.str[0]
    features = ['Cabin','Sex','Embarked']
    for feature in features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
        print(le.classes_)
    return df

def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df


df = pd.read_csv('titanic_train.csv')
y = df.Survived
x = df.drop(columns=['Survived'])
x = transform_features(x)

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

myclf = MyDummyClassifier()
myclf.fit(X_train,y_train)
pred = myclf.predict(X_test)
print(accuracy_score(y_test,pred))