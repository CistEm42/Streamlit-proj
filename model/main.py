import pandas as pd
import numpy as np
from sklearn import model_selection, linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle


def create_model(data):
    y=data['diagnosis']
    x=data.drop(['diagnosis'], axis=1)
    
    # splitting the data
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # scaling the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # training the model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # test the model
    y_pred = model.predict(X_test_scaled)
    print("The accuracy of the model is: ", accuracy_score(y_test, y_pred))
    print("Classification report is: \n", classification_report(y_test,y_pred))

    return model, scaler


    

def clean_data():
    # Fetching and cleaning the data
    data = pd.read_csv("data/data.csv")
    data.drop(["Unnamed: 32", 'id'], axis=1, inplace=True)
    data['diagnosis']=data['diagnosis'].replace({'M':1,'B':0})
    data['diagnosis'] = data['diagnosis'].astype("category", copy=False)
    print(data.head())
    return data


def main():
    data = clean_data()
    model, scaler = create_model(data)

    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('model/scaler.pkl','wb') as a:
        pickle.dump(scaler, a)






if __name__ == '__main__':
    main()