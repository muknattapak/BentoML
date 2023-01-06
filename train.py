from __future__ import print_function

# import custom model class
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import pandas as pd
import bentoml
import os
import sys
import psycopg2

def main():
    conn = psycopg2.connect(database="postgres",
                        host="es.aidery.io",
                        user="airflow",
                        password="airflow",
                        port="5433")

    cursor = conn.cursor()
    cursor.execute("SELECT * FROM applewatch;")

    tuples_list = cursor.fetchall()
    AP = pd.DataFrame(tuples_list)
    AP.drop([0],axis=1,inplace=True)
    AP.rename(columns ={1:'Heart',2:'Calories',3:'Steps',4:'Distance',5:'Age',6:'Gender',7:'Weight',8:'Height',9:'Activity'}, inplace = True)
    # Load the dataset
   # AP = pd.read_csv('/Users/rachapon_charoenyingpaisal/Documents/bentoml/test/AppleWatch.csv',usecols=['Heart','Calories','Steps','Distance','Activity'])
    # AP.drop(['Age','Gender','Weight','Height'],axis=1,inplace=True)
    AP.drop(['Height'],axis=1,inplace=True)
    AP['Activity'] = AP['Activity'].replace("0.Sleep", 0)
    AP['Activity'] = AP['Activity'].replace("1.Sedentary", 1)
    AP['Activity'] = AP['Activity'].replace("2.Light", 2)
    AP['Activity'] = AP['Activity'].replace("3.Moderate", 3)
    AP['Activity'] = AP['Activity'].replace("4.Vigorous", 4)
    AP['Gender'] = AP['Gender'].replace("M", 0)
    AP['Gender'] = AP['Gender'].replace("F", 1)
    #drop values
    AP.drop(AP[AP.Activity > 1].index, inplace=True)
    X=AP.drop('Activity',axis=1)
    y=AP['Activity']
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Fit and predict using LDA
    dt = DecisionTreeClassifier(random_state=42)
    params = {
    'max_depth': [2, 3, 5, 10, 20, 30, 40],
    'min_samples_leaf': [5, 10, 20, 50, 100, 150, 200],
    'criterion': ["gini", "entropy"]
    }
    
    grid_search = GridSearchCV(estimator=dt,
                           param_grid=params,
                           cv=5, n_jobs=-1, verbose=1, scoring = "accuracy")
    grid_search.fit(X_train, y_train)
    dt=grid_search.best_estimator_
    y_pred = dt.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Save model with BentoML
    saved_model = bentoml.sklearn.save_model(
        "dt_latest2",
        dt,
        signatures={"predict": {"batchable": True}},
    )
    print(f"Model saved: {saved_model}")


if __name__ == "__main__":
    main()