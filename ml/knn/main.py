'''
Author : Shubham Sangamnerkar
Roll no : 4351

Kids : C++
Adults : Python
Legends : Sanskrit XD 
'''

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from knn_actual import knn

class gV:
    df = None
    x_train = None
    x_test = None
    y_train = None
    y_test = None

def read_input(filepath):
    try:
        return pd.read_csv(filepath)
    except:
        print("Error in reading the file " + filepath)
        exit(1)

def summarize_data(df):
    print("HEAD")
    print(df.head())
    print()
    print("SUMMARY")
    print(df.describe())
    print()
    print("INFO")
    print(df.info())


def solve(model, x_train, x_test, y_train, y_test, pre):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(pre + "accuracy {}".format(accuracy_score(y_pred, y_test)))
    print()

def initialize_train_test(filepath, t_s):
    gV.df = read_input(filepath)
    x = gV.df.drop('Result', axis = 1)
    y = gV.df['Result']
    gV.x_train, gV.x_test, gV.y_train, gV.y_test = train_test_split(x, y, test_size = t_s) 

def solve_for_different_file(filepath):
    initialize_train_test(filepath, 0.3)
    solve(knn(k = 3, weights = 'uniform'), gV.x_train, gV.x_test, gV.y_train, gV.y_test, "Custom model ")
    solve(knn(k = 3, weights = 'distance'), gV.x_train, gV.x_test, gV.y_train, gV.y_test, "Custom distance weighted model ")
    solve(KNeighborsClassifier(n_neighbors = 3), gV.x_train, gV.x_test, gV.y_train, gV.y_test, "Sklearn knn model ")  

def predict_for_model(sample, model):
    model.fit(gV.x_train, gV.y_train)
    y_pred = model.predict(sample)
    print(y_pred)

def for_one_sample(sample, filepath):
    initialize_train_test(filepath, 0)
    predict_for_model(sample, knn(k = 3, weights = 'uniform'))
    predict_for_model(sample, knn(k = 3, weights = 'distance'))
    predict_for_model(sample, KNeighborsClassifier(n_neighbors = 3))
    
            
solve_for_different_file('dataset.csv')
#solve_for_different_file('dataset1.csv') Uncomment me to check accuracy on a bigger dataset
for_one_sample(pd.DataFrame({'X' : [6], 'Y' : [6]}), 'dataset.csv')


    
        