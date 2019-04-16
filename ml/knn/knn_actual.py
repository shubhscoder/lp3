'''
Author : Shubham Sangamnerkar
Roll no : 4351

Kids : C++
Adults : Python
Legends : Sanskrit XD 
'''

import heapq
import pandas as pd
import numpy as np

def distance(a, b, k):
    return np.power(np.sum(np.power(np.abs(a - b), k)), 1 / k) 

class knn:
    def __init__(self, k = 2, weights = 'uniform', distance_method = 'general', p = 2):
        assert k>0, "Number of neighbours should be a positive integer"
        assert weights in ['uniform', 'distance'], "Weight can be either uniform or distance"
        assert p>0, "Non positive power not supported"
        assert distance_method in "general", "General, eucildean, or manhattan supported"
        self.k = k
        self.weights = weights 
        self.metric = distance_method
        self.power = p
        self.x = None
        self.y = None
    
    def getneighbours(self, X):
        distances = []
        for i in range(len(self.x)):
            distances.append((distance(self.x[i], X, self.power), i))
        heapq.heapify(distances)
        return heapq.nsmallest(self.k, distances)
    
    def give_instance(self, X):
        class_freq = dict()
        dist = self.getneighbours(X)
        weights = {}
        if self.weights == 'uniform':
            weights = { i[1] : 1 for i in dist}
        else:
            for i in dist:
                if i[0] > 0:
                    weights[i[1]] = (1.0 / i[0] * 1.0)
                else:
                    return self.y[i[1]]
        
        total = sum(weights.values())
        for i in dist:
            op_class = self.y[i[1]]
            if op_class in class_freq:
                class_freq[op_class] += (weights[i[1]] * 1.0)
            else:
                class_freq[op_class] = (weights[i[1]] * 1.0)
        final_class = None
        max_freq = -10**18
        for key, value in class_freq.items():
            if max_freq < (value * 1.0 / total * 1.0):
                max_freq = (value * 1.0 / total * 1.0)
                final_class = key
        return final_class
            
    def fit(self, x, y):
        self.x = x.values
        self.y = y.values
    
    def predict(self, X):
        y_pred = [ self.give_instance(i) for i in X.values]
        return np.array(y_pred)