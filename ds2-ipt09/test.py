#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:44:24 2018

@author: linkededucation
"""

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kmeans3 import MKMeans
from collections import Counter
import numpy as np
 
def quality(X, kmeans):
    cs = kmeans.predict(X)
    flowers = [Counter(cs[:50]), Counter(cs[50:100]), Counter(cs[100:])]
    return flowers
  

def read_data():
    with open('iris.txt', 'r') as fp:
        X = [[*map(float, row.split(",")[:-1])] for row in fp.readlines()]
        
    return np.array(X)

 
def tests():
    X = read_data()
    
    print("*"*20)
    kmeans = MKMeans(3)
    kmeans.fit(X)
    print(quality(X, kmeans))
    print("-"*20)
    
    km = KMeans(n_clusters=3, random_state=0).fit(X)
    print(quality(X, km))
    print("-"*20)
    
    new_X = StandardScaler().fit_transform(X)
    kmeans = MKMeans(3)
    kmeans.fit(new_X)
    print("Com regularização: versão")
    print(quality(new_X, kmeans))
    print("-"*20)
    
    kmeans = KMeans(n_clusters=3, random_state=0).fit(new_X)
    print("Com regularização: scikit")
    print(quality(new_X, kmeans))
    print("-"*20)


    
tests()


    