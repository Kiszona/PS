from IPython.display import Image

# inline plotting instead of popping out
#matplotlib inline

# load utility classes/functions that has been taught in previous labs
# e.g., plot_decision_regions()
import os, sys
module_path = os.path.abspath(os.path.join('.'))
sys.path.append(module_path)
from lib import *
import random

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split

df = pd.read_csv('iris.data', header=None)

df.columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class label']
display(df.head())

X = df[['Sepal length','Sepal width', 'Petal length', 'Petal width']].values
PL = Petal_length = df['Petal length'].values
y = pd.factorize(df['Class label'])[0]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
pd.DataFrame(X_train).to_csv('X_train', header=None)
print('#Training data points: {}'.format(X_train.shape[0]))
print('#Testing data points: {}'.format(X_test.shape[0]))
print('Class labels: {} (mapped from {}'.format(np.unique(y), np.unique(df['Class label'])))


print(PL[0]+PL[1])
print('***********************************************************')
#algorytm

class perceptron(object):
    w0 = -1
    eta = 0.005
    wagi = [0, 0, 0, 0]

    def __init__(self):
        for i in range(4):
            self.wagi[i]=round(random.random()/10,3)
        self.wyswietl_wagi()
            
    def wyswietl_wagi(self):
        print('w0 = ', self.w0, ' wagi = ', self.wagi, ' eta = ', self.eta)

    def zeruj_wagi(self):
        for i in range(4):
            self.wagi[i]=round(random.random()/10,3)

    def dzialaj1(self, X, praw_odp):
        suma = 0
        odpowiedz = 0
        for i in range(4):
            suma += X[i]*self.wagi[i] + self.w0
        if suma >= 0:
            odpowiedz = 1
        else:
            odpowiedz = 0
        #print("odpowiedz :", odpowiedz)
        if odpowiedz == praw_odp:
            #print('odpowiedz prawidlowa')
            return 0
        else:
            #print('odpowiedz nieprawidlowa')
            for i in range(4):
                self.wagi[i] += self.eta * (praw_odp - odpowiedz) * X[i]
            ##self.wyswietl_wagi()
            return 1
        
neuron = perceptron()

il_bledow = 5
il_okr = 0
neuron.dzialaj1(X[1],1)
while  il_bledow >= 5:
    il_okr += 1
    il_bledow = 0
    for j in range(100):
        praw_odp = 1
        k = int(random.random()*150)
        #print(k)
        if k < 50:
            praw_odp = 1
        else:
            praw_odp = 0
        il_bledow += neuron.dzialaj1(X[k], praw_odp)
print('ilosc wykonanych petli: ', il_okr, ' ilosc bledow w ostatniej petli: ', il_bledow)
print('wagi: ',)
neuron.wyswietl_wagi()

#print(X_test, y_test, X_train,y_train)
      
