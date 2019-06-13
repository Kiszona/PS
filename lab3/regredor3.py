import time
import random
import csv
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale
start_Alg = time.time()
data = pd.read_csv("D:\\pwr\\sem6ark\\"
                   "Programowaniesieciowe\\"
                   "lab\\lab3\\all_stocks_5yr.csv",
                   sep='.',delimiter=',')
data = data.dropna()
data = data.values[:, 1:5]
Y = data[7:,3]
ndni = 7
X = list()
for i in range(len(data)-ndni):
    tmp = []
    for j in range(ndni):
        tmp += data[i+j, :].tolist()
    X.append(tmp)
X = np.array(X)
print("dlugosc Y: ", Y.shape, " || dlugosc X : ", X.shape)
# Y = data[:,3]
# X = data[:,0:3]
# data[:,0] = data[:,0]/max(data[:,0])
n = len(data)
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=.)

iloscN = range(10,290,10)
algorytmy = ('adam','sgd','lbfgs')
eta = [0.01, 0.001, 0.0001]
# iloscN = range(10,30,10)
# algorytmy = ('adam', 'sgd')
# eta = [0.01]
i0 = 0
i1 = 0
i2 = 0
usrednienie = 1
wyniki = np.zeros([len(iloscN), 9])
with open('gielda_wyniki.csv', mode = 'w' ) as plik:
    writer = csv.writer(plik)
    writer.writerows(wyniki)
wartosci_rand = list()
MIN = 2**30
regcv = MLPRegressor(learning_rate='constant',activation='relu')
iloscN = range(10,290,10)
algorytmy = ('adam','sgd','lbfgs')
eta = [0.01, 0.001, 0.0001]
parameters = {'hidden_layer_sizes': iloscN, 'solver' : algorytmy,'learning_rate': eta}

clf = GridSearchCV(regcv, parameters, n_jobs=10)

wyniki = clf.fit(X,Y)
