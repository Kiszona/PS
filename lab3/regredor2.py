import time
import random
import csv
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
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

# iloscN = range(10,290,10)
algorytmy = ('adam','sgd','lbfgs')
eta = [0.01, 0.001, 0.0001]
iloscN = range(10,30,10)
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

for alg in algorytmy:
    for e in eta:
        for N in iloscN:
            srbladkw = 0
            for i in range(usrednienie):

                start_okr = time.time()
                rand = int(random.random()*1000)
                wartosci_rand.append(rand)
                X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.3, random_state=rand)
                regresor = MLPRegressor(hidden_layer_sizes=(N,),
                                        solver=alg,
                                        learning_rate='constant',
                                        learning_rate_init=e,
                                        activation='relu')
                regresor.fit(X_train, y_train)
                # y_pred = regresor.predict(X_test)
                srbladkw += sum((np.array(y_test) - regresor.predict(X_test))**2)
                print("czas okrazenia petli: {}".format(int(time.time() - start_okr)))

            srbladkw = srbladkw/usrednienie
            print(srbladkw)
            if srbladkw < MIN:
                MIN = srbladkw
                najalg = alg
                naje = e
                najN = N
            wyniki[i2, (i1 + i0*3)] = srbladkw
            with open('gielda_wyniki.csv', mode='w') as plik:
                writer = csv.writer(plik)
                writer.writerows(wyniki)
            time.sleep(1)
            i2 +=1
        i2 = 0
        i1 +=1
    i1 = 0
    i0 += 1
np.savetxt("wyniki_gielda_savetxt.csv", wyniki.tolist(), fmt='%5.3f', delimiter=",")
print("czas trwania algorytmu: {}".format(time.time() - start_Alg))
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@: naj_eta: {}, naj_alg: {}, naj_licz_NL {}". format(naje, najalg, najN))