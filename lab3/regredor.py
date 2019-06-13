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
ndni = 7
data = data.values[:, 1:5]
data = (data - data.mean())/(data.max() - data.min())
Y = data[7:,3]
# Y = np.transpose(np.array([Y]))

X = list()
for i in range(len(data)-ndni):
    tmp = []
    for j in range(ndni):
        tmp += data[i+j, :].tolist()
    X.append(tmp)
X = np.array(X)

print("dlugosc Y: ", Y.shape, " || dlugosc X : ", X.shape)

n = len(data)

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
                y_test = np.array(y_test)
                regresor.fit(X_train, y_train)
                y_pred = np.array(regresor.predict(X_test))
                print("test:",y_test.shape, y_test,"pred: ", y_pred.shape,y_pred)
                print("max: ",max([y_test - y_pred]))

                srbladkw += sum((y_test - y_pred)**2)
                print("sredni blad kwadratowy: ", srbladkw/len(y_pred))
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
                writer.writerows(np.round(wyniki,6))
            time.sleep(1)
            i2 +=1
        i2 = 0
        i1 +=1
    i1 = 0
    i0 += 1
np.savetxt("wyniki_gielda_savetxt.csv", wyniki.tolist(), fmt='%2.3f', delimiter=",")
print("czas trwania algorytmu: {}".format(time.time() - start_Alg))
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@: naj_eta: {}, naj_alg: {}, naj_licz_NL {}". format(naje, najalg, najN))