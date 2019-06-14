import time
import random
import csv
import math
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


start_Alg = time.time()
data = pd.read_csv("D:\\pwr\\sem6ark\\"
                   "Programowaniesieciowe\\"
                   "lab\\lab3\\all_stocks_5yr.csv",
                   sep='.', delimiter=',')
data = data.dropna()
ndni = 7
data = data.values[:, 1:5]
data = (data - data.mean())/(data.max() - data.min())
# Y to kolejne probki po 7 dniu gdy nie zabraknie probek do szkolenia
Y = data[7:, 3]
# X to zbior danych wejsciowych oraz wyjsciowych 7 ostatnich dni(roboczych)
# przed zmienna przewidywana
X = list()
for i in range(len(data)-ndni):
    tmp = []
    for j in range(ndni):
        tmp += data[i+j, :].tolist()
    X.append(tmp)
X = np.array(X)
X_train, X_test, y_train, y_test =\
    train_test_split(X, Y, test_size=0.1, random_state=1)
y_test = np.array(y_test)
y_train = np.array(y_train)
mu, sig = 0, 2
szum = sig * np.random.randn(ndni*4,) + mu
# X_test[0] = X_test[0] + szum  #dodawany szum do zaklocenia wejscia

# standaryzacja danych
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train = np.array(X_train)
X_test = np.array(X_test)
print("wlasnosci danych wejsciowych: ")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
par = np.arange(1., 1.01, 0.01)
K_cent = range(14, 30, 2)
wyniki = np.zeros(len(K_cent) * len(par)).reshape(len(K_cent), len(par))

with open('rbf_wyniki.csv', mode='w') as plik:
    writer = csv.writer(plik)
    writer.writerows(wyniki)
i0 = 0
i1 = 0
print("ww")
for param in par:
    for cent in K_cent:
        start_okr = time.time()
        km = KMeans(n_clusters=cent, max_iter=80)
        km.fit(X_train)
        print("ww2")

        centres = km.cluster_centers_
        t_km = time.time()
        print("czas zrobienia km: {}".format(int(t_km - start_okr)))

        # obliczanie promienia
        MAX = 0
        for i in range(cent):
            for j in range(cent):
                d = np.linalg.norm(centres[i] - centres[j])
                # d = centres[i] - centres[j]
                if d > MAX:
                    MAX = d
        d = MAX
        t_promien = time.time()
        print("czas zrobienia policzzenia dredniocy max: {}"
              .format(int(t_promien - t_km)))
        G = np.empty((X_train.shape[0], cent), dtype=float)
        sigma = (d / math.sqrt(2*cent)) * param
        # trenowanie sieci
        for i in range(X_train.shape[0]):
            for j in range(cent):
                odl = np.linalg.norm(X_train[i] - centres[j])
                # odl = (X_train[i] - centres[j])
                G[i][j] = math.exp(-odl**2/(2*sigma)**2)
        t_train = time.time()
        print("czas trenowania: {}".format(int(t_train - t_promien)))

        W = np.dot(np.dot(np.linalg.inv(np.dot(G.T, G)), G.T), y_train)

        G_test = np.empty((X_test.shape[0], cent), dtype=float)
        # testowanie sieci
        for i in range(X_test.shape[0]):
            for j in range(cent):
                odl = np.linalg.norm(X_test[i] - centres[j])
                # odl = X_test[i] - centres[j]
                G_test[i][j] = math.exp(-odl**2/(2*sigma)**2)
        t_test = time.time()
        print("czas testowania: {}".format(int(t_test - t_train)))
        print("czas okrazenia petli: {}".format(int(time.time() - start_okr)))

        pred = np.dot(G_test, W)
        wyniki[i1, i0 * 10] = r2_score(pred, y_test)
        # wyniki[i1, i0 * 10]
        # wyniki[i1, i0 * 10]
        print("typ y_test", type(y_test), "typ y_pred:", type(y_test))
        print("pred 1: ", pred[0], pred.shape,
              "y_test 1: ", y_test[0], y_test.shape,
              "wyn r^2 : ", wyniki[i1, i0 * 10],
              "wyn acc_score :", accuracy_score(y_test.tolist(), pred.tolist()))
        with open('gielda_wyniki2.csv', mode='w') as plik:
            writer = csv.writer(plik)
            writer.writerows(wyniki)
        i1 += 1
    i1 = 0
    i0 += 1
plik.close()
