import time
from sklearn.metrics import accuracy_score
start_Alg = time.time()
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from sklearn.neural_network import  MLPClassifier
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_mldata


def shuffle(X, y):
    r = np.random.permutation(len(y))
    return X[r], y[r]

data = pd.read_csv("~/PycharmProjects/PS/lab2/mnist_train.csv")
#test = pd.read_csv("~/PycharmProjects/PS/lab2/mnist_test.csv")

print("dane wczytane")
Y = data.values[0:5000,0]
X = data.values[0:5000, 1:28**2 +1]
print("X i Y zrobione")
def toHotOne(Y):
    shape = Y.shape[0]
    tmp=np.zeros((shape,10))

    for y in Y:
        tmp[y] = 1

    return tmp

#mnist = fetch_mldata('MNIST original')

#X, Y = mnist.data, mnist.target

#print(X)
Y_hot_one = toHotOne(Y)
print("hotone zrobione")
#X_train, X_test, y_train, y_test = train_test_split(X/255.,Y_hot_one, test_size=0.20, random_state=0)
print("dane podzielone")
algorytmy = ['adam', 'sgd', 'lbfgs']
eta = [0.001, 0.0001, 0.00001]
najlepsza_dok = 0
for alg in algorytmy:
    for e in eta:
        for liczbaN in range(10,110,10):
            srednia_dokladnosc = 0
            for i in range(10):

                start_okr = time.time()
                y_pred_dokl = []
                #X_train, y_train = shuffle(X_train,y_train)
                X_train, X_test, y_train, y_test = train_test_split(X / 255., Y, test_size=0.20, random_state=i)
                mlp = MLPClassifier(hidden_layer_sizes=(liczbaN,), solver=alg, learning_rate='constant', learning_rate_init=e)

                mlp.fit(X_train, y_train)
                y_test = np.array(y_test)
                #print(y_test)
                y_pred = mlp.predict_proba(X_test)
                y_pred = np.array(y_pred)
                for j in range(y_pred.shape[0]):
                    y_pred_dokl.append(np.argmax(y_pred[j,0:y_pred.shape[1]]))
                y_pred_dokl = np.array(y_pred_dokl)

                #print('Dokladnosc: %.2f' % accuracy_score(y_test, y_pred_dokl))
                srednia_dokladnosc += accuracy_score(y_test, y_pred_dokl)
                #zamienic w Y_pred najwieksza kolumne najwieksza na 1 a reszte wyzerowac
                #print(" wykonano petle dla: algorytm = {}, eta = {}, liczbaNeuronow = {}".format(alg,e,liczbaN))

                print("czas okrazenia petli: {}".format(time.time() - start_okr))
            srednia_dokladnosc = srednia_dokladnosc/10
            print("srednia dokladnosc: {} dla algorytm = {}, eta = {}, liczbaNeuronow = {}".format(srednia_dokladnosc,alg,e,liczbaN))
            if(srednia_dokladnosc > najlepsza_dok):
                najlepsza_dok = srednia_dokladnosc
                naj_eta = eta
                naj_alg = alg
                naj_licz_N = liczbaN
#wybranie najlepszych parametrow dla mlp
print("czas trwania algorytmu: {}".format(time.time() - start_Alg))
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@: naj_eta: {}, naj_alg: {}, naj_licz_NL {}". format(naj_eta, naj_alg, naj_licz_N))
