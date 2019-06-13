import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import math
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as numpy
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
import random

#Dataset
data = pd.read_csv("D:\\pwr\\sem6ark\\"
                   "Programowaniesieciowe\\"
                   "lab\\lab3\\all_stocks_5yr.csv",
                   sep='.',delimiter=',')
data = numpy.array(data)

data = numpy.delete(data, [0 ,5 ,6], 1)
data = data.tolist()

prepareddata = []
targets = []

def datapreparation(n_days):
        global data
        global targets
        while len(data) > n_days:
                templist = []
                for i in range(n_days+1):
                        dat = data.pop(0)
                        for x in range(len(dat)):
                                if i == n_days and x > 0:
                                        targets.append(dat[3])
                                        break
                                else:
                                        templist.append(dat[x])
                prepareddata.append(templist)
                                

datapreparation(7)

dataarray = numpy.array(prepareddata)
targetarray = numpy.array(targets)


#Podzial danych 90%-10%
X_train, X_test, Y_train, Y_test = train_test_split(dataarray,targetarray,test_size=0.1, random_state=1)
print(X_test[0])
print(X_test[0].shape)
mu, sig = 0, 2
noise_test = sig * numpy.random.randn(29,) + mu
X_test[0] = X_test[0] + noise_test
print(X_test[0])

scaler= StandardScaler()
scaler.fit(X_train)
X_train= scaler.transform(X_train)
X_test= scaler.transform(X_test)

plik = open('Test.txt', 'w')


for param in numpy.arange(1.05,1.06,0.01):
        for item in range(30,140,5):
                K_cent= item
                km= KMeans(n_clusters= K_cent, max_iter= 100)
                km.fit(X_train)
                cent= km.cluster_centers_

                max=0 
                for i in range(K_cent):
                        for j in range(K_cent):
                                d= numpy.linalg.norm(cent[i]-cent[j])
                                if(d> max):
                                        max= d
                d= max

                sigma= (d/math.sqrt(2*K_cent)) * param

                shape= X_train.shape
                row= shape[0]
                column= K_cent
                G= numpy.empty((row,column), dtype= float)
                for i in range(row):
                        for j in range(column):
                                dist= numpy.linalg.norm(X_train[i]-cent[j])
                                G[i][j]= math.exp(-math.pow(dist,2)/math.pow(2*sigma,2))

                GTG= numpy.dot(G.T,G)
                GTG_inv= numpy.linalg.inv(GTG)
                fac= numpy.dot(GTG_inv,G.T)
                W= numpy.dot(fac,Y_train)

                row= X_test.shape[0]
                column= K_cent
                G_test= numpy.empty((row,column), dtype= float)
                for i in range(row):
                        for j in range(column):
                                dist= numpy.linalg.norm(X_test[i]-cent[j])
                                G_test[i][j]= math.exp(-math.pow(dist,2)/math.pow(2*sigma,2))

                prediction= numpy.dot(G_test,W)
                score = r2_score(prediction, Y_test)
                print(prediction[0])
                print(Y_test[0])
                ll = 'For param: ' + str(param)
                nn = 'For K_cent: ' + str(item)
                ss = 'Test accuracy' + str(score)
                plik.write("%s\n" % ll)
                plik.write("%s\n" % nn)
                plik.write("%s\n" % ss)
                print (score.mean())
                print(item)
                print(param)


plik.close()
