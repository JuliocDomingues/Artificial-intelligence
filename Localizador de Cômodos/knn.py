# -*- coding: utf-8 -*-
"""

link arquivo .csv:
https://drive.google.com/file/d/1ErXN1frPUrHVM-FIOIb8oMvZtEFFVCF_/view?usp=sharing

Feito por Julio César Domingues dos Santos

Inicio fase 1
"""

import os

from google.colab import drive
drive.mount('/content/drive')

os.chdir('/content/drive/MyDrive/datasets')

import sklearn
import pandas
import numpy as np

#Arquivo .csv com 25 exemplos de cada cômodo, ou seja, 75 exemplos no total.
df = pandas.read_csv('DadosWifi.csv')

"""Fim fase 1

Inicio fase 2
"""

x = df[['VIVO-4CE9','Cheris 2G','Luana']].values

y = df[df.columns[3]].values
len(y)

qv = x[0:25,0] #Quarto rede VIVO
qc = x[0:25,1] #Quarto rede Cheris
ql = x[0:25,2] #Quarto rede Luana
sv = x[0:50,0] #Sala rede VIVO
sc = x[0:50,1] #Sala rede Cheris
sl = x[0:50,2] #Sala rede Luana
q2v = x[0:75,0] #Quarto2 rede VIVO
q2c = x[0:75,1] #Quarto2 rede Cheris
q2l = x[0:75,2] #Quarto2 rede Luana

import matplotlib.pyplot as plt
plt.plot(qv[:],'bo',label='Quarto')
plt.plot(qc[:],'ro',label='Sala')
plt.plot(ql[:],'go',label='Quarto2')
plt.legend()

plt.plot(sv[:],'bo',label='Quarto')
plt.plot(sc[:],'ro',label='Sala')
plt.plot(sl[:],'go',label='Quarto2')
plt.legend()

plt.plot(q2v[:],'bo',label='Quarto')
plt.plot(q2c[:],'ro',label='Sala')
plt.plot(q2l[:],'go',label='Quarto2')
plt.legend()

#Normalização de todos os dados
xnorm = (x-x.mean(axis=0))/x.std(axis=0)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit,train_test_split
from sklearn.metrics import accuracy_score

#70% para o conjunto de treino e  30% para o conjunto de teste
xteste, xtreino, yteste, ytreino = train_test_split(xnorm, y, test_size=0.7, random_state=2)

#15% para o conjunto de teste e  15% para o conjunto de validação
xteste, xvalidacao, yteste, yvalidacao = train_test_split(xteste, yteste, test_size=0.5, random_state=6)

print(len(xtreino), len(ytreino))
print(len(xteste), len(yteste))
print(len(xvalidacao), len(yvalidacao))

"""Fim fase 2

Inicio fase 3
"""

#Escolha do melhor model
i = 0
model = []
model.append(KNeighborsClassifier(n_neighbors=5, weights='distance', p = 1))
model.append(KNeighborsClassifier(n_neighbors=5, weights='uniform', p = 1))
model.append(KNeighborsClassifier(n_neighbors=5, weights='distance', p = 2))
model.append(KNeighborsClassifier(n_neighbors=5, weights='uniform', p = 2))
model.append(KNeighborsClassifier(n_neighbors=3, weights='distance', p = 1))
model.append(KNeighborsClassifier(n_neighbors=3, weights='uniform', p = 1))
model.append(KNeighborsClassifier(n_neighbors=3, weights='distance', p = 2))
model.append(KNeighborsClassifier(n_neighbors=3, weights='uniform', p = 2))
sss = StratifiedShuffleSplit(n_splits=5, test_size= 3, random_state=3)
melhor = 0.0
model_i = 0
while i < 8:
  lacc = []
  #treino em várias etapas com o conjunto de validação
  for train_index,test_index in sss.split(xvalidacao,yvalidacao):
    xtrain = xnorm[train_index]
    ytrain = y[train_index]
    xtest  = xnorm[test_index]
    ytest  = y[test_index]
    model[i].fit(xtrain,ytrain)
    ypred = model[i].predict(xtest)
    acc = accuracy_score(ytest,ypred)
    print(acc)
    lacc.append(acc)
    if melhor < acc:
      melhor = acc
      model_i = i
  print("Para o model %d-> " % i, end=' ')
  print("Média: %4.3f Std: %4.3f "%(np.mean(lacc),np.std(lacc)))
  i += 1


print("O melhor model -> %d, seu score -> %g" % (model_i, melhor))

melhor_model = model[model_i]

"""Fim fase 3

Inicio fase 4
"""

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

#Treinamento com treino+validação
xtreinofinal = np.concatenate((xtreino, xvalidacao))
yvalidacaofinal = np.concatenate((ytreino, yvalidacao))

melhor_model.fit(xtreinofinal, yvalidacaofinal)

#Reports
r = melhor_model.predict(xteste)
a = accuracy_score(yteste, r)
p = precision_score(yteste, r, average='weighted')
recall = recall_score(yteste, r, average='weighted')
f1 = f1_score(yteste, r, average='weighted')
confusion = confusion_matrix(yteste, r)

print("Accuracy -> ", a)
print("Precision -> ", p)
print("Recall -> ", recall)
print("F1 -> ", f1)
print("Confusion -> ", confusion)

"""Fim fase 4"""