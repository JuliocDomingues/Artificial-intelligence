# -*- coding: utf-8 -*-
"""
Feito por Julio César Domingues dos Santos
"""

import copy
import matplotlib.pyplot as plt

def valid(x,y):
    r = True
    if x < 0 : r = False
    if x > 2 : r = False
    if y < 0 : r = False
    if y > 2 : r = False
    return r

def sons(s):
    r = []
    x = None
    y = None
    #localiza zero
    for i in range(len(s)):
        for j in range(len(s[i])):
            if s[i][j] == 0:
                x = i
                y = j

    # cima
    vx = x - 1
    vy = y
    if (valid(vx,vy)):
        ts = copy.deepcopy(s)
        t = ts[vx][vy]
        ts[vx][vy] = ts[x][y]
        ts[x][y] = t
        r.append(ts)

    # baixo
    vx = x + 1
    vy = y
    if (valid(vx,vy)):
        ts = copy.deepcopy(s)
        t = ts[vx][vy]
        ts[vx][vy] = ts[x][y]
        ts[x][y] = t
        r.append(ts)

    # direita
    vx = x 
    vy = y +1
    if (valid(vx,vy)):
        ts = copy.deepcopy(s)
        t = ts[vx][vy]
        ts[vx][vy] = ts[x][y]
        ts[x][y] = t
        r.append(ts)

    # esquerda
    vx = x 
    vy = y - 1
    if (valid(vx,vy)):
        ts = copy.deepcopy(s)
        t = ts[vx][vy]
        ts[vx][vy] = ts[x][y]
        ts[x][y] = t
        r.append(ts)

    return r

def printPuzzle(s):
    for v in s:
        print(v)

def son2str(s):
    s1 = s[0]+s[1]+s[2]
    return ''.join([str(v) for v in s1])

def bfs(start,goal):
    l = [start]
    fathers = dict()
    visited = [start]
    while (len(l)>0):
        father = l[0]
        del l[0]
        for son in sons(father):
            if son not in visited:
                visited.append(son)
                fathers[son2str(son)] = father
                if son == goal:
                    res = []
                    node = son
                    while node != start:
                        res.append(node)
                        node = fathers[son2str(node)]
                    res.append(start)
                    res.reverse()
                    print("BUSCA BFS")
                    print("Quantidade de nós ",len(visited))
                    return res
                else:
                    l.append(son)
    print("Sem Solucao")

s = [[4,1,3],[2,5,6],[0,7,8]]

# 4 1 3       1 2 3
# 2 5 6   ->  4 5 6
# 0 7 8       7 8 0
resp = bfs(s,[[1,2,3],[4,5,6],[7,8,0]])
qtd = 0
for s in resp:
    qtd+=1
    printPuzzle(s)
    print()
print("QUANTIDADE DE MOVIMENTOS = ", qtd)

"""Inicio da implementação do exercício 01"""

def h1(a,b):
  pos = 0
  for i in range(len(a)):
    for j in range(len(a[i])):
      if a[i][j] != b[i][j]:
        pos+=1;
  return pos

start = [[4,1,3],[2,5,6],[0,7,8]]
goal  = [[1,2,3],[4,5,6],[7,8,0]]
# 4 1 3       1 2 3
# 2 5 6   ->  4 5 6
# 0 7 8       7 8 0
h1(start, goal)

"""Fim da implementação do exercício 01"""

def h2(a,b): # distancia de manhatan
    dist = 0
    tam = len(a)*len(a[0])
    v = [[] for i in range(tam)]
    for i in range(len(a)):
        for j in range(len(a[i])):
            v[a[i][j]].append((i,j))
            v[b[i][j]].append((i,j))
    for i in range(tam):
        dist += abs(v[i][0][0]-v[i][1][0]) + abs(v[i][0][1]-v[i][1][1])
    return dist

start = [[4,1,3],[2,5,6],[0,7,8]]
goal  = [[1,2,3],[4,5,6],[7,8,0]]
# 4 1 3       1 2 3
# 2 5 6   ->  4 5 6
# 0 7 8       7 8 0
# 1 +2 +0+1+0 +0+1+1+2 = 8
h2(start,goal)

from heapq import heappush, heappop

def busca_heuristica(start,goal,heuristica):
    h = []
    heappush(h,(heuristica(start,goal),start))
    fathers = dict()
    visited = [start]
    while (len(h)>0):
        (_,father) = heappop(h)
        for son in sons(father):
            if son not in visited:
                visited.append(son)
                fathers[son2str(son)] = father
                if son == goal:
                    res = []
                    node = son
                    while node != start:
                        res.append(node)
                        node = fathers[son2str(node)]
                    res.append(start)
                    res.reverse()
                    return res, len(visited)
                else:
                    heappush(h,(heuristica(son,goal),son))
    print ("Sem Solucao")

start = [[4,1,3],[2,5,6],[0,7,8]]
goal  = [[1,2,3],[4,5,6],[7,8,0]]
# 4 1 3       1 2 3
# 2 5 6   ->  4 5 6
# 0 7 8       7 8 0
# 1 +2 +0+1+0 +0+1+1+2 = 8
(resp,_) = busca_heuristica(start,goal,h1)
qtd = 0
for s in resp:
    qtd+=1
    printPuzzle(s)
    print()
print("QUANTIDADE DE MOVIMENTOS = ", qtd)

"""Inicio da implementação do exercício 02"""

goal  = [[1,2,3],[4,5,6],[7,8,0]]

qtdmovimentosh1 = [[] for i in range(0,5)]
qtdmovimentosh2 = [[] for i in range(0,5)]

qtdnosh1 = [[] for i in range(0,5)]
qtdnosh2 = [[] for i in range(0,5)]

start5  = [[0,2,3],[1,5,6],[4,7,8]]
start10 = [[2,3,6],[0,1,5],[4,7,8]]
start15 = [[2,6,5],[1,3,8],[0,4,7]]
start20 = [[2,0,5],[1,6,8],[4,3,7]]
start25 = [[1,2,0],[4,3,8],[7,5,6]]


#5 movimentos h1
(resp, qtdnosh1[0]) = busca_heuristica(start5,goal,h1)
qtd = 0
for s in resp:
    qtd+=1

qtdmovimentosh1[0] = qtd

#10 movimentos h1
(resp, qtdnosh1[1]) = busca_heuristica(start10,goal,h1)
qtd = 0
for s in resp:
    qtd+=1

qtdmovimentosh1[1] = qtd

#15 movimentos h1
(resp, qtdnosh1[2]) = busca_heuristica(start15,goal,h1)
qtd = 0
for s in resp:
    qtd+=1

qtdmovimentosh1[2] = qtd

#20 movimentos h1
(resp, qtdnosh1[3]) = busca_heuristica(start20,goal,h1)
qtd = 0
for s in resp:
    qtd+=1

qtdmovimentosh1[3] = qtd

#25 movimentos h1
(resp, qtdnosh1[4]) = busca_heuristica(start25,goal,h1)
qtd = 0
for s in resp:
    qtd+=1

qtdmovimentosh1[4] = qtd

#5 movimentos h2
(resp, qtdnosh2[0]) = busca_heuristica(start5,goal,h2)
qtd = 0
for s in resp:
    qtd+=1

qtdmovimentosh2[0] = qtd

#10 movimentos h2
(resp, qtdnosh2[1]) = busca_heuristica(start10,goal,h2)
qtd = 0
for s in resp:
    qtd+=1

qtdmovimentosh2[1] = qtd

#15 movimentos h2
(resp, qtdnosh2[2]) = busca_heuristica(start15,goal,h2)
qtd = 0
for s in resp:
    qtd+=1

qtdmovimentosh2[2] = qtd

#20 movimentos h2
(resp, qtdnosh2[3]) = busca_heuristica(start20,goal,h2)
qtd = 0
for s in resp:
    qtd+=1

qtdmovimentosh2[3] = qtd

#25 movimentos h2
(resp, qtdnosh2[4]) = busca_heuristica(start25,goal,h2)
qtd = 0
for s in resp:
    qtd+=1

qtdmovimentosh2[4] = qtd

print("Quantidade de movimento h1 ->", qtdmovimentosh1)
print("Quantidade de movimento h2 ->", qtdmovimentosh2)

print("Quantidade de nós visitados h1 ->", qtdnosh1)
print("Quantidade de nós visitados h2 ->", qtdnosh2)

"""Fim da implementação do exercício 02

Inicio da implementação do exercício 03
"""

plt.plot([5, 10, 15, 20, 25], qtdnosh1, label='Heurística 01', color='red', marker="8")
plt.plot(qtdmovimentosh2, qtdnosh2, label='Heurística 02', color='purple', marker="8")
plt.xlabel('Número de movimentos')
plt.ylabel('Número de nós visitados')
plt.legend(['h1','h2'])

"""Fim da implementação do exercício 03

Inicio implementação do exercício 04
"""

def a_star(start,goal,heuristica):
    h = []
    heappush(h,(heuristica(start,goal) + len(h),start))
    fathers = dict()
    visited = [start]
    while (len(h)>0):
        (_,father) = heappop(h)
        for son in sons(father):
            if son not in visited:
                visited.append(son)
                fathers[son2str(son)] = father
                if son == goal:
                    res = []
                    node = son
                    while node != start:
                        res.append(node)
                        node = fathers[son2str(node)]
                    res.append(start)
                    res.reverse()
                    return res, len(visited)
                else:
                    node = son
                    g = 0
                    while node != start:
                      node = fathers[son2str(node)]
                      g+=1
                    heappush(h,(heuristica(son,goal) + g,son))
    print ("Sem Solucao")

start5  = [[0,2,3],[1,5,6],[4,7,8]]
start10 = [[2,3,6],[0,1,5],[4,7,8]]
start15 = [[2,6,5],[1,3,8],[0,4,7]]
start20 = [[5,0,8],[2,1,7],[4,6,3]]
start25 = [[5,8,7],[6,0,1],[2,4,3]]

goal  = [[1,2,3],[4,5,6],[7,8,0]]

qtdmovimentosh1 = [[] for i in range(0,5)]
qtdmovimentosh2 = [[] for i in range(0,5)]

qtdnosh1 = [[] for i in range(0,5)]
qtdnosh2 = [[] for i in range(0,5)]

#5 movimentos h1
(resp, qtdnosh1[0]) = a_star(start5,goal,h1)
qtd = 0
for s in resp:
    qtd+=1
qtdmovimentosh1[0] = qtd

#10 movimentos h1
(resp, qtdnosh1[1]) = a_star(start10,goal,h1)
qtd = 0
for s in resp:
    qtd+=1
qtdmovimentosh1[1] = qtd

#15 movimentos h1
(resp, qtdnosh1[2]) = a_star(start15,goal,h1)
qtd = 0
for s in resp:
    qtd+=1
qtdmovimentosh1[2] = qtd

#20 movimentos h1
(resp, qtdnosh1[3]) = a_star(start20,goal,h1)
qtd = 0
for s in resp:
    qtd+=1
qtdmovimentosh1[3] = qtd

#25 movimentos h1
(resp, qtdnosh1[4]) = a_star(start25,goal,h1)
qtd = 0
for s in resp:
    qtd+=1
qtdmovimentosh1[4] = qtd


#5 movimentos h2
(resp, qtdnosh2[0]) = a_star(start5,goal,h2)
qtd = 0
for s in resp:
    qtd+=1
qtdmovimentosh2[0] = qtd

#10 movimentos h2
(resp, qtdnosh2[1]) = a_star(start10,goal,h2)
qtd = 0
for s in resp:
    qtd+=1
qtdmovimentosh2[1] = qtd

#15 movimentos h2
(resp, qtdnosh2[2]) = a_star(start15,goal,h2)
qtd = 0
for s in resp:
    qtd+=1
qtdmovimentosh2[2] = qtd

#20 movimentos h2
(resp, qtdnosh2[3]) = a_star(start20,goal,h2)
qtd = 0
for s in resp:
    qtd+=1
qtdmovimentosh2[3] = qtd

#25 movimentos h2
(resp, qtdnosh2[4]) = a_star(start25,goal,h2)
qtd = 0
for s in resp:
    qtd+=1
qtdmovimentosh2[4] = qtd

print("Quantidade de movimento h1 ->", qtdmovimentosh1)
print("Quantidade de movimento h2 ->", qtdmovimentosh2)

print("Quantidade de nós visitados h1 ->", qtdnosh1)
print("Quantidade de nós visitados h2 ->", qtdnosh2)

plt.plot(qtdmovimentosh1, qtdnosh1, label='Heurística 01', color='red', marker="8")
plt.plot(qtdmovimentosh2, qtdnosh2, label='Heurística 02', color='purple', marker="8")
plt.xlabel('Número de movimentos')
plt.ylabel('Número de nós visitados')
plt.legend(['h1','h2'])

"""Fim da implementação do exercício 04"""