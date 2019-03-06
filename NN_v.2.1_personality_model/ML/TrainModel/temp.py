import numpy as np 
from operator import add


a=[np.random.random(5),2,3]
b=[np.random.random(5),4,6]

d=list(map(add,b,a))
print(a)
print(b)
print(d)
