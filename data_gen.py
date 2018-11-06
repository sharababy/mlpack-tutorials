import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

size = 400

def circle(x,r):
	return math.sqrt(r**2-x**2)

X = np.random.uniform(low=-3.0,high=3.0,size=size)
Y = []
O = np.random.randint(low=0,high=2,size=size) # [low,high)

for x,o in zip(X,O): 
	if o == 1 :
		Y.append(circle(x,3.5))		
	else: 
		Y.append(circle(x,4.0))		


data = []

for x,y,o in zip(X,Y,O):

	data.append([x,y,int(o)])

plt.scatter(X, Y, c=O)
plt.show()

np.savetxt("foo.csv", data, delimiter=",")
