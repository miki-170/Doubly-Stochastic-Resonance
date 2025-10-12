# Importing neccesary libraries
import numpy as np 



# Functions 
#_______________________

# Initial Conditions on the system - close to the solution (mean 0)

def initial_cond(N,c):
    x_n=np.zeros((N,N),dtype=float)
    for i in range(N):
        for j in range(N):
            x_n[i,j]=c*np.random.normal(0)
    return x_n

#_______________________
# Constants

# Size of the grid
N=3

# Dimensions
d=2

# Total number of steps
Lim=1000

# Time step
dt=0

# Time 
t=0

# Coefficient for generating IC
c = 0.0001


print(initial_cond(N,c))


