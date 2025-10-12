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


def f(x):
    return -x * (1+x**2)**2

def g(x):
    return 1 + x**2
#def integration_step(N,x,x_n,D,d,dt,xi,ddzeta):


#_______________________
# Constants

# Size of the grid
N=3

# Dimensions
d=2

# Total number of steps
Lim=1000

# Time step
dt=0.1

# Time 
t=0

# Coefficient for generating IC
c = 0.0001

# The step to print the solution 
N_p=10

#__________________________
# Main 

# Create the system
x=np.zeros((N,N),dtype=float)


#First Order Scheme
#_________________________

# Initialise IC

# Main step
x_n=initial_cond(N,c)

# Create noises

dzeta=initial_cond(N,1)
xi=initial_cond(N,1)


# sq_m and sq_a

sq_m=np.sqrt(xi.var()*dt)

sq_a=np.sqrt(dzeta.var()*dt)

# Boundary condtitions


for i in range(N):
    x_n[N-1][i]=x[1][i]
    x_n[0][i]=x[N-2][i]
    x_n[i][N-1]=x[i][1]
    x_n[i][0]=x[i][N-2]





