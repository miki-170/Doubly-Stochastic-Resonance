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

def der_g(x):
    return 2*x

def integration_step(N,x,x_n,D,d,dt,xi,dzeta,sq_m,sq_a):
    xi_var=xi.var()
    
    for i in range(1,N-1):
        for j in range(1,N-1):
            x_n[i,j] = x[i][j] + dt * (f(x[i][j]) + D/d * (x[i+1][j] + x[i-1][j]+ x[i][j+1] + x[i][j-1] - d * x[i][j]) + xi_var/2 * g(x[i][j]) * der_g(x[i][j])) + sq_m * g(x[i][j]) * xi[i][j] + sq_a * dzeta[i][j]
    
    return x_n

def update_system(t,dt,x_n):
    return (t+dt,x_n)

#_______________________
# Constants

# Size of the grid
N=3

# Strengh of the coupling
D=1

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



x_n=integration_step(N,x,x_n,D,d,dt,xi,dzeta,sq_m,sq_a)


t,x = update_system(t,dt,x_n)

print(t)
