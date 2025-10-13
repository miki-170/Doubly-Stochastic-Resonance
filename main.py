# Importing neccesary libraries
import numpy as np 



# Functions 
#_______________________

# Initial Conditions on the system - close to the solution (mean 0)

def initial_cond(N,c,v=1):
    x_n=np.zeros((N,N),dtype=float)
    for i in range(N):
        for j in range(N):
            x_n[i,j]=c*np.random.normal(0,np.sqrt(v))
    return x_n

# Function f from the defintion
def f(x):
    return (-x) * (1+x**2)**2

# Function g from the definition 
def g(x):
    return 1 + x**2

# Derivative of g
def der_g(x):
    return 2*x

# Intermediate integrating step for second order scheme
def inter_integration_step(N,x,x_n,D,d,dt,xi,dzeta,sq_m,sq_a):

    for i in range(1,N-1):
        for j in range(1,N-1):
            x_n[i,j] = x[i][j] + dt * (f(x[i][j]) + D/(d) * (x[i+1][j] + x[i-1][j]+ x[i][j+1] + x[i][j-1] - d * x[i][j]))  + sq_m * g(x[i][j]) * xi[i][j] + sq_a * dzeta[i][j]
   
    return x_n

# Main integrating step for second order scheme

def main_integration_step(N,x,x_n,x_tmr,D,d,dt,sq_m,sq_a,xi,dzeta):

    for i in range(1,N-1):
        for j in range(1,N-1):
            x_n[i][j] = x[i][j]+( f(x[i][j])+ D/d * (x[i+1][j] + x[i-1][j]+ x[i][j+1] + x[i][j-1] - d* x[i][j]) +f(x_tmr[i][j] + D/d * (x_tmr[i+1][j] + x_tmr[i-1][j]+ x_tmr[i][j+1] + x_tmr[i][j-1] - d* x_tmr[i][j])) )*dt/2 + sq_m * g(x[i][j]) * xi[i][j]/2 + sq_m * g(x_tmr[i][j])*xi[i][j]/2 + sq_a * dzeta[i][j]

    return x_n

# Updating system with new t and x after integration
def update_system(t,dt,x_n):
    return (t+dt,x_n)

# Average of the function: 
def av_state(x):
    return np.mean(x)


#_______________________
# Constants

# Size of the grid
N=5

# Strengh of the coupling
D=20

# Dimensions
d=2

# Total number of steps
Lim=10

# Initialise time 
t=0

# Time step
dt=0.01

# Coefficient for generating IC
c = 0.0001

# The number of printable steps
N_p=10

# Auxilary constant for printing
tmp=Lim/N_p
# print solutions A=1 yes A=0 no
A=0

#__________________________
# Main 

#First Order Scheme
#_________________________

#Space for values for simulation


dzeta_var_values=[3]

xi_var_values=[5]

m_final=[]



# Create the system
x=np.zeros((N,N),np.double)

m_values=[]



for k in range(Lim):
    
    # Create new system
    if k==0:
        x=initial_cond(N,c)
    
    x_n=np.zeros((N,N),dtype=np.double)

    # Create intermediate step

    x_tmr=np.zeros((N,N),dtype=np.double)

    # Create noises

    dzeta=initial_cond(N,1)
    xi=initial_cond(N,1)

    # sq_m and sq_a

    sq_m=np.sqrt(xi.var()*dt)

    sq_a=np.sqrt(dzeta.var()*dt)

    # Boundary condtitions

    
    x_tmr[N-1][:]=x[1][:]
    x_tmr[0][:]=x[N-2][:]
    x_tmr[:][N-1]=x[:][1]
    x_tmr[:][0]=x[:][N-2]

    # Predictor step
    
    x_tmr=inter_integration_step(N,x,x_tmr,D,d,dt,xi,dzeta,sq_m,sq_a)

    
    # Boundary conditions for the main step

    x_n[N-1][:]=x_tmr[1][:]
    x_n[0][:]=x_tmr[N-2][:]
    x_n[:][N-1]=x_tmr[:][1]
    x_n[:][0]=x_tmr[:][N-2]


    # Main Step

    x_n = main_integration_step(N,x,x_n,x_tmr,D,d,dt,sq_m,sq_a,xi,dzeta)


    # Updating the system 
    t,x = update_system(t, dt, x_n)

    
    if A==1:
        # Printing N_p results while modelling
        if (k+1)/tmp==int((k+1)/tmp):
            print("\n")
            print(f"Step {k+1}")
            print("\n")
            print(x) 
            print("\n")

    # Calculating average state of the system
    m_instant = av_state(x)

    m_values.append(m_instant)
    del(x_n)


m_average=np.mean(np.abs(m_values))

print(x)





        
