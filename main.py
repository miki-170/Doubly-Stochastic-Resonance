# Importing neccesary libraries
import numpy as np 
import matplotlib.pyplot as plt



# Functions 
#_______________________

# Initial Conditions on the system - close to the solution (mean 0)

def initial_cond(N,v=2):
    return np.random.normal(0,v,(N,N))
   

# Function f from the defintion
def f(x):
    
    return (-x) * (1+(x + 1e-8)**2)**2

# Function g from the definition 
def g(x):
    return 1 + x**2

# Derivative of g
def der_g(x):
    return 2*x

# Intermediate integrating step for second order scheme
def inter_integration_step(N,x,x_tmr,D,d,dt,xi,dzeta,sq_m,sq_a):

    for i in range(1,N-1):
        for j in range(1,N-1):
            x_tmr[i,j] = x[i][j] + dt * (f(x[i][j]) + D/(d) * (x[i+1][j] + x[i-1][j]+ x[i][j+1] + x[i][j-1] - d * x[i][j]))  + sq_m * g(x[i][j]) * xi[i][j] + sq_a * dzeta[i][j]
   
    return x_tmr

# Main integrating step for second order scheme

def main_integration_step(N,x,x_n,x_tmr,D,d,dt,sq_m,sq_a,xi,dzeta):

    for i in range(1,N-1):
        for j in range(1,N-1):
            x_n[i][j] = x[i][j]+( f(x[i][j])+ D/d * (x[i+1][j] + x[i-1][j]+ x[i][j+1] + x[i][j-1] - d* x[i][j]) +f(x_tmr[i][j] + D/d * (x_tmr[i+1][j] + x_tmr[i-1][j]+ x_tmr[i][j+1] + x_tmr[i][j-1] - d* x_tmr[i][j])) )*dt/2 + sq_m * g(x[i][j]) * xi[i][j]/2 + sq_m * g(x_tmr[i][j])*xi[i][j]/2 + sq_a * dzeta[i][j]

    return x_n

# Updating system with new t and x after integration
def update_system(t,dt,x_n):
    return (t+dt,x_n)




#_______________________
# Constants

# Size of the grid
N=10

# Strengh of the coupling
D=20

# Dimensions
d=2

# Initialise time 
t=0

# Total time

T=10

# Time step
dt=0.00001

# Total number of steps
Lim=int(1/dt)

# Number of simulations 

S=1


# Coefficient for generating IC
c = 0.0001

# Auxilary constant for printing
tmp=Lim/T

# print solutions A=1 yes A=0 no
A=0

#__________________________
# Main 

#Second Order Scheme
#_________________________

#Space for values for simulation


dzeta_var_values=[3]

xi_var_values=[5]

m_final=[]

for repetition in range(S):
    m_values=[]



    for k in range(Lim):
        
        # Create new system
        if k==0:
            x=np.zeros((N,N))
            x_n=initial_cond(N)*c
        

        # Calculating average state of the system
        
        m_values.append(np.mean(x))

        # Create intermediate step

        x_tmr=np.zeros((N,N))

        # Create noises

        dzeta=initial_cond(N,1)*np.sqrt(dt)
        xi=initial_cond(N,1)*np.sqrt(dt)

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
        
        
        del(x_tmr)

    m_values.append(np.mean(x))
    xs=np.linspace(0,T,len(m_values))


    plt.plot(xs,m_values)


# Adjusting the graph
plt.grid()
plt.xlim(0,10)
plt.ylabel("Average field of oscilators")
plt.xlabel("Time")


plt.show()






        
