# Importing neccesary libraries
import numpy as np 
import matplotlib.pyplot as plt
import time
start_time=time.time()

# Functions 
#_______________________

# Initial Conditions on the system - close to the solution (mean 0)

def initial_cond(N,v=1):
    return np.random.normal(0,v,(N,N))
   
omega=0.1

Amp=0.1

def F(x):
    return Amp * np.cos(omega*x)

# Function f from the defintion
def f(x):
    
    return (-x) * (1+(x)**2)**2

# Function g from the definition 
def g(x):
    return 1 + x**2

# Derivative of g
def der_g(x):
    return 2*x

# Intermediate integrating step for second order scheme
def inter_integration_step(x,x_tmr,D,d,dt,xi,dzeta,sq_m,sq_a,t):
    
    laplacian = (x[2: , 1:-1] + x[:-2 , 1:-1]+ x[1:-1 , 2:] + x[1:-1 , :-2] - d * x[1:-1 , 1:-1])

    x_tmr[1:-1, 1:-1] = x[1:-1, 1:-1] + dt * (f(x[1:-1, 1:-1]) + D/d * laplacian + (xi_var/2) * g(x[1:-1, 1:-1]) * der_g(x[1:-1, 1:-1]) )  + sq_m * g(x[1:-1, 1:-1]) * xi[1:-1, 1:-1] + sq_a * dzeta[1:-1, 1:-1]
   
    return x_tmr

# Main integrating step for second order scheme

def main_integration_step(x,x_n,x_tmr,D,d,dt,sq_m,sq_a,xi,dzeta,t):

    laplacian_x = (x[2:, 1:-1] + x[:-2, 1:-1]+ x[1:-1, 2:] + x[1:-1, :-2] - d * x[1:-1, 1:-1])
    laplacian_tmr = (x_tmr[2:, 1:-1] + x_tmr[:-2 , 1:-1]+ x_tmr[1:-1, 2:] + x_tmr[1:-1 , :-2] - d * x_tmr[1:-1 , 1:-1])

    x_n[1:-1 , 1:-1] = x[1:-1 , 1:-1] +  (f(x[1:-1 , 1:-1])+ D/d * laplacian_x + (xi_var/2) * g(x_tmr[1:-1, 1:-1]) * der_g(x_tmr[1:-1, 1:-1]) + f(x_tmr[1:-1 , 1:-1]) + D/d * laplacian_tmr + (xi_var/2) * g(x[1:-1, 1:-1]) * der_g(x[1:-1, 1:-1]))*dt/2 + sq_m * g(x[1:-1 , 1:-1]) * xi[1:-1 , 1:-1]/2 + sq_m * g(x_tmr[1:-1 , 1:-1])*xi[1:-1 , 1:-1]/2 + sq_a * dzeta[1:-1 , 1:-1]

    return x_n

# Updating system with new t and x after integration
def update_system(t,dt,x_n):
    return (t+dt,x_n)



def run_simulation(N,c,dt,D,d,t,xi_var,dz_var,A,G,Lim):

    # Initate the array for the average field potential 
    m_values=[]

    # Create new system
    x=np.zeros((N,N))
    
    x_n=initial_cond(N)*c

       
    # sq_m and sq_a

    sq_m=np.sqrt(xi_var*dt)
    sq_a=np.sqrt(dz_var*dt)

    for k in range(Lim):
        
        # Create intermediate step

        x_tmr=np.zeros((N,N))

        # Calculating average state of the system
        
        m_values.append(np.mean(x))

    
        # Create noises

        dzeta=initial_cond(N,np.sqrt(dz_var))
        xi=initial_cond(N,np.sqrt(xi_var))


        # Boundary condtitions

        x_tmr[-1,:]=0
        x_tmr[0,:]=0
        x_tmr[:,-1]=0
        x_tmr[:,0]=0

        # checking if it diverged
        if np.any(np.isinf(x)) or np.any(np.isnan(x)):
            return 0
        
        # Predictor step
        
        x_tmr=inter_integration_step(x,x_tmr,D,d,dt,xi,dzeta,sq_m,sq_a,t)

        
        # Boundary conditions for the main step

        x_n[-1,:]=0
        x_n[0,:]=0
        x_n[:,-1]=0
        x_n[:,0]=0

        # checking if it diverged
        if np.any(np.isinf(x)) or np.any(np.isnan(x)):
            return 0

        # Main Step

        x_n = main_integration_step(x,x_n,x_tmr,D,d,dt,sq_m,sq_a,xi,dzeta,t)

        # Updating the system 
        t,x = update_system(t, dt, x_n)

        # checking if it diverged
        if np.any(np.isinf(x)) or np.any(np.isnan(x)):
            return 0, m_values[-1]
        

        if A==1:
            # Printing N_p results while modelling
            if (k+1)/tmp==int((k+1)/tmp):
                print("\n")
                print(f"Step {k+1}")
                print("\n")
                print(x) 
                print("\n")
                
        
        
        del(xi)
        del(dzeta)
    # Plotting the average mean field
    m_values.append(np.mean(x))
    if G==1:
        xs=np.linspace(0,T,len(m_values))
        plt.plot(xs,m_values,label=f"variance of xi is {xi_var} ")

        


    return 1, m_values[-1]

#_______________________
# Constants

# Size of the grid
N=18

# Dimensions
d=2

# Initialise time 
t=0

# Total time

T=10

# Time step
dt=0.0001

# Total number of steps
Lim=int(1/dt)*T

# Number of simulations 

S=1

# Coefficient for generating IC
c = 0.0001

# Auxilary constant for printing
tmp=Lim/T

# print solutions A=1 yes A=0 no
A=0

# Graph solutions G=1 yes G=0 no
G=1



#__________________________
# Main 

#Second Order Scheme
#_________________________

#Space for values for simulation

dzeta_var_values=[0]

xi_var_values=[0,2,4]

# Strengh of the coupling
D_values=[20]



for repetition in range(S):

    final_state=[]
    for dz_var in dzeta_var_values:
       
       # Create the table of convergence, if value is 1 it converged, if it is 0 it diverged
       div=np.zeros((len(D_values),len(xi_var_values)))

       for xi_var, i in zip(xi_var_values,range(len(xi_var_values))):
            for Ds, j in zip(D_values,range(len(D_values))):
                div[j][i] , m = run_simulation(N,c,dt,Ds,d,t,xi_var,dz_var,A,G,Lim)
                final_state.append(abs(round(np.mean(m),2)))
    
    print(div)





# Adjusting the graph
if G==1:
    plt.grid()
    plt.xlim(0,T)
    plt.ylabel("Average field of oscilators")
    plt.xlabel("Time")
    plt.title(f"Simulations for xi_var = {xi_var_values} , dz_var= {dzeta_var_values} and D = {D_values}")
    plt.legend()
    plt.show()
        

plt.plot(xi_var_values,final_state)
plt.grid()
plt.xlim(0,max(xi_var_values))
plt.ylabel("Order parameter")
plt.xlabel("xi variance")
plt.title("Order parameter against variance")
plt.show()

print(f"{round(time.time()-start_time,2)}seconds")