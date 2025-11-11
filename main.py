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

def F(t):
    return Amp * np.cos(omega*t)

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
def inter_integration_step(x,x_tmr,D,d,dt,xi,dzeta,sq_m,sq_a,t,xi_var):
    

    # implented calculating the Laplacian over the whole grid
    laplacian = (np.roll(x,1,axis=0) + np.roll(x,-1,axis=0)+ np.roll(x,1,axis=1) + np.roll(x,-1,axis=1)  - 2*d * x)

    x_tmr = x + dt * (f(x) + D/(2*d) * laplacian + F(t))  + sq_m * g(x) * xi+ sq_a * dzeta
   
    return x_tmr

# Main integrating step for second order scheme

def main_integration_step(x,x_n,x_tmr,D,d,dt,sq_m,sq_a,xi,dzeta,t,xi_var):

    
    # implented calculating the Laplacian over the whole grid
    laplacian_x = (np.roll(x,1,axis=0) + np.roll(x,-1,axis=0)+ np.roll(x,1,axis=1) + np.roll(x,-1,axis=1)  - (2*d)* x)
    laplacian_tmr = (np.roll(x_tmr,1,axis=0) + np.roll(x_tmr,-1,axis=0)+ np.roll(x_tmr,1,axis=1) + np.roll(x_tmr,-1,axis=1)  - (2*d) * x_tmr)

    x_n = x +  (f(x)+ D/(2*d) * laplacian_x +  f(x_tmr) + D/(2*d) * laplacian_tmr + F(t) + F(t+dt))*dt/2 + sq_m * g(x) * xi/2 + sq_m * g(x_tmr)*xi/2 + sq_a * dzeta

    return x_n


# Updating system with new t and x after integration
def update_system(t,dt,x_n):
    return (t+dt,x_n)


def run_simulation_second_order(N,c,dt,D,d,t,xi_var,dz_var,A,G,Lim):

    # Initate the array for the average field potential 
    m_values=[]

    # Create new system
    x = initial_cond(N, v=1) * c   # initial field, small random around 0
    
    x_n = np.zeros((N,N))

    x_tmr= np.zeros((N,N))
       
    # sq_m and sq_a

    sq_m=np.sqrt(xi_var*dt)
    sq_a=np.sqrt(dz_var*dt)

    for k in range(Lim):
    
        # Create noises

        dzeta=initial_cond(N)
        xi=initial_cond(N)
    

        # checking if it diverged
        if np.any(np.isinf(x)) or np.any(np.isnan(x)):
            return 0, 0
        

        
        # Predictor step
        
        x_tmr=inter_integration_step(x,x_tmr,D,d,dt,xi,dzeta,sq_m,sq_a,t,xi_var)
        
        # checking if it diverged
        if np.any(np.isinf(x)) or np.any(np.isnan(x)):
            return 0 ,0

        # Main Step

        x_n = main_integration_step(x,x_n,x_tmr,D,d,dt,sq_m,sq_a,xi,dzeta,t,xi_var)
        
        

        # Updating the system 
        t,x = update_system(t, dt, x_n)

       

        # Calculating average state of the system
        
        m_values.append(np.mean(x))

        # checking if it diverged
        if np.any(np.isinf(x)) or np.any(np.isnan(x)):
            return 0, 0
        

        if A==1:
            # Printing N_p results while modelling
            if (k+1)/tmp==int((k+1)/tmp):
                print("\n")
                print(f"Step {k+1}")
                print("\n")
                #print(x) 
                print("\n")
                
        
        
        del(xi)
        del(dzeta)
        
    # Plotting the average mean field
    m_values.append(np.mean(x))
    if G==1:
        xs=np.linspace(0,T,len(m_values))
        plt.plot(xs,m_values)
        plt.grid()
        plt.xlim(0,T)
        plt.ylim(-1,1)
        plt.ylabel("Average field of oscilators")
        plt.xlabel("Time")
        plt.title(f"Simulations for xi_var = {xi_var} , dz_var= {dz_var} and D = {D}")
        #plt.savefig(f'Graphs/xi_var-equal-to-{xi_var}.png')
        plt.show()
        plt.clf()



    # Excluding the time before the system converges to a steady state
    m_arr = np.array(m_values) # your recorded mean-field trace
    dr=0.4# drop first 40 % of time units
    #dr/=T
    transient = int(dr * len(m_arr))   
    steady_m_abs = np.mean(np.abs(m_arr[transient:]))
    print("Order parameter (⟨|m|⟩ after transient) = ", steady_m_abs)
    del(x)
    return 1, steady_m_abs




        

#_______________________
# Constants

# Size of the grid
N=50

# Dimensions
d=2

# Initialise time 
t_init=0

# Total time

T=100

# Time step
dt=2.5*10**(-4)

# Total number of steps
Lim=int(T/dt)

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

dzeta_var_values=[0.01,1.05,5]

xi_var_values=[3]

# Strengh of the coupling
D_values=[20]



for repetition in range(S):

    
    for dz_var in dzeta_var_values:
        final_state=[]
        # Create the table of convergence, if value is 1 it converged, if it is 0 it diverged
        div=np.zeros((len(D_values),len(xi_var_values)))
        for xi_var, i in zip(xi_var_values,range(len(xi_var_values))):
            for Ds, j in zip(D_values,range(len(D_values))):
                div[j][i] , m = run_simulation_second_order(N,c,dt,Ds,d,t_init,xi_var,dz_var,A,G,Lim)
                final_state.append(abs(round(np.mean(m),2)))
        plt.clf()
        #plt.plot(xi_var_values,final_state,label=f"dz_var = {dz_var}")
        
        print(div)
        
print(f"{round(time.time()-start_time,2)}seconds")


"""plt.grid()
plt.xlim(0,max(xi_var_values))
plt.ylim(0,1)
plt.ylabel("Order parameter")
plt.xlabel("xi variance")
plt.title("Order parameter against variance")
plt.legend()
plt.show()"""