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
    return -x * (1+x**2)**2

# Function g from the definition 
def g(x):
    return 1 + x**2

# Derivative of g
def der_g(x):
    return 2*x

# Integration step for first order scheme
def integration_step(N,x,x_n,D,d,dt,xi,dzeta,sq_m,sq_a):
    xi_var=xi.var()
    
    for i in range(1,N-1):
        for j in range(1,N-1):
            x_n[i,j] = x[i][j] + dt * (f(x[i][j]) + D/d * (x[i+1][j] + x[i-1][j]+ x[i][j+1] + x[i][j-1] - d * x[i][j]) + xi_var/2 * g(x[i][j]) * der_g(x[i][j])) + sq_m * g(x[i][j]) * xi[i][j] + sq_a * dzeta[i][j]
    
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
N=4

# Strengh of the coupling
D=20

# Dimensions
d=2

# Total number of steps
Lim=1000

# Initialise time 
t=0

# Time step
dt=0.001

# Coefficient for generating IC
c = 0.0001

# The number of printable steps
N_p=10

# Auxilary constant for printing
tmp=Lim/N_p


#__________________________
# Main 




#First Order Scheme
#_________________________

#Space for values for simulation


dzeta_var_values=[3]

xi_var_values=[2]

m_final=[]

for l in range(1):
    dzes=3
    for xis in xi_var_values:

        # Create the system
        x=np.zeros((N,N),dtype=float)

        m_values=[]
        # Main step
        x_n=initial_cond(N,c)

        for k in range(Lim):

            

            # Create noises

            dzeta=initial_cond(N,1,dzes)
            xi=initial_cond(N,1,xis)


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

        m_final.append((xis,m_average))

print(m_final)



        
