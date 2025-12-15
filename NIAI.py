# Importing neccesary libraries
import numpy as np 
import matplotlib.pyplot as plt
import time
from scipy.ndimage import zoom
#Get the zoom function from Scipy !!!!!

start_time=time.time()


# Create the constants for astrocytes

a=1
b=2
c=0.2
d=0.075
g=0.2
epsilon=0.01
D_u=100
D_v=100
c_0=2
c_1=0.185
v_1, v_2, v_3, v_4,v_5, v_6=6,0.11, 2.2,0.3,0.025,0.2
k_1, k_2, k_3, k_4=0.5, 1 ,0.1 , 1.1
d_1, d_2, d_3,d_5=0.13,1.049,0.9434,0.082
IP_3s=0.16
tau_r=0.14
alpha=0.8
a_2=0.14



# Functions for the astrocytes layer

def Li_Rinzel_derivatives(IP3,Ca,h,I_neuro):

    # Creating the auxilary parameters
    
    I_er = c_1 * v_1 * ((IP3 / (IP3 + d_1))**3) * ((Ca / (Ca + d_5))**3) * (h**3) * ((c_0 - Ca) / c_1 - Ca)

    I_leak = c_1 * v_2 * ((c_0 - Ca) / c_1 - Ca)

    I_pump = v_3 * (Ca**2) / (Ca**2 + k_3**2)
    
    I_in = v_5 + v_6 * (IP3**2) / (IP3**2 + k_2**2)

    I_out = k_1 * Ca

    # For clarity
    Q= d_2 * (IP3 + d_1) / (IP3 + d_3)

    H = (Q) / (Q + Ca)
    
    tau_n = 1/(a_2 * (Q + Ca))

    I_plc = v_4 * (Ca + (1 - alpha) * k_4) / (Ca + k_4)

    # Calculating the differences
    dCa = I_er - I_pump + I_leak + I_in - I_out
    
    dIP3 = (IP_3s - IP3) *tau_r + I_plc + I_neuro

    dh = (H - h) / tau_n

    return dCa , dIP3 , dh

# Integrating function for the oscillators

def initial_cond(N,v=1):
    return np.random.normal(0,v,(N,N))
   
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
def inter_integration_step(x,x_tmr,D,d,dt,xi,dzeta,sq_m,sq_a,t,xi_var, I_total):
    

    # implented calculating the Laplacian over the whole grid
    laplacian = (np.roll(x,1,axis=0) + np.roll(x,-1,axis=0)+ np.roll(x,1,axis=1) + np.roll(x,-1,axis=1)  - 2*d * x)

    x_tmr = x + dt * (f(x) + D/(2*d) * laplacian + F(t) + I_total)  + sq_m * g(x) * xi+ sq_a * dzeta
   
    return x_tmr

# Main integrating step for second order scheme

def main_integration_step(x,x_n,x_tmr,D,d,dt,sq_m,sq_a,xi,dzeta,t,xi_var, I_total):

    
    # implented calculating the Laplacian over the whole grid
    laplacian_x = (np.roll(x,1,axis=0) + np.roll(x,-1,axis=0)+ np.roll(x,1,axis=1) + np.roll(x,-1,axis=1)  - (2*d)* x)
    laplacian_tmr = (np.roll(x_tmr,1,axis=0) + np.roll(x_tmr,-1,axis=0)+ np.roll(x_tmr,1,axis=1) + np.roll(x_tmr,-1,axis=1)  - (2*d) * x_tmr)

    x_n = x +  (f(x)+ D/(2*d) * laplacian_x +  f(x_tmr) + D/(2*d) * laplacian_tmr + F(t) + F(t+dt) + 2 * I_total)*dt/2 + sq_m * g(x) * xi/2 + sq_m * g(x_tmr)*xi/2 + sq_a * dzeta

    return x_n

# Updating system with new t and x after integration
def update_system(t,dt,x_n):
    return (t+dt,x_n)

def normal(Ca):
    return Ca/np.mean(Ca)-1

# MAIN FUNCTION

def run_simulation_second_order(N,dt,D,d,t,xi_var,dz_var,G,Lim):

    # Initate the array for the average potential 
    m_values=[]
    Ca_values=[]
    IP3_values=[]
    

    
    # Create new system
    x = initial_cond(N, v=1) * scalar   # initial field, small random around 0

    # Astrocytes (43x43)
    Ca = np.zeros((N_2,N_2))+0.1
    IP3 = np.zeros((N_2,N_2))+0.1
    h = np.zeros((N_2,N_2))+0.1

    
    x_n = np.zeros((N,N))
    x_tmr= np.zeros((N,N))
       
    # sq_m and sq_a

    sq_m=np.sqrt(xi_var*dt)
    sq_a=np.sqrt(dz_var*dt)

    # Create I inpuptn pattern:

    I_pattern=np.zeros((N,N))
    I_pattern[20:180, 90:110] = 1.0
    #I_pattern[60:70, 20:110] = 1.0


    # Shwoing input pattern
    plt.imshow(I_pattern,vmin=0)
    #plt.show()
    plt.close()

    # Factors for coupling the layers

    # Neuron to Astrocyte
    zoom_in = N_2 / N_1
    # Astrocyte to Neuron
    zoom_out = N_1 / N_2

    for k in range(Lim):
        
        # --- 1. Calculate current ---
       
        # matrix of average states with coupling terms

        neuron_active_mask = (x > neuron_threshold).astype(float)
        
        # Downscale to Astrocyte grid (Mean pooling approximation via zoom)
        neuron_activity_downscaled = zoom(neuron_active_mask, zoom_in, order=1)
        I_neuro_input = np.where(neuron_activity_downscaled > 0.5, I_neuro_coupling, 0.0)

        
        # Integrating Astrocytes
    
        dCa , dIP3, dh= Li_Rinzel_derivatives(IP3,Ca, h, I_neuro_input)
        Ca += dCa * dt
        IP3 += dIP3 * dt
        h += dh * dt

        Ca_values.append(np.mean(Ca))
        IP3_values.append(np.mean(IP3))
        
        
        # activity check

        astro_active = np.where(Ca>threshold, I_astro_strength,0)
        I_astro_field = zoom(astro_active, zoom_out , order=1)


        if k<int(1/dt)*10:
            current_app = I_app* I_pattern
        else:
            current_app = 0
        
        I_total = I_astro_field + current_app
        
        if k%int(T/dt/5)==0:

            fig , ( ax1, ax2, ax3 )= plt.subplots(1,3,figsize=(12, 3))
            fig.subplots_adjust(left=0.03,right= 0.95, bottom=0.1, top=0.9, wspace=0.1)
            
            boundary =np.max(Ca)

            for ax,data, limit in zip([ax1,ax2,ax3],[I_neuro_input,I_total,Ca],[I_neuro_coupling,I_astro_strength+I_app,boundary]):
                im = ax.imshow(np.round(data,2),vmin=0,vmax = limit)
                
                fig.colorbar(im, ax=ax,format='%.2f')
            
            ax1.set_title("Neuronal response to astrocytes")

            ax2.set_title("Total current passed onto neurons")

            ax3.set_title("Ca")

            print(f"Graph {int(k/int(T/dt/5))} ")
            plt.savefig(f"xi_var_{xi_var}/evolution{int(k/int(T/dt/5))}.png")
            plt.close(fig)

        # --- 2. Integrating neurons ---
        # Create noises

        dzeta=initial_cond(N)
        xi=initial_cond(N)
    

        # checking if it diverged
        if np.any(np.isinf(x)) or np.any(np.isnan(x)):
            return 0, 0
        
        # Predictor step
        x_tmr=inter_integration_step(x,x_tmr,D,d,dt,xi,dzeta,sq_m,sq_a,t,xi_var, I_total)

        # Main Step
        x_n = main_integration_step(x,x_n,x_tmr,D,d,dt,sq_m,sq_a,xi,dzeta,t,xi_var, I_total)

        # Updating the system 
        t,x = update_system(t, dt, x_n)

        # Calculating average state of the system
        
        
        m_values.append(np.mean(x))
            
    # Plotting the average mean field
    if G==1:

        xs=np.linspace(0,T,len(m_values))


        # Plots of Calcium paths
        plt.plot(xs,Ca_values,label='Ca')
        plt.plot(xs,IP3_values,label='IP3')
        plt.plot(xs,m_values,label='Oscillators average')

        # max for Graphs
        Lower_lim = min(min(Ca_values),min(IP3_values),min(m_values))
        Upper_lim = max(max(Ca_values),max(IP3_values),max(m_values))


        plt.legend()
        plt.grid()
        plt.xlim(0,T)
        plt.ylim(Lower_lim-0.1,Upper_lim+0.1)
        plt.xlabel("Time")
        plt.title(f"Simulations for xi_var = {xi_var} , dz_var= {dz_var} and D = {D}")
        plt.savefig(f"xi_var_{xi_var}/Oscillations_and_Ca_behaviour.png")
        plt.close()

        # Plotting the Ca layer
        plt.imshow(Ca)
        plt.colorbar()
        plt.xlim(left=0)
        plt.savefig(f"xi_var_{xi_var}/Calcium final state.png")
        plt.close()




    # Excluding the time before the system converges to a steady state
    m_arr = np.array(m_values) # your recorded mean-field trace
    dr=0.4# drop first 40 % of time units
    #dr/=T
    transient = int(dr * len(m_arr))   
    steady_m_abs = np.mean(np.abs(m_arr[transient:]))
    print("Order parameter (⟨|m|⟩ after transient) = ", steady_m_abs)
    
    return 1, steady_m_abs



#_______________________
# Parameters for the simulation

# For function F
omega=0
Amp=0

# Size of the grid
N_1=202 #Neurons
N_2=67 #Astrocytes

# Dimensions
d=2

# Initialise time 
t_init=0

# Total time

T=1

# Time step
dt=10**(-3)

# Total number of steps
Lim=int(T/dt)

# Coefficient for generating IC
scalar= 0.00001

# Auxilary constant for printing
tmp=Lim/T

# Graph paths G=1 yes G=0 no
G=1

# Graph m dependence
m_Graph=0

# Values of noise for simulation

dzeta_var_values=[0]

xi_var_values=[0]

# Strengh of the coupling
D_values=[20]

# Intensity for the current input to the model

I_app = 0.5 # Strenth of pattern input
I_astro_strength= 0.3 # Input from astrocytes after threshold
threshold=0.1 # For astrocytes values acting on neurons
I_neuro_coupling = 0.05 # Value of neurons firing back if 50% is more active 
neuron_threshold = 0.25 # Value of neuronal excitability

 




for dz_var in dzeta_var_values:
        final_state=[]
        # Create the table of convergence, if value is 1 it converged, if it is 0 it diverged
        div=np.zeros((len(D_values),len(xi_var_values)))
        for xi_var, i in zip(xi_var_values,range(len(xi_var_values))):
            for Ds, j in zip(D_values,range(len(D_values))):
                div[j][i] , m = run_simulation_second_order(N_1,dt,Ds,d,t_init,xi_var,dz_var,G,Lim)
                final_state.append(abs(round(np.mean(m),2)))
        plt.close()
        print(div)

print(f"{time.time()-start_time} seconds")




if m_Graph==1:
    plt.plot(xi_var_values,final_state,label=f"dz_var = {dz_var}")
    plt.grid()
    plt.xlim(0,max(xi_var_values))
    plt.ylim(0)
    plt.ylabel("Order parameter")
    plt.xlabel("xi variance")
    plt.title("Order parameter against variance")
    plt.legend()
    plt.savefig("Noise infulence.png")