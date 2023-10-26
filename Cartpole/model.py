import numpy as np
from functools import partial
from scipy.integrate import solve_ivp

class Cartpole():
    def __init__(self):

        self.dim_n=4
        self.dim_m=1
        self.x=np.zeros((self.dim_n))

    def integrate(self,u,t_interval):
        dx_dt=partial(self.dynamics, u=u)
        sol=solve_ivp(dx_dt,t_interval, self.x, method='RK45', t_eval=None,rtol=1e-7, atol=1e-7, dense_output=False, events=None, vectorized=False)
        self.x=sol.y[...,-1]
        #return partial(self.dynamics, u=u)
        
    def dynamics(self,t,x,u):
        m=0.1
        M=1
        L=0.8
        g=9.8
        theta=x[0]
        omega=x[1]
        x1=x[2]
        x2=x[3]

        theta_d=omega
        omega_d=(-u*np.cos(theta)-m*L*omega**2*np.sin(theta)*np.cos(theta)+(M+m)*g*np.sin(theta))/(L*(M+m*(np.sin(theta)**2)))
        x1_d=x2
        x2_d=(u+m*np.sin(theta)*(L*omega**2-g*np.cos(theta)))/(M+m*np.sin(theta)**2)

        return [theta_d,omega_d,x1_d,x2_d]

    def Read_sensor(self):
        return self.x#self.x[0]

    def randomly_initialize(self):
        self.x = np.random.uniform(low=-0.05, high=0.05, size=(self.dim_n,))
        return self.x
