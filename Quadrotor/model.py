import math
import numpy as np
# import utils
from functools import partial
from scipy.integrate import solve_ivp

def SINDy(dXdt,Theta,lam=0.1):
    W = _reg_(Theta,dXdt) # initial guess: Least-squares
    for k in range(20):
        #find small coefficients
        smallinds = np.absolute(W)<lam
        # threshold
        W[smallinds]=0
        for ind in range(dXdt.shape[0]):
            # n is state dimension
            biginds =np.invert(smallinds[ind,...])
            #Regress dynamics onto remaining terms to find sparse Xi
            #print(np.where(biginds)[0])
            #print(biginds)
            W[ind,biginds]= _reg_(Theta[np.where(biginds)[0].T,...],dXdt[ind,...])
    return W

def identification_RLS(W,S,dXdt,Theta,forget_factor=0.9):
    g=S*Theta*(1/(forget_factor+Theta.T*S*Theta))

    W=W+(dXdt-W*Theta)*g.T
    S=(1/forget_factor)*(S-g*Theta.T*S)
    # %omega_tilda1=omega+S*Theta*(dXdt-omega'* Theta)';
    # %S_tilda=1/(lambda+Theta'*S*Theta) *(S-S*Theta*Theta'*S);
    return [W,S]


def _reg_(a, b):
    return np.linalg.lstsq(a.T, b.T,rcond=None)[0].T

#convert from real domain to [-pi,pi]
def rad_regu(angle):
    angle=angle%(2*np.pi)
    if angle>np.pi:
        angle=angle-2*np.pi

    return angle
def x_dot_approx(X_buf,h):
    return (X_buf[:,2]-X_buf[:,1])/(h)#double_pend_cart(0, X_buf[...,2], U[i-2],u_lim)# #double_pend_cart(0, X_buf[...,2], U[i-2],u_lim)#    (X_buf[...,0]-8*X_buf[...,1]+8*X_buf[...,3]-X_buf[...,4])/(12*h)  #(X_buf[...,0]-8*X_buf[...,1]+8*X_buf[...,3]-X_buf[...,4])/(12*h) #X_dict[...,m_dict-2]-4*X_dict[...,m_dict-1]+3*X_dict[...,m_dict] (-X_dict[...,m_dict-1]+X_dict[...,m_dict])/(h)
    #return (X_buf[:,0]-8*X_buf[:,1]+8*X_buf[:,3]-X_buf[:,4])/(12*h)

def rot_to_euler(R):
    # %gamma   beta   alpha
    # % x       y      z
    # %Roll   Pitch   Yaw
    # %Phi    Theta   Sai
    psi=math.atan2(R[1,0],R[0,0])*180/np.pi
    theta=math.atan2(-R[2,0],math.sqrt(R[2,1]**2+R[2,2]**2))*180/np.pi
    phi=math.atan2(R[2,1],R[2,2])*180/np.pi

    return [phi, theta, psi]

def RLS(x_new,y_new,W,P):
    lam=0.99
    g = np.matmul(P,x_new)/(np.matmul(np.matmul(x_new,P),x_new)+lam)
    W = W +np.outer((y_new-np.matmul(W,x_new)),g)
    P = (P-np.outer(g,np.matmul(x_new,P)))/lam
    return W,P

def Gradient_descent(x_new,y_new,W, learning_rate):
    N = len(y_new)
    for i in range(N):
        Error_W_deriv = -2*x_new * (y_new[i] - np.inner(W[i,...],x_new ))

    # We subtract because the derivatives point in direction of steepest ascent
        W[i,...]-= Error_W_deriv * learning_rate

    return W

class Quadrotor():
    def __init__(self,m=0.33,Ixx=1.395e-05, Iyy=1.436e-05, Izz=2.173e-05, g=9.81):
        self.m=m
        self.Ixx=Ixx
        self.Iyy=Iyy
        self.Izz=Izz
        self.g=g
        self.dim_n=18
        self.dim_m=4
        self.measure_dim=12
        self.x=np.zeros((self.dim_n))
        self.x[9]=1
        self.x[13]=1
        self.x[17]=1
        self.x_dot=np.zeros((self.dim_n))
        self.measure=np.zeros((self.measure_dim))

    def integrate(self,u,t_interval):
        dx_dt=partial(self.dynamics, u=u)
        sol=solve_ivp(dx_dt,t_interval, self.x, method='RK45', t_eval=None,rtol=1e-6, atol=1e-6, dense_output=False, events=None, vectorized=False)
        self.x=sol.y[...,-1]
        #return partial(self.dynamics, u=u)
    
    def dynamics(self,t,x,u):
        m=0.033
        Ftav=self.pwm_to_Ftav(u+np.array([42000,42000,42000,42000]))
        F=[0,0,Ftav[0]]
        tav=Ftav[1:]
        #x,y,z=x[:3]
        uvw=x[3:6]
        pqr=x[6:9]
        p,q,r=x[6:9]
        R=x[9:].reshape((3,3))
        Q = np.array([[0,-r,q],
                      [r,0,-p],
                      [-q,p,0]])
        g_e3 = np.array([0,0,self.g])
        J = np.diag([self.Ixx, self.Iyy, self.Izz])
        #linear velocity
        xyz_dot=uvw
        #linear accelration
        uvw_dot=-g_e3+(1/m)*np.matmul(R,F)
        #rotational accelration
        pqr_dot = np.matmul(np.linalg.inv(J),(tav-np.cross(pqr,np.matmul(J,pqr))))
        #rotational matrix
        R_dot=np.matmul(R,Q).flatten()
        self.x_dot[:3]=xyz_dot
        self.x_dot[3:6]=uvw_dot
        self.x_dot[6:9]=pqr_dot #angular velocity_dot
        self.x_dot[9:]=R_dot
        return self.x_dot
    
    def pwm_to_Ftav(self,pwm):
        pwm1,pwm2,pwm3,pwm4=pwm
        gain=0.2685
        bias=4070.3
        w1=pwm1*gain+bias
        w2=pwm2*gain+bias
        w3=pwm3*gain+bias
        w4=pwm4*gain+bias
        d=39.73e-3
        CT=3.1582e-10
        CD=7.9379e-12
        F=CT*(w1**2+w2**2+w3**2+w4**2)
        tav_x=CT*d*(w4**2-w2**2)
        tav_y=CT*d*(w3**2-w1**2)
        tav_z=CD*(w4**2+w2**2-w3**2-w1**2)
        return [F,tav_x,tav_y,tav_z]
    
    def Read_sensor(self):
        self.measure[:9]= self.x[:9]
        self.measure[9:]=rot_to_euler(self.x[9:].reshape(3,3))
        return self.measure
    
    def Read_sensor_with_noise(self,sig):
        self.measure[:9]= self.x[:9]
        self.measure[9:]=rot_to_euler(self.x[9:].reshape(3,3))
        self.measure=(np.random.normal(0, sig, self.measure_dim)+np.ones((self.measure_dim)))*self.measure
        return self.measure
    
    def randomly_initialize(self):
        self.x=np.zeros((self.dim_n))
        self.x[9]=1
        self.x[13]=1
        self.x[17]=1
        self.x_dot=np.zeros((self.dim_n))
        self.x[:3]=np.random.uniform(low=0, high=1, size=(3))
        return self.Read_sensor()
